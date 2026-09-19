// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * SQLite-backed Store implementation using better-sqlite3.
 *
 * Schema:
 *   documents  — ingested source documents with metadata
 *   chunks     — tokenized sub-documents for retrieval
 *   chunks_fts — FTS5 virtual table for BM25 full-text search
 *   embeddings — vector representations of chunks (stored as JSON)
 *
 * Design decisions:
 *   - better-sqlite3 is synchronous; we wrap in async for interface conformance.
 *   - Transactions are used for batch writes (chunks + FTS, embeddings) to
 *     ensure atomicity and dramatically improve write throughput.
 *   - Dense search loads all vectors into memory. This is acceptable for
 *     <100K chunks. For larger datasets, migrate to pgvector or qdrant.
 *   - FTS5 rank is already BM25-scored (lower = better match); we negate
 *     so that higher scores = better relevance in SearchResult.
 */

import Database from 'better-sqlite3'
import { mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { homedir } from 'node:os'
import type {
  Store,
  Document,
  Chunk,
  EmbeddingVector,
  SearchResult,
} from '../core/types.js'

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Resolve a path that may start with `~` to an absolute path,
 * creating parent directories as needed.
 */
function resolvePath(p: string): string {
  if (p.startsWith('~')) p = p.replace('~', homedir())
  mkdirSync(dirname(p), { recursive: true })
  return p
}

/**
 * Cosine similarity between two vectors of equal length.
 * Returns 0 if either vector has zero magnitude (avoids division by zero).
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  return denom === 0 ? 0 : dot / denom
}

// ── Row types (internal, maps directly to SQLite rows) ───────────────────────

interface DocumentRow {
  id: string
  source: string
  content: string
  metadata: string // JSON string
  created_at: number // unix timestamp (seconds)
}

interface ChunkRow {
  id: string
  document_id: string
  content: string
  index_num: number
  token_count: number
  metadata: string // JSON string
}

interface EmbeddingRow {
  chunk_id: string
  vector: string // JSON array of floats
  model: string
}

interface FtsMatchRow {
  id: string
  document_id: string
  rank: number
}

// ── Schema ───────────────────────────────────────────────────────────────────

const SCHEMA_SQL = `
-- Documents table
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata TEXT DEFAULT '{}',
  created_at INTEGER NOT NULL
);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  index_num INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  metadata TEXT DEFAULT '{}'
);

-- FTS5 virtual table for BM25 full-text search.
-- Uses porter stemmer + unicode61 tokenizer for multilingual support.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  id UNINDEXED,
  document_id UNINDEXED,
  content,
  tokenize='porter unicode61'
);

-- Embeddings table (stores vectors as JSON arrays of floats).
CREATE TABLE IF NOT EXISTS embeddings (
  chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  vector TEXT NOT NULL,
  model TEXT NOT NULL
);
`

// ── Store implementation ─────────────────────────────────────────────────────

/**
 * Factory function that creates a SQLite-backed Store.
 *
 * @param dbPath - Path to the SQLite database file (supports `~` expansion).
 *                 Parent directories are created automatically.
 * @returns A fully initialized Store instance.
 */
export function createSqliteStore(dbPath: string): Store {
  const resolvedPath = resolvePath(dbPath)
  const db = new Database(resolvedPath)

  // ── Bootstrap ────────────────────────────────────────────────────────────

  // Enable WAL mode for better concurrent read performance.
  db.pragma('journal_mode = WAL')
  // Enable foreign key enforcement (SQLite has it off by default).
  db.pragma('foreign_keys = ON')

  // Create all tables if they don't exist.
  db.exec(SCHEMA_SQL)

  // ── Prepared statements (reused across calls) ────────────────────────────

  const stmtUpsertDocument = db.prepare(`
    INSERT OR REPLACE INTO documents (id, source, content, metadata, created_at)
    VALUES (?, ?, ?, ?, ?)
  `)

  const stmtUpsertChunk = db.prepare(`
    INSERT OR REPLACE INTO chunks (id, document_id, content, index_num, token_count, metadata)
    VALUES (?, ?, ?, ?, ?, ?)
  `)

  const stmtUpsertFts = db.prepare(`
    INSERT OR REPLACE INTO chunks_fts (id, document_id, content)
    VALUES (?, ?, ?)
  `)

  const stmtUpsertEmbedding = db.prepare(`
    INSERT OR REPLACE INTO embeddings (chunk_id, vector, model)
    VALUES (?, ?, ?)
  `)

  const stmtSelectAllEmbeddings = db.prepare(`
    SELECT chunk_id, vector, model FROM embeddings
  `)

  const stmtSelectChunksByDocId = db.prepare(`
    SELECT id, document_id, content, index_num, token_count, metadata
    FROM chunks
    WHERE document_id = ?
    ORDER BY index_num ASC
  `)

  const stmtSelectDocById = db.prepare(`
    SELECT id, source, content, metadata, created_at
    FROM documents
    WHERE id = ?
  `)

  const stmtSelectChunkById = db.prepare(`
    SELECT id, document_id, content, index_num, token_count, metadata
    FROM chunks
    WHERE id = ?
  `)

  // ── FTS5 search statement ────────────────────────────────────────────────
  // FTS5 MATCH syntax: plain terms are ANDed by default.
  // rank is BM25-scored (lower = better match).

  const stmtFtsSearch = db.prepare(`
    SELECT fts.id, fts.document_id, fts.rank
    FROM chunks_fts AS fts
    WHERE chunks_fts MATCH ?
    ORDER BY fts.rank
    LIMIT ?
  `)

  // Join FTS results with chunks to get full content + metadata.
  const stmtChunkById = db.prepare(`
    SELECT id, document_id, content, index_num, token_count, metadata
    FROM chunks
    WHERE id = ?
  `)

  // ── Row → Domain mappers ─────────────────────────────────────────────────

  function rowToDocument(row: DocumentRow): Document {
    return {
      id: row.id,
      source: row.source,
      content: row.content,
      metadata: JSON.parse(row.metadata) as Record<string, unknown>,
      createdAt: new Date(row.created_at * 1000),
    }
  }

  function rowToChunk(row: ChunkRow): Chunk {
    return {
      id: row.id,
      documentId: row.document_id,
      content: row.content,
      index: row.index_num,
      tokenCount: row.token_count,
      metadata: JSON.parse(row.metadata) as Record<string, unknown>,
    }
  }

  // ── Store interface implementation ───────────────────────────────────────

  return {
    /**
     * Insert or replace a document.
     * Converts metadata to JSON string and createdAt to unix timestamp.
     */
    async upsertDocument(doc: Document): Promise<void> {
      stmtUpsertDocument.run(
        doc.id,
        doc.source,
        doc.content,
        JSON.stringify(doc.metadata),
        Math.floor(doc.createdAt.getTime() / 1000),
      )
    },

    /**
     * Batch insert chunks + their FTS5 entries in a single transaction.
     * INSERT OR REPLACE handles idempotent upserts.
     */
    async upsertChunks(chunks: Chunk[]): Promise<void> {
      if (chunks.length === 0) return

      const insertAll = db.transaction(() => {
        for (const chunk of chunks) {
          stmtUpsertChunk.run(
            chunk.id,
            chunk.documentId,
            chunk.content,
            chunk.index,
            chunk.tokenCount,
            JSON.stringify(chunk.metadata),
          )
          // Mirror into FTS5 for BM25 search.
          stmtUpsertFts.run(chunk.id, chunk.documentId, chunk.content)
        }
      })

      insertAll()
    },

    /**
     * Batch insert embeddings in a single transaction.
     * Each embedding is stored as a JSON-serialized float array.
     */
    async upsertEmbeddings(embeddings: EmbeddingVector[]): Promise<void> {
      if (embeddings.length === 0) return

      const insertAll = db.transaction(() => {
        for (const emb of embeddings) {
          stmtUpsertEmbedding.run(
            emb.id,
            JSON.stringify(emb.vector),
            emb.model,
          )
        }
      })

      insertAll()
    },

    /**
     * Dense vector search: load all embeddings, compute cosine similarity,
     * return top K results sorted by descending score.
     *
     * For <100K chunks this is fast (<100ms). Beyond that, migrate to a
     * dedicated vector store (pgvector, qdrant, etc.).
     */
    async searchDense(
      vector: number[],
      topK: number,
    ): Promise<SearchResult[]> {
      const rows = stmtSelectAllEmbeddings.all() as EmbeddingRow[]

      // Score every embedding against the query vector.
      const scored: Array<{ row: EmbeddingRow; score: number }> = []
      for (const row of rows) {
        const embVector = JSON.parse(row.vector) as number[]
        const score = cosineSimilarity(vector, embVector)
        scored.push({ row, score })
      }

      // Sort by score descending, take top K.
      scored.sort((a, b) => b.score - a.score)
      const topResults = scored.slice(0, topK)

      // Resolve chunk metadata for each result.
      const results: SearchResult[] = []
      for (const { row, score } of topResults) {
        const chunkRow = stmtChunkById.get(row.chunk_id) as ChunkRow | undefined
        if (!chunkRow) continue // chunk was deleted; skip

        results.push({
          chunkId: row.chunk_id,
          documentId: chunkRow.document_id,
          content: chunkRow.content,
          score,
          method: 'dense',
          metadata: JSON.parse(chunkRow.metadata) as Record<string, unknown>,
        })
      }

      return results
    },

    /**
     * BM25 full-text search using FTS5.
     *
     * FTS5 MATCH syntax handles tokenization automatically. The porter
     * stemmer + unicode61 tokenizer handles stemming and unicode folding.
     * FTS5 rank is BM25-scored (lower = better); we negate so that
     * higher scores = better relevance in SearchResult.
     */
    async searchBM25(
      query: string,
      topK: number,
    ): Promise<SearchResult[]> {
      // Sanitize query: FTS5 interprets certain characters specially
      // (AND, OR, NOT, -, +, *, ", ^, ~, :, \, (, )). Strip them and
      // join tokens with implicit AND for a simple keyword search.
      const tokens = query
        .trim()
        .split(/\s+/)
        .map((t) => t.replace(/[^a-zA-Z0-9]/g, ''))
        .filter((t) => t.length > 0)

      if (tokens.length === 0) return []

      const ftsQuery = tokens.join(' ')

      const ftsRows = stmtFtsSearch.all(ftsQuery, topK) as FtsMatchRow[]

      const results: SearchResult[] = []
      for (const ftsRow of ftsRows) {
        const chunkRow = stmtChunkById.get(ftsRow.id) as ChunkRow | undefined
        if (!chunkRow) continue

        // Negate FTS5 rank: lower rank = better match, so negate to get
        // a "higher is better" score consistent with cosine similarity.
        const score = -ftsRow.rank

        results.push({
          chunkId: ftsRow.id,
          documentId: ftsRow.document_id,
          content: chunkRow.content,
          score,
          method: 'bm25',
          metadata: JSON.parse(chunkRow.metadata) as Record<string, unknown>,
        })
      }

      return results
    },

    /**
     * Retrieve all chunks for a document, ordered by their index position.
     */
    async getChunksByDocumentId(documentId: string): Promise<Chunk[]> {
      const rows = stmtSelectChunksByDocId.all(documentId) as ChunkRow[]
      return rows.map(rowToChunk)
    },

    /**
     * Retrieve a single document by ID, or null if not found.
     */
    async getDocumentById(id: string): Promise<Document | null> {
      const row = stmtSelectDocById.get(id) as DocumentRow | undefined
      return row ? rowToDocument(row) : null
    },

    /**
     * Retrieve a single chunk by ID, or null if not found.
     */
    async getChunkById(id: string): Promise<Chunk | null> {
      const row = stmtSelectChunkById.get(id) as ChunkRow | undefined
      return row ? rowToChunk(row) : null
    },

    /**
     * Close the database connection. Idempotent — safe to call multiple times.
     */
    async close(): Promise<void> {
      db.close()
    },
  }
}
