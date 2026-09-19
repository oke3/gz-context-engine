// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

import type {
  ContextEngineConfig,
  Document,
  SearchResult,
  ContextItem,
  GenerationRequest,
  GenerationResponse,
  Store,
  Chunker,
} from './types.js'
import { DEFAULT_CONFIG } from './types.js'
import { buildConfig } from './config.js'

/**
 * Lazy-init wrapper. Creates a component on first access, reuses after.
 */
class Lazy<T> {
  private instance: T | null = null
  private factory: () => Promise<T>

  constructor(factory: () => Promise<T>) {
    this.factory = factory
  }

  async get(): Promise<T> {
    if (!this.instance) {
      this.instance = await this.factory()
    }
    return this.instance
  }

  isReady(): boolean {
    return this.instance !== null
  }

  async reset(): Promise<void> {
    this.instance = null
  }
}

/**
 * Main context engine facade. All public API goes through this class.
 * Internal components (store, chunker, embedder, reranker, generator)
 * are lazily initialized on first use.
 */
export class ContextEngine {
  private config: ContextEngineConfig
  private store: Lazy<Store>
  private chunker: Lazy<Chunker>
  private closed = false

  constructor(config?: Partial<ContextEngineConfig>) {
    // Eagerly build config (async file I/O happens in `init()`).
    // Store the overrides; actual config resolves on first use.
    this.config = DEFAULT_CONFIG
    if (config) {
      this.config = { ...DEFAULT_CONFIG, ...config }
    }

    this.store = new Lazy<Store>(async () => {
      const resolved = await buildConfig(this.config as Partial<ContextEngineConfig>)
      this.config = resolved
      // Store creation delegated to store module (imported lazily)
      const { createSqliteStore } = await import('../store/sqlite.js')
      return createSqliteStore(resolved.store.dbPath)
    })

    this.chunker = new Lazy<Chunker>(async () => {
      const resolved = this.store.isReady()
        ? this.config
        : await buildConfig(this.config as Partial<ContextEngineConfig>)
      const { createChunker } = await import('../ingest/chunker.js')
      return createChunker(
        resolved.chunking.strategy,
        resolved.chunking.maxChunkTokens,
        resolved.chunking.overlapTokens,
      )
    })
  }

  // ── Ingestion ──

  async ingest(source: string, content?: string): Promise<Document> {
    this.assertNotClosed()
    const store = await this.store.get()
    const chunker = await this.chunker.get()

    const body = content ?? (await readFile(source))
    const id = crypto.randomUUID()
    const doc: Document = {
      id,
      source,
      content: body,
      metadata: {},
      createdAt: new Date(),
    }

    await store.upsertDocument(doc)

    const chunks = await chunker.chunk(body, id)
    await store.upsertChunks(chunks)

    // Embeddings generated lazily if an embedding provider is configured
    await this.embedChunks(chunks)

    return doc
  }

  async ingestBatch(
    sources: { source: string; content?: string }[],
  ): Promise<Document[]> {
    this.assertNotClosed()
    const results: Document[] = []
    for (const item of sources) {
      results.push(await this.ingest(item.source, item.content))
    }
    return results
  }

  // ── Retrieval ──

  async search(
    query: string,
    options?: { topK?: number; method?: 'hybrid' | 'dense' | 'bm25' },
  ): Promise<SearchResult[]> {
    this.assertNotClosed()
    const store = await this.store.get()
    const method = options?.method ?? 'hybrid'
    const topK = options?.topK ?? this.config.retrieval.rerankTopK

    let results: SearchResult[] = []

    if (method === 'dense' || method === 'hybrid') {
      const embedding = await this.embedQuery(query)
      const denseResults = await store.searchDense(
        embedding,
        this.config.retrieval.denseTopK,
      )
      if (method === 'dense') {
        results = denseResults.slice(0, topK)
      } else {
        // Hybrid: combine with BM25 via RRF
        const bm25Results = await store.searchBM25(
          query,
          this.config.retrieval.bm25TopK,
        )
        results = this.rrfMerge(denseResults, bm25Results)
      }
    } else {
      results = await store.searchBM25(query, this.config.retrieval.bm25TopK)
    }

    return results.slice(0, topK)
  }

  async retrieve(
    query: string,
    options?: { topK?: number },
  ): Promise<ContextItem[]> {
    this.assertNotClosed()
    const topK = options?.topK ?? this.config.retrieval.rerankTopK
    const searchResults = await this.search(query, { topK })
    const store = await this.store.get()

    const items: ContextItem[] = []
    for (const result of searchResults) {
      const doc = await store.getDocumentById(result.documentId)
      items.push({
        content: result.content,
        source: doc?.source ?? result.documentId,
        score: result.score,
        citation: `[${items.length + 1}] ${doc?.source ?? 'unknown'}#chunk-${result.chunkId.slice(0, 8)}`,
        tokenCount: result.content.split(/\s+/).length, // rough estimate
      })
    }

    return items
  }

  // ── Generation ──

  async query(request: GenerationRequest): Promise<GenerationResponse> {
    this.assertNotClosed()
    // Resolve context if not provided
    const context =
      request.context.length > 0
        ? request.context
        : await this.retrieve(request.query)

    const resolvedRequest: GenerationRequest = {
      ...request,
      context,
      model: request.model ?? this.config.generation.model,
      temperature: request.temperature ?? this.config.generation.temperature,
      maxTokens: request.maxTokens ?? this.config.generation.maxTokens,
    }

    const { createGenerationProvider } = await import(
      '../generate/provider.js'
    )
    const provider = createGenerationProvider(this.config.generation.provider)
    return provider.generate(resolvedRequest)
  }

  async *queryStream(request: GenerationRequest): AsyncGenerator<string> {
    this.assertNotClosed()
    const context =
      request.context.length > 0
        ? request.context
        : await this.retrieve(request.query)

    const resolvedRequest: GenerationRequest = {
      ...request,
      context,
      model: request.model ?? this.config.generation.model,
      temperature: request.temperature ?? this.config.generation.temperature,
      maxTokens: request.maxTokens ?? this.config.generation.maxTokens,
      stream: true,
    }

    const { createGenerationProvider } = await import(
      '../generate/provider.js'
    )
    const provider = createGenerationProvider(this.config.generation.provider)

    if (!provider.generateStream) {
      throw new Error(
        `Provider "${provider.name}" does not support streaming`,
      )
    }

    yield* provider.generateStream(resolvedRequest)
  }

  // ── Lifecycle ──

  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    if (this.store.isReady()) {
      await (await this.store.get()).close()
    }
  }

  // ── Private Helpers ──

  private assertNotClosed(): void {
    if (this.closed) {
      throw new Error('ContextEngine has been closed')
    }
  }

  private async embedChunks(
    chunks: Array<{ id: string; content: string }>,
  ): Promise<void> {
    try {
      const { createEmbeddingProvider } = await import(
        '../embed/provider.js'
      )
      const store = await this.store.get()
      const provider = createEmbeddingProvider(this.config.embedding.provider)

      const texts = chunks.map((c) => c.content)
      const vectors = await provider.embed(texts)

      await store.upsertEmbeddings(
        vectors.map((vector: number[], i: number) => ({
          id: chunks[i].id,
          vector,
          model: this.config.embedding.model,
        })),
      )
    } catch {
      // Embedding is optional — ingestion succeeds without it
    }
  }

  private async embedQuery(query: string): Promise<number[]> {
    const { createEmbeddingProvider } = await import('../embed/provider.js')
    const provider = createEmbeddingProvider(this.config.embedding.provider)
    const vectors = await provider.embed([query])
    return vectors[0]
  }

  /**
   * Reciprocal Rank Fusion: merge two result lists by rank position.
   */
  private rrfMerge(
    denseResults: SearchResult[],
    bm25Results: SearchResult[],
  ): SearchResult[] {
    const k = this.config.retrieval.rrfK
    const scoreMap = new Map<string, { result: SearchResult; score: number }>()

    denseResults.forEach((r, i) => {
      const existing = scoreMap.get(r.chunkId)
      const rrfScore = 1 / (k + i + 1)
      if (existing) {
        existing.score += rrfScore
      } else {
        scoreMap.set(r.chunkId, { result: r, score: rrfScore })
      }
    })

    bm25Results.forEach((r, i) => {
      const existing = scoreMap.get(r.chunkId)
      const rrfScore = 1 / (k + i + 1)
      if (existing) {
        existing.score += rrfScore
      } else {
        scoreMap.set(r.chunkId, { result: r, score: rrfScore })
      }
    })

    return Array.from(scoreMap.values())
      .sort((a, b) => b.score - a.score)
      .map((entry) => ({ ...entry.result, score: entry.score, method: 'hybrid' as const }))
  }
}

/**
 * Read a file from disk. Used by ingest when no content is provided.
 */
async function readFile(path: string): Promise<string> {
  const { readFile: nodeReadFile } = await import('node:fs/promises')
  return nodeReadFile(path, 'utf-8')
}
