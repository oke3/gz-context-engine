// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * MCP (Model Context Protocol) server for gz-context-engine.
 *
 * Exposes the context engine as a set of MCP tools and resources
 * that any MCP-compatible client (Claude Desktop, Cursor, etc.) can use.
 *
 * Tools:
 *   - `search`  — Hybrid search across the knowledge base
 *   - `fetch`   — Fetch a specific document or chunk by ID
 *   - `ask`     — End-to-end RAG: search + retrieve + generate
 *   - `ingest`  — Ingest a document into the knowledge base
 *
 * Resources:
 *   - `gz://status` — Engine status (config, chunk count, doc count)
 *   - `gz://config` — Current configuration
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { z } from 'zod/v3'
import type { ContextEngine } from '../core/engine.js'
import type { SearchResult } from '../core/types.js'

// ── Zod Schemas ──────────────────────────────────────────────────────────────

const SearchInput = {
  query: z.string().describe('The search query'),
  topK: z.number().optional().describe('Number of results to return (default: 5)'),
  method: z.enum(['hybrid', 'dense', 'bm25']).optional().describe('Search method (default: hybrid)'),
}

const FetchInput = {
  id: z.string().describe('The document or chunk ID to fetch'),
  type: z.enum(['document', 'chunk']).describe('Whether to fetch a document or chunk'),
}

const AskInput = {
  question: z.string().describe('The question to answer using RAG'),
  model: z.string().optional().describe('Override the generation model'),
}

const IngestInput = {
  source: z.string().describe('Source identifier (file path, URL, etc.)'),
  content: z.string().describe('The document content to ingest'),
}

// ── Server Factory ───────────────────────────────────────────────────────────

/**
 * Create and configure an MCP server backed by the given ContextEngine.
 *
 * @param engine - A fully initialized ContextEngine instance.
 * @returns An McpServer instance ready to be connected to a transport.
 */
export function createMcpServer(engine: ContextEngine): McpServer {
  const server = new McpServer(
    {
      name: 'gz-context-engine',
      version: '0.1.0',
    },
    {
      capabilities: {
        tools: {},
        resources: {},
      },
    },
  )

  // ── Tools ────────────────────────────────────────────────────────────────

  /**
   * Tool: search — Hybrid search across the knowledge base.
   */
  server.registerTool(
    'search',
    {
      description:
        'Search the knowledge base using hybrid (dense + BM25), dense-only, or BM25-only retrieval. ' +
        'Returns ranked chunks with relevance scores.',
      inputSchema: SearchInput,
    },
    async ({ query, topK, method }) => {
      try {
        const results = await engine.search(query, {
          topK: topK ?? 5,
          method: method ?? 'hybrid',
        })

        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                results.map(formatSearchResult),
                null,
                2,
              ),
            },
          ],
        }
      } catch (error) {
        return formatToolError(error)
      }
    },
  )

  /**
   * Tool: fetch — Fetch a specific document or chunk by ID.
   */
  server.registerTool(
    'fetch',
    {
      description:
        'Fetch a specific document or chunk from the knowledge base by its ID. ' +
        'Returns the full content along with metadata.',
      inputSchema: FetchInput,
    },
    async ({ id, type }) => {
      try {
        if (type === 'document') {
          // Access the store directly via engine internals (lazy init)
          const store = await (engine as any).store.get()
          const doc = await store.getDocumentById(id)
          if (!doc) {
            return {
              content: [{ type: 'text' as const, text: `Document not found: ${id}` }],
              isError: true,
            }
          }
          return {
            content: [
              {
                type: 'text' as const,
                text: JSON.stringify(
                  {
                    id: doc.id,
                    source: doc.source,
                    content: doc.content,
                    metadata: doc.metadata,
                    createdAt: doc.createdAt.toISOString(),
                  },
                  null,
                  2,
                ),
              },
            ],
          }
        }

        // Fetch chunk
        const store = await (engine as any).store.get()
        const chunk = await store.getChunkById(id)
        if (!chunk) {
          return {
            content: [{ type: 'text' as const, text: `Chunk not found: ${id}` }],
            isError: true,
          }
        }
        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                {
                  id: chunk.id,
                  documentId: chunk.documentId,
                  content: chunk.content,
                  index: chunk.index,
                  tokenCount: chunk.tokenCount,
                  metadata: chunk.metadata,
                },
                null,
                2,
              ),
            },
          ],
        }
      } catch (error) {
        return formatToolError(error)
      }
    },
  )

  /**
   * Tool: ask — End-to-end RAG: search + retrieve + generate.
   */
  server.registerTool(
    'ask',
    {
      description:
        'Ask a question and get an answer generated from the knowledge base. ' +
        'Performs hybrid search, assembles context, and calls the LLM to generate a response with citations.',
      inputSchema: AskInput,
    },
    async ({ question, model }) => {
      try {
        const response = await engine.query({
          query: question,
          context: [],
          model,
        })

        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                {
                  answer: response.text,
                  model: response.model,
                  tokens: response.tokens,
                  cost: response.cost,
                  latencyMs: response.latencyMs,
                  citations: response.citations,
                },
                null,
                2,
              ),
            },
          ],
        }
      } catch (error) {
        return formatToolError(error)
      }
    },
  )

  /**
   * Tool: ingest — Ingest a document into the knowledge base.
   */
  server.registerTool(
    'ingest',
    {
      description:
        'Ingest a document into the knowledge base. The document is chunked, embedded, and stored. ' +
        'Returns the document ID and the number of chunks created.',
      inputSchema: IngestInput,
    },
    async ({ source, content }) => {
      try {
        const doc = await engine.ingest(source, content)
        const store = await (engine as any).store.get()
        const chunks = await store.getChunksByDocumentId(doc.id)

        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(
                {
                  documentId: doc.id,
                  source: doc.source,
                  chunkCount: chunks.length,
                  contentLength: content.length,
                },
                null,
                2,
              ),
            },
          ],
        }
      } catch (error) {
        return formatToolError(error)
      }
    },
  )

  // ── Resources ────────────────────────────────────────────────────────────

  /**
   * Resource: gz://status — Engine status.
   */
  server.registerResource(
    'status',
    'gz://status',
    {
      description: 'Current engine status: configuration, chunk count, document count.',
      mimeType: 'application/json',
    },
    async () => {
      try {
        const store = await (engine as any).store.get()

        // Count documents and chunks via raw SQL (store doesn't expose counts)
        const db = (store as any).db
        const docCount = db?.prepare('SELECT COUNT(*) as cnt FROM documents').get()?.cnt ?? 0
        const chunkCount = db?.prepare('SELECT COUNT(*) as cnt FROM chunks').get()?.cnt ?? 0
        const embeddingCount = db?.prepare('SELECT COUNT(*) as cnt FROM embeddings').get()?.cnt ?? 0

        return {
          contents: [
            {
              uri: 'gz://status',
              mimeType: 'application/json',
              text: JSON.stringify(
                {
                  status: 'ok',
                  documents: docCount,
                  chunks: chunkCount,
                  embeddings: embeddingCount,
                  timestamp: new Date().toISOString(),
                },
                null,
                2,
              ),
            },
          ],
        }
      } catch (error) {
        return {
          contents: [
            {
              uri: 'gz://status',
              mimeType: 'application/json',
              text: JSON.stringify({ status: 'error', error: String(error) }),
            },
          ],
        }
      }
    },
  )

  /**
   * Resource: gz://config — Current configuration.
   */
  server.registerResource(
    'config',
    'gz://config',
    {
      description: 'Current engine configuration (redacted API keys).',
      mimeType: 'application/json',
    },
    async () => {
      try {
        // Access config via engine internals
        const config = (engine as any).config
        // Redact sensitive fields
        const safeConfig = JSON.parse(JSON.stringify(config))
        if (safeConfig.embedding?.apiKey) safeConfig.embedding.apiKey = '***'
        if (safeConfig.generation?.apiKey) safeConfig.generation.apiKey = '***'

        return {
          contents: [
            {
              uri: 'gz://config',
              mimeType: 'application/json',
              text: JSON.stringify(safeConfig, null, 2),
            },
          ],
        }
      } catch (error) {
        return {
          contents: [
            {
              uri: 'gz://config',
              mimeType: 'application/json',
              text: JSON.stringify({ error: String(error) }),
            },
          ],
        }
      }
    },
  )

  return server
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Format a SearchResult for display in tool output.
 */
function formatSearchResult(result: SearchResult): Record<string, unknown> {
  return {
    chunkId: result.chunkId,
    documentId: result.documentId,
    score: Math.round(result.score * 1000) / 1000,
    method: result.method,
    content: result.content,
    metadata: result.metadata,
  }
}

/**
 * Format an error as an MCP tool error response.
 */
function formatToolError(error: unknown): {
  content: Array<{ type: 'text'; text: string }>
  isError: true
} {
  const message = error instanceof Error ? error.message : String(error)
  return {
    content: [{ type: 'text' as const, text: `Error: ${message}` }],
    isError: true as const,
  }
}

// ── Main Entry Point ─────────────────────────────────────────────────────────

/**
 * Start the MCP server over stdio transport. This is the entry point
 * for `gz-context mcp` CLI command.
 *
 * @param engine - A fully initialized ContextEngine instance.
 */
export async function startMcpServer(engine: ContextEngine): Promise<void> {
  const server = createMcpServer(engine)
  const transport = new StdioServerTransport()

  // Handle clean shutdown
  process.on('SIGINT', async () => {
    await server.close()
    await engine.close()
    process.exit(0)
  })

  process.on('SIGTERM', async () => {
    await server.close()
    await engine.close()
    process.exit(0)
  })

  await server.connect(transport)
}
