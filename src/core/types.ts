// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

// ── Document & Chunk Types ──

export interface Document {
  id: string
  source: string
  content: string
  metadata: Record<string, unknown>
  createdAt: Date
}

export interface Chunk {
  id: string
  documentId: string
  content: string
  index: number
  tokenCount: number
  metadata: Record<string, unknown>
}

// ── Embedding Types ──

export interface EmbeddingVector {
  id: string
  vector: number[]
  model: string
}

// ── Retrieval Types ──

export interface SearchResult {
  chunkId: string
  documentId: string
  content: string
  score: number
  method: 'dense' | 'bm25' | 'hybrid'
  metadata: Record<string, unknown>
}

export interface RerankedResult extends SearchResult {
  originalScore: number
  rerankScore: number
}

export interface ContextItem {
  content: string
  source: string
  score: number
  citation: string
  tokenCount: number
}

// ── Generation Types ──

export interface GenerationRequest {
  query: string
  context: ContextItem[]
  systemPrompt?: string
  temperature?: number
  maxTokens?: number
  model?: string
  stream?: boolean
}

export interface GenerationResponse {
  text: string
  model: string
  tokens: { input: number; output: number }
  cost: number
  latencyMs: number
  citations: string[]
}

// ── Provider Types ──

export interface EmbeddingProvider {
  name: string
  embed(texts: string[]): Promise<number[][]>
  dimensions: number
}

export interface GenerationProvider {
  name: string
  generate(request: GenerationRequest): Promise<GenerationResponse>
  generateStream?(request: GenerationRequest): AsyncGenerator<string>
}

export interface RerankerProvider {
  name: string
  rerank(query: string, documents: string[], topK: number): Promise<number[]>
}

// ── Config Types ──

export interface ContextEngineConfig {
  dataDir: string
  embedding: {
    provider: string
    model: string
    dimensions: number
  }
  store: {
    type: string
    dbPath: string
  }
  retrieval: {
    denseTopK: number
    bm25TopK: number
    rerankTopK: number
    rrfK: number
  }
  generation: {
    provider: string
    model: string
    temperature: number
    maxTokens: number
  }
  chunking: {
    strategy: string
    maxChunkTokens: number
    overlapTokens: number
  }
}

// ── Evaluation Types ──

export interface EvalCase {
  query: string
  expectedAnswer: string
  expectedChunks?: string[]
  metadata?: Record<string, unknown>
}

export interface EvalResult {
  case: EvalCase
  actualAnswer: string
  retrievedChunks: string[]
  metrics: {
    precision: number
    recall: number
    faithfulness: number
    relevance: number
  }
  latencyMs: number
}

// ── Internal Types (not exported to consumers) ──

export interface Store {
  upsertDocument(doc: Document): Promise<void>
  upsertChunks(chunks: Chunk[]): Promise<void>
  upsertEmbeddings(embeddings: EmbeddingVector[]): Promise<void>
  searchDense(vector: number[], topK: number): Promise<SearchResult[]>
  searchBM25(query: string, topK: number): Promise<SearchResult[]>
  getChunksByDocumentId(documentId: string): Promise<Chunk[]>
  getDocumentById(id: string): Promise<Document | null>
  getChunkById(id: string): Promise<Chunk | null>
  close(): Promise<void>
}

export interface Chunker {
  chunk(content: string, documentId: string): Promise<Chunk[]>
}

export interface PipelineMetrics {
  documentsIngested: number
  chunksCreated: number
  embeddingsGenerated: number
  totalLatencyMs: number
}

// ── Defaults ──

export const DEFAULT_CONFIG: ContextEngineConfig = {
  dataDir: '~/.gz-context',
  embedding: {
    provider: 'openai',
    model: 'text-embedding-3-small',
    dimensions: 1536,
  },
  store: {
    type: 'sqlite',
    dbPath: '~/.gz-context/context.db',
  },
  retrieval: {
    denseTopK: 20,
    bm25TopK: 20,
    rerankTopK: 5,
    rrfK: 60,
  },
  generation: {
    provider: 'openai',
    model: 'gpt-4o-mini',
    temperature: 0.1,
    maxTokens: 2048,
  },
  chunking: {
    strategy: 'recursive',
    maxChunkTokens: 512,
    overlapTokens: 64,
  },
}
