// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

// ── Public API ──
//
// Only export the public surface: the engine facade, all types,
// config builder, and the evaluation harness. Internal modules
// (store, chunker, embed, generate) are implementation details
// and must NOT be re-exported.

export { ContextEngine } from './core/engine.js'
export type {
  Document,
  Chunk,
  EmbeddingVector,
  SearchResult,
  RerankedResult,
  ContextItem,
  GenerationRequest,
  GenerationResponse,
  EmbeddingProvider,
  GenerationProvider,
  RerankerProvider,
  ContextEngineConfig,
  EvalCase,
  EvalResult,
} from './core/types.js'
export { DEFAULT_CONFIG } from './core/types.js'
export { buildConfig, buildConfigSync } from './core/config.js'

// ── Evaluation ──
export { EvalSuite } from './eval/harness.js'
export type { EvalReport, EvalRunOptions } from './eval/harness.js'
export {
  precisionAtK,
  recallAtK,
  faithfulness,
  relevance,
} from './eval/harness.js'
