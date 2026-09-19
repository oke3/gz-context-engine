# gz-context-engine

> Production-grade context engine for AI agents. Hybrid RAG, cross-encoder reranking, token-aware assembly, MCP server, built-in eval harness.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Ground Zero LLC](https://img.shields.io/badge/Built%20by-Ground%20Zero%20LLC-purple)](https://github.com/oke3)
[![npm](https://img.shields.io/npm/v/@ground-zero-llc/gz-context-engine)](https://www.npmjs.com/package/@ground-zero-llc/gz-context-engine)
[![CI](https://github.com/oke3/gz-context-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/oke3/gz-context-engine/actions)

---

## Why

Most "RAG" tutorials hand you an API call and a vector store and call it a day.

Production RAG is harder: naive vector search misses keyword-critical queries, top-K truncation ignores context window budgets, and nobody knows if the pipeline actually works because there's no eval harness.

`gz-context-engine` is the context layer that sits between your documents and your LLM. It handles **retrieval** (hybrid BM25 + dense + RRF fusion), **reranking**, **token-aware context assembly**, **generation** with cost tracking, and ships with a **built-in evaluation framework** so you can measure precision, recall, faithfulness, and latency — not just hope it works.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        gz-context-engine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Ingest   │──▶│ Chunk    │──▶│ Embed    │──▶│ SQLite     │  │
│  │ Pipeline │   │ (recur/  │   │ (OpenAI/ │   │ Store      │  │
│  │          │   │  md/fixed│   │  custom) │   │ (bm25+vec) │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────┬──────┘  │
│                                                      │         │
│                    ┌─────────────────────────────────┘         │
│                    ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Retrieval Pipeline                     │  │
│  │                                                          │  │
│  │  Dense (ANN) ──┐                                         │  │
│  │                 ├──▶ RRF Fusion ──▶ Rerank ──▶ Top-K     │  │
│  │  BM25 (FTS) ──┘                                         │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│  ┌────────────────────────▼─────────────────────────────────┐  │
│  │              Context Assembly + Generation               │  │
│  │                                                          │  │
│  │  Token budget ──▶ Prompt assembly ──▶ LLM ──▶ Response   │  │
│  │                   (citations)          (cost tracking)   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │ CLI             │  │ MCP Server  │  │ Eval Harness       │  │
│  │ (ingest/search/ │  │ (stdio)     │  │ (P/R/F/R latency)  │  │
│  │  ask/status)    │  │             │  │                    │  │
│  └─────────────────┘  └─────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
npm install @ground-zero-llc/gz-context-engine

# Ingest your docs
gz-context ingest ./docs/

# Search
gz-context search "how does tokenization work"

# Ask (end-to-end RAG)
gz-context ask "What are the key architectural decisions?"

# Check status
gz-context status
```

## Features

### Hybrid Search (BM25 + Dense + RRF)

Combines sparse (BM25/FTS5) and dense (embedding cosine) retrieval, then merges via **Reciprocal Rank Fusion** with configurable `k`. Handles both semantic and keyword-critical queries.

```ts
// Copyright (c) 2026 Ground Zero LLC.
const results = await engine.search("vector database performance", {
  method: 'hybrid',  // or 'dense' | 'bm25'
  topK: 10,
})
```

### Cross-Encoder Reranking

Pluggable `RerankerProvider` interface — swap in any cross-encoder (Cohere, custom ONNX, etc.) to re-score top-N candidates before context assembly.

```ts
// Copyright (c) 2026 Ground Zero LLC.
interface RerankerProvider {
  name: string
  rerank(query: string, documents: string[], topK: number): Promise<number[]>
}
```

### Smart Chunking

Three strategies out of the box:
- **Recursive** — splits on paragraph → sentence → token boundaries (default)
- **Markdown** — respects heading hierarchy and code blocks
- **Fixed** — simple token-count windows with overlap

```json
{
  "chunking": {
    "strategy": "recursive",
    "maxChunkTokens": 512,
    "overlapTokens": 64
  }
}
```

### Token-Aware Context Assembly

`retrieve()` returns `ContextItem[]` with pre-computed `tokenCount` values. Build prompts that stay within model limits without hard-coding counts.

```ts
// Copyright (c) 2026 Ground Zero LLC.
const context = await engine.retrieve("How does RRF work?", { topK: 5 })

// context[0] = {
//   content: "Reciprocal Rank Fusion combines...",
//   source: "docs/architecture.md",
//   score: 0.892,
//   citation: "[1] docs/architecture.md#chunk-a1b2c3d4",
//   tokenCount: 147,
// }
```

### Multi-Provider Generation

Pluggable `GenerationProvider` interface. Ships with OpenAI-compatible; swap in Anthropic, local models, or any HTTP endpoint.

```ts
// Copyright (c) 2026 Ground Zero LLC.
const response = await engine.query({
  query: "Explain the RRF algorithm",
  context: [],          // auto-retrieves if empty
  model: 'gpt-4o-mini', // override per-query
  temperature: 0.1,
})

console.log(response.text)        // Generated answer
console.log(response.citations)   // ["[1] docs/arch.md#chunk-a1b2c3d4"]
console.log(response.cost)        // 0.000342
console.log(response.latencyMs)   // 1847
```

### MCP Server

Drop-in MCP server for any MCP-compatible client (Claude Desktop, Cursor, etc.).

```bash
# Start via CLI
gz-context mcp
```

```jsonc
// Claude Desktop config
{
  "mcpServers": {
    "gz-context": {
      "command": "gz-context",
      "args": ["mcp"]
    }
  }
}
```

**Tools:** `search`, `fetch`, `ask`, `ingest`
**Resources:** `gz://status`, `gz://config`

### Built-in Eval Harness

Measure what matters. The eval suite computes **Precision@K**, **Recall@K**, **Faithfulness**, **Relevance**, plus p50/p99 latency and cost per run.

```ts
// Copyright (c) 2026 Ground Zero LLC.
import { EvalSuite } from '@ground-zero-llc/gz-context-engine'

const suite = new EvalSuite(engine)
await suite.load([
  {
    query: "What is the chunking strategy?",
    expectedAnswer: "Recursive splitting...",
    expectedChunks: ["chunk-id-1", "chunk-id-2"],
  },
])

const report = await suite.run({ concurrency: 5 })
console.log(report.summary.meanPrecision)  // 0.85
console.log(report.summary.p50LatencyMs)   // 1200
```

### Local-First (SQLite)

Zero infrastructure. All data stored in a single SQLite database with FTS5 for BM25 and vector columns for dense search. Runs anywhere — laptop, CI, edge.

## CLI Reference

```
gz-context <command> [options]

Commands:
  ingest <path>       Ingest a file or directory into the knowledge base
  search <query>      Hybrid search across the knowledge base
  ask <question>      End-to-end RAG: search + retrieve + generate
  status              Show engine status (config, chunk count, doc count)
  eval <suite.json>   Run evaluation suite
  mcp                 Start MCP server (stdio transport)

Options:
  --config <path>     Custom config file
  --top-k <n>         Override top K for retrieval
  --model <name>      Override generation model
  --method <method>   Search method: hybrid, dense, bm25
  -c, --concurrency   Max concurrent eval cases
  --json              Output as JSON
```

### Examples

```bash
# Ingest a directory recursively
gz-context ingest ./knowledge-base/

# Search with specific method
gz-context search "deployment process" --method bm25 --top-k 10

# Ask with a different model
gz-context ask "Summarize the architecture" --model gpt-4o

# Run eval suite with JSON output
gz-context eval ./eval-suites/regression.json --json -c 10

# Check engine status
gz-context status
```

## Configuration

Config resolution order: **defaults ← `~/.gz-context/config.json` ← overrides**.

```jsonc
{
  "dataDir": "~/.gz-context",
  "embedding": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "dimensions": 1536
  },
  "store": {
    "type": "sqlite",
    "dbPath": "~/.gz-context/context.db"
  },
  "retrieval": {
    "denseTopK": 20,       // candidates from vector search
    "bm25TopK": 20,        // candidates from BM25
    "rerankTopK": 5,       // final results after fusion
    "rrfK": 60             // RRF smoothing parameter
  },
  "generation": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "maxTokens": 2048
  },
  "chunking": {
    "strategy": "recursive",
    "maxChunkTokens": 512,
    "overlapTokens": 64
  }
}
```

## API Reference

```ts
// Copyright (c) 2026 Ground Zero LLC.
import { ContextEngine, buildConfig } from '@ground-zero-llc/gz-context-engine'

// Create with defaults
const engine = new ContextEngine()

// Or with config overrides
const engine = new ContextEngine({
  embedding: { provider: 'openai', model: 'text-embedding-3-small', dimensions: 1536 },
  generation: { provider: 'openai', model: 'gpt-4o', temperature: 0.1, maxTokens: 4096 },
})

// ── Ingestion ──

await engine.ingest('/path/to/file.md')                     // from disk
await engine.ingest('doc://manual', contentString)           // from string
await engine.ingestBatch([{ source: 'a.md' }, { source: 'b.md' }])

// ── Retrieval ──

const results = await engine.search(query, { method: 'hybrid', topK: 10 })
const context = await engine.retrieve(query, { topK: 5 })

// ── Generation ──

const response = await engine.query({ query, context: [] })
for await (const chunk of engine.queryStream({ query, context: [] })) {
  process.stdout.write(chunk)
}

// ── Lifecycle ──

await engine.close()
```

## Architecture Deep Dive

### 1. Ingestion Pipeline

Documents are chunked using a configurable strategy, then embedded via a pluggable `EmbeddingProvider`. Chunks, embeddings, and source metadata are written to SQLite in a single transaction.

### 2. Retrieval (Hybrid RRF)

The search pipeline runs dense and BM25 retrieval in parallel, then merges via **Reciprocal Rank Fusion**:

```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

Where `k` (default: 60) controls how much rank position matters. Lower `k` = more weight to top results. The merged list is truncated to `rerankTopK`.

### 3. Context Assembly

`retrieve()` wraps search results into `ContextItem` objects with source attribution, citation strings, and pre-computed token counts. This makes it trivial to stay within model token budgets:

```ts
// Copyright (c) 2026 Ground Zero LLC.
const context = await engine.retrieve(query, { topK: 5 })
const totalTokens = context.reduce((sum, c) => sum + c.tokenCount, 0)
// Fit into your prompt template with confidence
```

### 4. Lazy Initialization

All internal components (store, chunker, embedder, generator) use lazy initialization. Nothing touches disk or makes API calls until the first `ingest()` or `search()` call. This makes construction instant and avoids wasting resources if you only need one subsystem.

### 5. Provider Abstraction

Every external dependency is behind a clean interface:

| Interface | Purpose | Ships with |
|-----------|---------|------------|
| `EmbeddingProvider` | Text → vectors | OpenAI-compatible |
| `GenerationProvider` | Context → answer | OpenAI-compatible |
| `RerankerProvider` | Re-score candidates | — (bring your own) |
| `Store` | Persistence + search | SQLite (FTS5 + vec) |

## Eval Suite

### Defining Test Cases

```json
[
  {
    "query": "What chunking strategy does the engine use?",
    "expectedAnswer": "The engine uses recursive chunking...",
    "expectedChunks": ["chunk-id-from-ingestion"],
    "metadata": { "category": "architecture", "difficulty": "easy" }
  },
  {
    "query": "How does RRF merge dense and sparse results?",
    "expectedAnswer": "Reciprocal Rank Fusion assigns scores...",
    "expectedChunks": ["chunk-a", "chunk-b"]
  }
]
```

### Running from CLI

```bash
gz-context eval ./eval-suites/smoke.json
gz-context eval ./eval-suites/full.json --json -c 10
```

### Running from Code

```ts
// Copyright (c) 2026 Ground Zero LLC.
import { EvalSuite } from '@ground-zero-llc/gz-context-engine'

const suite = new EvalSuite(engine)
await suite.load(cases)
const report = await suite.run({ concurrency: 5 })

// Report includes per-case metrics + summary
report.summary.meanPrecision     // 0.82
report.summary.meanRecall        // 0.91
report.summary.meanFaithfulness  // 0.88
report.summary.meanRelevance     // 0.79
report.summary.p50LatencyMs      // 1340
report.summary.p99LatencyMs      // 2800
```

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **Precision@K** | Fraction of retrieved chunks that are in the expected set |
| **Recall@K** | Fraction of expected chunks that were actually retrieved |
| **Faithfulness** | How well the generated answer aligns with retrieved context |
| **Relevance** | How well the answer addresses the original query |

## Development

```bash
# Clone
git clone https://github.com/oke3/gz-context-engine.git
cd gz-context-engine

# Install
bun install

# Build
bun run build

# Dev (watch mode)
bun run dev

# Type check
bun run lint

# Test
bun run test
```

## License

MIT — Ground Zero LLC

---

Built by [Ground Zero LLC](https://github.com/oke3) — AI infrastructure for the agentic age.
