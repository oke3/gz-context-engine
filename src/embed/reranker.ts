// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * Reranker providers for the context engine.
 *
 * After initial retrieval (dense + BM25), a reranker re-scores the
 * top candidates with a more powerful model to improve precision.
 */

import type { RerankerProvider } from '../core/types.js'

// ── Factory ──

/**
 * Create a reranker provider by name.
 *
 * Supported providers:
 *   - `'cohere'`    — Cohere Rerank v3 API (requires `COHERE_API_KEY`)
 *   - `'bm25-only'` — No-op passthrough (keeps original order)
 *
 * @throws {Error} If the provider name is unknown or a required env var is missing.
 */
export function createRerankerProvider(
  providerName: string,
): RerankerProvider {
  switch (providerName) {
    case 'cohere':
      return new CohereRerankerProvider()
    case 'bm25-only':
      return new BM25OnlyReranker()
    default:
      throw new Error(
        `Unknown reranker provider: '${providerName}'. ` +
          `Supported providers: cohere, bm25-only.`,
      )
  }
}

// ── Provider: Cohere Rerank v3 ──

/**
 * Calls Cohere's `/v2/rerank` endpoint directly via `fetch()`.
 * Returns relevance scores (0–1) for each document against the query.
 */
class CohereRerankerProvider implements RerankerProvider {
  readonly name = 'cohere'

  private apiKey: string
  private model = 'rerank-english-v3.0'

  constructor() {
    const key = process.env.COHERE_API_KEY
    if (!key) {
      throw new Error(
        'COHERE_API_KEY environment variable is required for the cohere reranker provider.',
      )
    }
    this.apiKey = key
  }

  async rerank(
    query: string,
    documents: string[],
    topK: number,
  ): Promise<number[]> {
    if (documents.length === 0) return []

    // Cohere Rerank v2 API.
    const response = await fetch('https://api.cohere.com/v2/rerank', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        model: this.model,
        query,
        documents,
        top_n: Math.min(topK, documents.length),
        return_documents: false,
      }),
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '<no body>')
      throw new Error(
        `Cohere rerank request failed: ${response.status} ${response.statusText} — ${body}`,
      )
    }

    const data = (await response.json()) as {
      results: Array<{ index: number; relevance_score: number }>
    }

    // Cohere returns only `top_n` results. Map back to full-document ordering:
    // documents not in the result get a score of 0.
    const scores = new Array<number>(documents.length).fill(0)
    for (const r of data.results) {
      scores[r.index] = r.relevance_score
    }

    return scores
  }
}

// ── Provider: BM25-only (no-op) ──

/**
 * Passthrough reranker that preserves the original document order.
 * Use when you don't have a reranker API key — the retrieval layer's
 * BM25 ranking will be the sole signal.
 */
class BM25OnlyReranker implements RerankerProvider {
  readonly name = 'bm25-only'

  async rerank(
    _query: string,
    documents: string[],
    _topK: number,
  ): Promise<number[]> {
    // Assign descending scores so the original order is preserved.
    // Index 0 gets the highest score, last index gets the lowest.
    return documents.map((_, i) => 1 - i / Math.max(documents.length, 1))
  }
}
