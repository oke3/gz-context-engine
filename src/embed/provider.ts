// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * Embedding providers for the context engine.
 *
 * Each provider implements `EmbeddingProvider` and produces dense
 * vectors for text. The factory `createEmbeddingProvider` wires
 * the correct backend based on a name string.
 */

import type { EmbeddingProvider } from '../core/types.js'

// ── Factory ──

/**
 * Create an embedding provider by name.
 *
 * Supported providers:
 *   - `'openai'`  — OpenAI text-embedding-3-small (requires `OPENAI_API_KEY`)
 *   - `'local'`   — Deterministic hash-based vectors (no API key)
 *   - `'noop'`    — Zero vectors (testing only)
 *
 * @throws {Error} If the provider name is unknown or a required env var is missing.
 */
export function createEmbeddingProvider(
  providerName: string,
): EmbeddingProvider {
  switch (providerName) {
    case 'openai':
      return new OpenAIEmbeddingProvider()
    case 'local':
      return new LocalEmbeddingProvider()
    case 'noop':
      return new NoopEmbeddingProvider()
    default:
      throw new Error(
        `Unknown embedding provider: '${providerName}'. ` +
          `Supported providers: openai, local, noop.`,
      )
  }
}

// ── Provider: OpenAI ──

/**
 * Calls OpenAI's `/v1/embeddings` endpoint directly via `fetch()`.
 * No SDK dependency — keeps the bundle small.
 */
class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'openai'
  readonly dimensions = 1536

  private apiKey: string
  private model = 'text-embedding-3-small'
  private batchSize = 100

  constructor() {
    const key = process.env.OPENAI_API_KEY
    if (!key) {
      throw new Error(
        'OPENAI_API_KEY environment variable is required for the openai embedding provider.',
      )
    }
    this.apiKey = key
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return []

    const results: number[][] = new Array(texts.length)
    const batches = chunkArray(texts, this.batchSize)

    let offset = 0
    for (const batch of batches) {
      const vectors = await this.embedBatch(batch)
      for (let i = 0; i < vectors.length; i++) {
        results[offset + i] = vectors[i]
      }
      offset += batch.length
    }

    return results
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
      }),
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '<no body>')
      throw new Error(
        `OpenAI embedding request failed: ${response.status} ${response.statusText} — ${body}`,
      )
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[]; index: number }>
    }

    // Sort by index to guarantee ordering matches input.
    data.data.sort((a, b) => a.index - b.index)
    return data.data.map((d) => d.embedding)
  }
}

// ── Provider: Local (deterministic hash-based) ──

/**
 * Produces deterministic pseudo-random vectors from text using a
 * simple hash. Useful for development/testing without an API key.
 *
 * The output is NOT semantically meaningful — it only guarantees
 * that identical inputs always produce identical vectors, which
 * is sufficient for testing retrieval logic.
 */
class LocalEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'local'
  readonly dimensions = 384

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((text) => this.hashToVector(text))
  }

  private hashToVector(text: string): number[] {
    const vec = new Array<number>(this.dimensions)
    const seed = this.fnv1a(text)

    // Simple xorshift32 PRNG seeded with the text hash.
    let state = seed === 0 ? 1 : seed
    for (let i = 0; i < this.dimensions; i++) {
      state ^= state << 13
      state ^= state >> 17
      state ^= state << 5
      // Map to [-1, 1] range.
      vec[i] = (state / 0x7fffffff) * 2 - 1
    }

    // Normalize to unit length so cosine similarity is meaningful.
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0))
    if (norm > 0) {
      for (let i = 0; i < vec.length; i++) {
        vec[i] /= norm
      }
    }

    return vec
  }

  /** FNV-1a 32-bit hash. */
  private fnv1a(text: string): number {
    let hash = 0x811c9dc5
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i)
      hash = Math.imul(hash, 0x01000193)
    }
    return hash >>> 0
  }
}

// ── Provider: Noop ──

/**
 * Returns zero vectors of the configured dimension.
 * Use for testing ingestion pipelines without embedding overhead.
 */
class NoopEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'noop'
  readonly dimensions = 1536

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map(() => new Array<number>(this.dimensions).fill(0))
  }
}

// ── Utilities ──

/** Split an array into chunks of `size`. */
function chunkArray<T>(arr: T[], size: number): T[][] {
  const result: T[][] = []
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size))
  }
  return result
}
