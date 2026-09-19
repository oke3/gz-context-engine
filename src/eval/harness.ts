// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * Evaluation harness for measuring retrieval + generation quality.
 *
 * Provides the `EvalSuite` class that loads test cases, runs them
 * against the ContextEngine, and computes precision, recall,
 * faithfulness, and relevance metrics.
 *
 * Metrics:
 *   - Precision@K:   Fraction of retrieved chunks that are relevant
 *   - Recall@K:      Fraction of expected chunks that were retrieved
 *   - Faithfulness:  Does the generated answer align with retrieved context?
 *   - Relevance:     Does the answer address the query?
 */

import type { EvalCase, EvalResult, SearchResult } from '../core/types.js'
import type { ContextEngine } from '../core/engine.js'

// ── Types ────────────────────────────────────────────────────────────────────

interface EvalReport {
  cases: EvalResult[]
  summary: {
    meanPrecision: number
    meanRecall: number
    meanFaithfulness: number
    meanRelevance: number
    p50LatencyMs: number
    p99LatencyMs: number
    totalCost: number
  }
}

interface EvalRunOptions {
  /** Max concurrent case evaluations (default: 5). */
  concurrency?: number
}

// ── Metric Functions ─────────────────────────────────────────────────────────

/**
 * Precision@K: What fraction of retrieved chunks are relevant?
 *
 * Uses expectedChunks as the ground truth. A retrieved chunk is
 * "relevant" if its ID appears in the expected set.
 *
 * @param retrievedIds - IDs of retrieved chunks (ordered by rank)
 * @param expectedIds  - IDs of expected (relevant) chunks
 * @returns Score between 0 and 1
 */
function precisionAtK(retrievedIds: string[], expectedIds: string[]): number {
  if (retrievedIds.length === 0) return 0

  const expectedSet = new Set(expectedIds)
  let relevantCount = 0

  for (const id of retrievedIds) {
    if (expectedSet.has(id)) {
      relevantCount++
    }
  }

  return relevantCount / retrievedIds.length
}

/**
 * Recall@K: What fraction of expected chunks were actually retrieved?
 *
 * @param retrievedIds - IDs of retrieved chunks
 * @param expectedIds  - IDs of expected (relevant) chunks
 * @returns Score between 0 and 1
 */
function recallAtK(retrievedIds: string[], expectedIds: string[]): number {
  if (expectedIds.length === 0) return 1 // No expectations = trivially satisfied

  const retrievedSet = new Set(retrievedIds)
  let foundCount = 0

  for (const id of expectedIds) {
    if (retrievedSet.has(id)) {
      foundCount++
    }
  }

  return foundCount / expectedIds.length
}

/**
 * Faithfulness: Does the generated answer align with the retrieved context?
 *
 * Simple NLI approximation: check if claims in the answer are supported
 * by the context. Implemented as keyword overlap between the answer and
 * the combined context text.
 *
 * @param answer    - The generated answer text
 * @param context   - The retrieved context text (combined)
 * @returns Score between 0 and 1
 */
function faithfulness(answer: string, context: string): number {
  if (!answer || !context) return 0

  const answerTokens = extractTokens(answer)
  const contextTokens = extractTokens(context)

  if (answerTokens.length === 0) return 1 // Empty answer is trivially faithful
  if (contextTokens.length === 0) return 0

  const contextSet = new Set(contextTokens)
  let supportedCount = 0

  for (const token of answerTokens) {
    if (contextSet.has(token)) {
      supportedCount++
    }
  }

  return supportedCount / answerTokens.length
}

/**
 * Relevance: Does the answer address the query?
 *
 * Measures keyword overlap between the query and the answer.
 * A highly relevant answer should contain many of the query terms.
 *
 * @param query  - The original user query
 * @param answer - The generated answer
 * @returns Score between 0 and 1
 */
function relevance(query: string, answer: string): number {
  if (!query || !answer) return 0

  const queryTokens = extractTokens(query)
  const answerTokens = extractTokens(answer)

  if (queryTokens.length === 0) return 1
  if (answerTokens.length === 0) return 0

  const answerSet = new Set(answerTokens)
  let overlapCount = 0

  for (const token of queryTokens) {
    if (answerSet.has(token)) {
      overlapCount++
    }
  }

  return overlapCount / queryTokens.length
}

// ── Text Processing ──────────────────────────────────────────────────────────

/** Common stop words to exclude from token analysis. */
const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
  'into', 'through', 'during', 'before', 'after', 'above', 'below',
  'between', 'out', 'off', 'over', 'under', 'again', 'further',
  'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
  'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
  'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
  'too', 'very', 'just', 'because', 'but', 'and', 'or', 'if', 'while',
  'that', 'this', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
  'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their',
  'what', 'which', 'who', 'whom',
])

/**
 * Extract normalized tokens from text.
 * Lowercases, splits on non-alphanumeric, removes stop words and
 * tokens shorter than 2 characters.
 */
function extractTokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length >= 2 && !STOP_WORDS.has(t))
}

// ── Percentile Calculation ───────────────────────────────────────────────────

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = Math.ceil((p / 100) * sorted.length) - 1
  return sorted[Math.max(0, idx)]
}

// ── EvalSuite Class ──────────────────────────────────────────────────────────

/**
 * Evaluation suite for measuring retrieval + generation quality.
 *
 * Usage:
 * ```ts
 * const suite = new EvalSuite(engine)
 * await suite.load(cases)
 * const report = await suite.run({ concurrency: 5 })
 * console.log(report.summary)
 * ```
 */
class EvalSuite {
  private engine: ContextEngine
  private cases: EvalCase[] = []

  constructor(engine: ContextEngine) {
    this.engine = engine
  }

  /**
   * Load test cases into the suite.
   */
  async load(cases: EvalCase[]): Promise<void> {
    this.cases = cases
  }

  /**
   * Run all loaded test cases and produce an evaluation report.
   */
  async run(options?: EvalRunOptions): Promise<EvalReport> {
    const concurrency = options?.concurrency ?? 5
    const results: EvalResult[] = []

    // Process cases in batches for concurrency control
    for (let i = 0; i < this.cases.length; i += concurrency) {
      const batch = this.cases.slice(i, i + concurrency)
      const batchResults = await Promise.all(
        batch.map((c) => this.runCase(c)),
      )
      results.push(...batchResults)
    }

    return this.buildReport(results)
  }

  /**
   * Run a single evaluation case.
   */
  private async runCase(evalCase: EvalCase): Promise<EvalResult> {
    const start = performance.now()

    // Step 1: Retrieve relevant chunks
    const searchResults = await this.engine.search(evalCase.query, {
      topK: 10,
    })

    const retrievedChunks = searchResults.map((r) => r.chunkId)

    // Step 2: Generate answer using retrieved context
    let actualAnswer = ''
    try {
      const response = await this.engine.query({
        query: evalCase.query,
        context: [],
      })
      actualAnswer = response.text
    } catch {
      // Generation failure — answer is empty but retrieval still counts
      actualAnswer = '[generation failed]'
    }

    const latencyMs = performance.now() - start

    // Step 3: Compute metrics
    const expectedChunks = evalCase.expectedChunks ?? []

    const p = precisionAtK(retrievedChunks, expectedChunks)
    const r = recallAtK(retrievedChunks, expectedChunks)

    // Faithfulness: compare answer against the combined retrieved context
    const contextText = searchResults.map((sr: SearchResult) => sr.content).join('\n\n')
    const f = faithfulness(actualAnswer, contextText)

    // Relevance: compare answer against the query
    const rel = relevance(evalCase.query, actualAnswer)

    return {
      case: evalCase,
      actualAnswer,
      retrievedChunks,
      metrics: {
        precision: p,
        recall: r,
        faithfulness: f,
        relevance: rel,
      },
      latencyMs: Math.round(latencyMs),
    }
  }

  /**
   * Aggregate individual case results into a summary report.
   */
  private buildReport(results: EvalResult[]): EvalReport {
    if (results.length === 0) {
      return {
        cases: [],
        summary: {
          meanPrecision: 0,
          meanRecall: 0,
          meanFaithfulness: 0,
          meanRelevance: 0,
          p50LatencyMs: 0,
          p99LatencyMs: 0,
          totalCost: 0,
        },
      }
    }

    const n = results.length

    const meanPrecision = results.reduce((s: number, r: EvalResult) => s + r.metrics.precision, 0) / n
    const meanRecall = results.reduce((s: number, r: EvalResult) => s + r.metrics.recall, 0) / n
    const meanFaithfulness = results.reduce((s: number, r: EvalResult) => s + r.metrics.faithfulness, 0) / n
    const meanRelevance = results.reduce((s: number, r: EvalResult) => s + r.metrics.relevance, 0) / n

    const latencies = results.map((r) => r.latencyMs).sort((a, b) => a - b)
    const p50LatencyMs = percentile(latencies, 50)
    const p99LatencyMs = percentile(latencies, 99)

    // Cost is not tracked per-case in EvalResult — sum is 0 unless
    // the caller collects costs separately.
    const totalCost = 0

    return {
      cases: results,
      summary: {
        meanPrecision,
        meanRecall,
        meanFaithfulness,
        meanRelevance,
        p50LatencyMs,
        p99LatencyMs,
        totalCost,
      },
    }
  }
}

// ── Exports ──────────────────────────────────────────────────────────────────

export { EvalSuite }
export type { EvalReport, EvalRunOptions }
export {
  precisionAtK,
  recallAtK,
  faithfulness,
  relevance,
}
