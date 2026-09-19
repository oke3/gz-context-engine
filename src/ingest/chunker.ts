// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * Text chunking strategies for the ingest pipeline.
 *
 * Each strategy splits a document's content into token-bounded chunks
 * while preserving natural text boundaries wherever possible.
 */

import type { Chunk, Chunker } from '../core/types.js'

// ── Helpers ──

/** Rough token estimate — whitespace-split word count. */
function estimateTokens(text: string): number {
  const trimmed = text.trim()
  if (trimmed.length === 0) return 0
  return trimmed.split(/\s+/).length
}

/** Strip leading/trailing whitespace, collapse internal runs. */
function normalize(text: string): string {
  return text.replace(/\n{3,}/g, '\n\n').trim()
}

/** True when the string is only whitespace / newlines. */
function isBlank(text: string): boolean {
  return text.trim().length === 0
}

// ── Factory ──

export type ChunkingStrategy = 'recursive' | 'markdown' | 'fixed'

/**
 * Create a `Chunker` backed by the chosen strategy.
 *
 * @param strategy       Splitting strategy (default: `'recursive'`).
 * @param maxChunkTokens Soft ceiling on tokens per chunk.
 * @param overlapTokens  Tokens of overlap carried forward between chunks.
 */
export async function createChunker(
  strategy: ChunkingStrategy | string = 'recursive',
  maxChunkTokens = 512,
  overlapTokens = 64,
): Promise<Chunker> {
  switch (strategy) {
    case 'markdown':
      return new MarkdownChunker(maxChunkTokens, overlapTokens)
    case 'fixed':
      return new FixedChunker(maxChunkTokens, overlapTokens)
    case 'recursive':
    default:
      return new RecursiveChunker(maxChunkTokens, overlapTokens)
  }
}

// ── Shared utilities ──

/**
 * Given a list of raw text segments, merge consecutive small segments
 * until they approach `maxTokens`, then emit overlapping chunks.
 *
 * This is the core loop shared by all strategies after they produce
 * an initial list of "splittable units."
 */
function assembleChunks(
  units: string[],
  documentId: string,
  maxTokens: number,
  overlapTokens: number,
): Chunk[] {
  if (units.length === 0) return []

  const chunks: Chunk[] = []
  let buffer = ''
  let bufferTokens = 0
  let chunkIndex = 0
  let charOffset = 0

  for (const unit of units) {
    const unitTokens = estimateTokens(unit)

    // If adding this unit would exceed the limit, flush the buffer first.
    if (bufferTokens > 0 && bufferTokens + unitTokens > maxTokens) {
      chunks.push(makeChunk(documentId, chunkIndex++, buffer, charOffset))
      charOffset += buffer.length

      // Compute overlap: walk backwards from the end of the flushed buffer
      // and collect words until we hit overlapTokens.
      const overlapText = tailTokens(buffer, overlapTokens)
      buffer = overlapText + unit
      bufferTokens = estimateTokens(buffer)
      continue
    }

    buffer = buffer.length > 0 ? buffer + '\n\n' + unit : unit
    bufferTokens += unitTokens
  }

  // Flush remaining buffer.
  if (!isBlank(buffer)) {
    chunks.push(makeChunk(documentId, chunkIndex, buffer, charOffset))
  }

  return chunks
}

/** Build a single `Chunk` with sequential id and metadata. */
function makeChunk(
  documentId: string,
  index: number,
  content: string,
  charOffset: number,
): Chunk {
  return {
    id: `${documentId}-chunk-${index}`,
    documentId,
    content: normalize(content),
    index,
    tokenCount: estimateTokens(content),
    metadata: {
      charOffset,
      charLength: content.length,
    },
  }
}

/** Return the last N whitespace-delimited tokens from `text`. */
function tailTokens(text: string, n: number): string {
  if (n <= 0) return ''
  const words = text.split(/\s+/)
  if (words.length <= n) return text
  return words.slice(-n).join(' ') + ' '
}

// ── Strategy: Recursive ──

/**
 * Splits text by paragraphs → sentences → words, recursing into smaller
 * boundaries only when a unit exceeds `maxTokens`.
 */
class RecursiveChunker implements Chunker {
  constructor(
    private maxTokens: number,
    private overlapTokens: number,
  ) {}

  async chunk(content: string, documentId: string): Promise<Chunk[]> {
    if (isBlank(content)) return []

    const normalized = normalize(content)
    const paragraphs = splitParagraphs(normalized)
    const units = paragraphs.flatMap((p) =>
      estimateTokens(p) > this.maxTokens ? splitSentences(p) : [p],
    )
    const finalUnits = units.flatMap((u) =>
      estimateTokens(u) > this.maxTokens ? splitWords(u, this.maxTokens) : [u],
    )

    return assembleChunks(finalUnits, documentId, this.maxTokens, this.overlapTokens)
  }
}

function splitParagraphs(text: string): string[] {
  return text
    .split(/\n\n+/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
}

function splitSentences(text: string): string[] {
  // Heuristic: split on sentence-ending punctuation followed by whitespace.
  // Keeps abbreviations like "Mr." intact by requiring uppercase or quote after dot.
  return text
    .split(/(?<=[.!?])\s+(?=[A-Z"\u201C\u201D])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
}

function splitWords(text: string, maxTokens: number): string[] {
  const words = text.split(/\s+/)
  const result: string[] = []
  for (let i = 0; i < words.length; i += maxTokens) {
    result.push(words.slice(i, i + maxTokens).join(' '))
  }
  return result
}

// ── Strategy: Markdown ──

/**
 * Splits on markdown headers (`#` through `######`). The header line
 * becomes a prefix of every chunk that follows it until the next header.
 * Code blocks, list blocks, and tables are treated as atomic units.
 */
class MarkdownChunker implements Chunker {
  private headerRe = /^(#{1,6})\s+.+$/m

  constructor(
    private maxTokens: number,
    private overlapTokens: number,
  ) {}

  async chunk(content: string, documentId: string): Promise<Chunk[]> {
    if (isBlank(content)) return []

    const normalized = normalize(content)
    const sections = splitMarkdownSections(normalized)

    // Each section may still exceed maxTokens; split recursively within.
    const units: string[] = []
    for (const section of sections) {
      if (estimateTokens(section) <= this.maxTokens) {
        units.push(section)
      } else {
        // Fall back to recursive paragraph/sentence splitting.
        const paragraphs = splitParagraphs(section)
        const subUnits = paragraphs.flatMap((p) =>
          estimateTokens(p) > this.maxTokens ? splitSentences(p) : [p],
        )
        units.push(
          ...subUnits.flatMap((u) =>
            estimateTokens(u) > this.maxTokens ? splitWords(u, this.maxTokens) : [u],
          ),
        )
      }
    }

    return assembleChunks(units, documentId, this.maxTokens, this.overlapTokens)
  }
}

/**
 * Split markdown into sections bounded by headers.
 * Each section includes its header line as the first line.
 */
function splitMarkdownSections(text: string): string[] {
  const lines = text.split('\n')
  const sections: string[] = []
  let current: string[] = []
  let inCodeBlock = false

  for (const line of lines) {
    // Track fenced code blocks — never split inside them.
    if (line.trimStart().startsWith('```')) {
      inCodeBlock = !inCodeBlock
      current.push(line)
      continue
    }

    if (inCodeBlock) {
      current.push(line)
      continue
    }

    // A markdown header starts a new section.
    if (/^#{1,6}\s+/.test(line.trimStart())) {
      if (current.length > 0) {
        sections.push(current.join('\n').trim())
      }
      current = [line]
    } else {
      current.push(line)
    }
  }

  if (current.length > 0) {
    sections.push(current.join('\n').trim())
  }

  return sections.filter((s) => s.length > 0)
}

// ── Strategy: Fixed ──

/**
 * Fixed-size token windows with overlap. Ignores natural boundaries
 * (useful when you want uniform chunk sizes for downstream processing).
 */
class FixedChunker implements Chunker {
  constructor(
    private maxTokens: number,
    private overlapTokens: number,
  ) {}

  async chunk(content: string, documentId: string): Promise<Chunk[]> {
    if (isBlank(content)) return []

    const words = normalize(content).split(/\s+/)
    const chunks: Chunk[] = []
    let start = 0
    let chunkIndex = 0

    while (start < words.length) {
      const end = Math.min(start + this.maxTokens, words.length)
      const chunkWords = words.slice(start, end)
      const chunkText = chunkWords.join(' ')

      chunks.push(makeChunk(documentId, chunkIndex, chunkText, 0))

      // Advance by maxTokens minus overlap to create overlap window.
      const step = this.maxTokens - this.overlapTokens
      start += step > 0 ? step : this.maxTokens
      chunkIndex++
    }

    return chunks
  }
}
