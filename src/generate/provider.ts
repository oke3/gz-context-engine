// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * Generation providers for the context engine.
 *
 * Each provider implements `GenerationProvider` and calls a specific
 * LLM backend via raw `fetch()`. No SDK dependencies — keeps the
 * bundle small and gives us full control over request/response shapes.
 *
 * Supported providers:
 *   - `'openai'`   — OpenAI Chat Completions API (requires `OPENAI_API_KEY`)
 *   - `'anthropic'` — Anthropic Messages API (requires `ANTHROPIC_API_KEY`)
 *   - `'local'`     — Deterministic mock response (no API key)
 */

import type {
  GenerationProvider,
  GenerationRequest,
  GenerationResponse,
  ContextItem,
} from '../core/types.js'

// ── Factory ──────────────────────────────────────────────────────────────────

/**
 * Create a generation provider by name.
 *
 * @param providerName - One of `'openai'`, `'anthropic'`, `'local'`.
 * @throws {Error} If the provider name is unknown or a required env var is missing.
 */
export function createGenerationProvider(
  providerName: string,
): GenerationProvider {
  switch (providerName) {
    case 'openai':
      return new OpenAIGenerationProvider()
    case 'anthropic':
      return new AnthropicGenerationProvider()
    case 'local':
      return new LocalGenerationProvider()
    default:
      throw new Error(
        `Unknown generation provider: '${providerName}'. ` +
          `Supported providers: openai, anthropic, local.`,
      )
  }
}

// ── Prompt Construction ──────────────────────────────────────────────────────

/**
 * Build the system prompt that instructs the LLM to use retrieved
 * context and cite sources. The prompt embeds context items as
 * numbered references so the model can reference them in its answer.
 */
function buildSystemPrompt(
  context: ContextItem[],
  userSystemPrompt?: string,
): string {
  const parts: string[] = []

  if (userSystemPrompt) {
    parts.push(userSystemPrompt)
  }

  parts.push(
    'You are a helpful assistant. Answer the user\'s question using ONLY ' +
      'the provided context references below. If the context does not contain ' +
      'enough information to answer the question, say so honestly.',
  )

  if (context.length > 0) {
    parts.push('')
    parts.push('## Context References')
    parts.push('')
    for (const item of context) {
      parts.push(`${item.citation}`)
      parts.push(item.content)
      parts.push('')
    }
    parts.push(
      'Cite sources inline using the reference numbers (e.g., [1], [2]). ' +
        'Always attribute claims to their source references.',
    )
  }

  return parts.join('\n')
}

/**
 * Format context items into a numbered reference block for the user message.
 * Used by providers that prefer context in the user message rather than
 * the system prompt (e.g., Anthropic with long contexts).
 */
function formatContextForUser(context: ContextItem[]): string {
  if (context.length === 0) return ''

  const lines: string[] = ['## Retrieved Context', '']
  for (const item of context) {
    lines.push(`${item.citation}`)
    lines.push(item.content)
    lines.push('')
  }
  return lines.join('\n')
}

// ── Pricing Table ────────────────────────────────────────────────────────────

/**
 * Cost per 1M tokens (input / output) for known models.
 * Updated as of 2026 pricing. Unknown models default to 0.
 */
const MODEL_PRICING: Record<
  string,
  { input: number; output: number }
> = {
  // OpenAI
  'gpt-4o':            { input: 2.50,  output: 10.00 },
  'gpt-4o-mini':       { input: 0.15,  output: 0.60 },
  'gpt-4-turbo':       { input: 10.00, output: 30.00 },
  'gpt-3.5-turbo':     { input: 0.50,  output: 1.50 },
  'o1':                { input: 15.00, output: 60.00 },
  'o1-mini':           { input: 3.00,  output: 12.00 },
  'o3-mini':           { input: 1.10,  output: 4.40 },
  // Anthropic
  'claude-sonnet-4-20250514': { input: 3.00,  output: 15.00 },
  'claude-3-5-sonnet-20241022': { input: 3.00, output: 15.00 },
  'claude-3-5-haiku-20241022':  { input: 0.80, output: 4.00 },
  'claude-3-haiku-20240307':    { input: 0.25, output: 1.25 },
}

/**
 * Calculate the cost in USD for a given model and token counts.
 * Falls back to $0 for unknown models.
 */
function calculateCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
): number {
  const pricing = MODEL_PRICING[model]
  if (!pricing) return 0

  const inputCost = (inputTokens / 1_000_000) * pricing.input
  const outputCost = (outputTokens / 1_000_000) * pricing.output
  return Math.round((inputCost + outputCost) * 1_000_000) / 1_000_000
}

// ── Provider: OpenAI ─────────────────────────────────────────────────────────

/**
 * Calls OpenAI's Chat Completions API directly via `fetch()`.
 * Supports both streaming (SSE) and non-streaming modes.
 */
class OpenAIGenerationProvider implements GenerationProvider {
  readonly name = 'openai'

  private apiKey: string

  constructor() {
    const key = process.env.OPENAI_API_KEY
    if (!key) {
      throw new Error(
        'OPENAI_API_KEY environment variable is required for the openai generation provider.',
      )
    }
    this.apiKey = key
  }

  async generate(request: GenerationRequest): Promise<GenerationResponse> {
    const start = performance.now()
    const model = request.model ?? 'gpt-4o-mini'
    const systemPrompt = buildSystemPrompt(
      request.context,
      request.systemPrompt,
    )

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: request.query },
        ],
        temperature: request.temperature ?? 0.1,
        max_tokens: request.maxTokens ?? 2048,
        stream: false,
      }),
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '<no body>')
      throw new Error(
        `OpenAI generation request failed: ${response.status} ${response.statusText} — ${body}`,
      )
    }

    const data = (await response.json()) as {
      choices: Array<{ message: { content: string } }>
      usage?: { prompt_tokens: number; completion_tokens: number }
      model: string
    }

    const text = data.choices?.[0]?.message?.content ?? ''
    const inputTokens = data.usage?.prompt_tokens ?? 0
    const outputTokens = data.usage?.completion_tokens ?? 0
    const latencyMs = performance.now() - start

    return {
      text,
      model: data.model ?? model,
      tokens: { input: inputTokens, output: outputTokens },
      cost: calculateCost(model, inputTokens, outputTokens),
      latencyMs: Math.round(latencyMs),
      citations: request.context.map((c) => c.citation),
    }
  }

  async *generateStream(
    request: GenerationRequest,
  ): AsyncGenerator<string> {
    const model = request.model ?? 'gpt-4o-mini'
    const systemPrompt = buildSystemPrompt(
      request.context,
      request.systemPrompt,
    )

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: request.query },
        ],
        temperature: request.temperature ?? 0.1,
        max_tokens: request.maxTokens ?? 2048,
        stream: true,
      }),
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '<no body>')
      throw new Error(
        `OpenAI streaming request failed: ${response.status} ${response.statusText} — ${body}`,
      )
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body for streaming')

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        // Keep the last (potentially incomplete) line in the buffer
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data: ')) continue

          const payload = trimmed.slice(6)
          if (payload === '[DONE]') return

          try {
            const parsed = JSON.parse(payload) as {
              choices: Array<{
                delta?: { content?: string }
                finish_reason?: string
              }>
            }
            const content = parsed.choices?.[0]?.delta?.content
            if (content) yield content
          } catch {
            // Skip malformed JSON lines (heartbeats, etc.)
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }
}

// ── Provider: Anthropic ──────────────────────────────────────────────────────

/**
 * Calls Anthropic's Messages API directly via `fetch()`.
 * Uses `x-api-key` header (not `Authorization: Bearer`).
 * Supports streaming via SSE.
 */
class AnthropicGenerationProvider implements GenerationProvider {
  readonly name = 'anthropic'

  private apiKey: string

  constructor() {
    const key = process.env.ANTHROPIC_API_KEY
    if (!key) {
      throw new Error(
        'ANTHROPIC_API_KEY environment variable is required for the anthropic generation provider.',
      )
    }
    this.apiKey = key
  }

  async generate(request: GenerationRequest): Promise<GenerationResponse> {
    const start = performance.now()
    const model = request.model ?? 'claude-3-5-haiku-20241022'

    // Anthropic prefers the system prompt as a top-level parameter,
    // and puts context references in the user message.
    const systemPrompt = buildSystemPrompt(
      [],
      request.systemPrompt,
    )
    const contextBlock = formatContextForUser(request.context)

    const userContent = contextBlock
      ? `${contextBlock}\n\n## Question\n\n${request.query}`
      : request.query

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: request.maxTokens ?? 2048,
        temperature: request.temperature ?? 0.1,
        system: systemPrompt,
        messages: [{ role: 'user', content: userContent }],
      }),
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '<no body>')
      throw new Error(
        `Anthropic generation request failed: ${response.status} ${response.statusText} — ${body}`,
      )
    }

    const data = (await response.json()) as {
      content: Array<{ type: string; text: string }>
      usage: { input_tokens: number; output_tokens: number }
      model: string
    }

    const text =
      data.content?.filter((b) => b.type === 'text').map((b) => b.text).join('') ?? ''
    const inputTokens = data.usage?.input_tokens ?? 0
    const outputTokens = data.usage?.output_tokens ?? 0
    const latencyMs = performance.now() - start

    return {
      text,
      model: data.model ?? model,
      tokens: { input: inputTokens, output: outputTokens },
      cost: calculateCost(model, inputTokens, outputTokens),
      latencyMs: Math.round(latencyMs),
      citations: request.context.map((c) => c.citation),
    }
  }

  async *generateStream(
    request: GenerationRequest,
  ): AsyncGenerator<string> {
    const model = request.model ?? 'claude-3-5-haiku-20241022'

    const systemPrompt = buildSystemPrompt([], request.systemPrompt)
    const contextBlock = formatContextForUser(request.context)
    const userContent = contextBlock
      ? `${contextBlock}\n\n## Question\n\n${request.query}`
      : request.query

    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        max_tokens: request.maxTokens ?? 2048,
        temperature: request.temperature ?? 0.1,
        system: systemPrompt,
        messages: [{ role: 'user', content: userContent }],
        stream: true,
      }),
    })

    if (!response.ok) {
      const body = await response.text().catch(() => '<no body>')
      throw new Error(
        `Anthropic streaming request failed: ${response.status} ${response.statusText} — ${body}`,
      )
    }

    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body for streaming')

    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data: ')) continue

          const payload = trimmed.slice(6)
          try {
            const parsed = JSON.parse(payload) as {
              type: string
              delta?: { type?: string; text?: string }
            }

            // Anthropic sends content_block_delta events for text
            if (
              parsed.type === 'content_block_delta' &&
              parsed.delta?.type === 'text_delta' &&
              parsed.delta.text
            ) {
              yield parsed.delta.text
            }
          } catch {
            // Skip malformed JSON lines
          }
        }
      }
    } finally {
      reader.releaseLock()
    }
  }
}

// ── Provider: Local (mock) ──────────────────────────────────────────────────

/**
 * Returns a deterministic mock response for development/testing.
 * No API key required. Simulates latency and token counting.
 */
class LocalGenerationProvider implements GenerationProvider {
  readonly name = 'local'

  async generate(request: GenerationRequest): Promise<GenerationResponse> {
    // Simulate realistic latency
    await sleep(50 + Math.random() * 100)

    const contextSummary =
      request.context.length > 0
        ? `Based on ${request.context.length} reference(s) (${request.context.map((c) => c.citation).join(', ')}), `
        : ''

    const text =
      `${contextSummary}` +
      `here is a mock answer to: "${request.query.slice(0, 80)}${request.query.length > 80 ? '...' : ''}"\n\n` +
      `This response was generated by the local mock provider. ` +
      `In production, replace this with a real LLM provider (openai/anthropic).`

    // Rough token estimate
    const inputTokens = estimateTokens(request.query) + estimateContextTokens(request.context)
    const outputTokens = estimateTokens(text)

    return {
      text,
      model: request.model ?? 'local-mock',
      tokens: { input: inputTokens, output: outputTokens },
      cost: 0,
      latencyMs: 50,
      citations: request.context.map((c) => c.citation),
    }
  }

  async *generateStream(
    request: GenerationRequest,
  ): AsyncGenerator<string> {
    const response = await this.generate(request)
    // Simulate streaming by yielding word-by-word
    const words = response.text.split(/(\s+)/)
    for (const word of words) {
      await sleep(10)
      yield word
    }
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/** Rough token estimate — whitespace-split word count. */
function estimateTokens(text: string): number {
  const trimmed = text.trim()
  if (trimmed.length === 0) return 0
  return trimmed.split(/\s+/).length
}

/** Sum of token counts across context items. */
function estimateContextTokens(context: ContextItem[]): number {
  return context.reduce((sum, c) => sum + c.tokenCount, 0)
}
