// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

/**
 * CLI for gz-context-engine.
 *
 * Commands:
 *   gz-context ingest <file|dir>     Ingest file(s) or directory
 *   gz-context search <query>        Hybrid search
 *   gz-context ask <question>        End-to-end RAG query
 *   gz-context status                Show engine status
 *   gz-context eval <suite.json>     Run evaluation suite
 *   gz-context mcp                   Start MCP server (stdio transport)
 *
 * Options:
 *   --config <path>   Custom config file
 *   --top-k <n>       Override top K
 *   --model <name>    Override generation model
 *   --json            Output as JSON
 */

import { Command } from 'commander'
import { readFile, stat, readdir } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { homedir } from 'node:os'
import { ContextEngine } from './core/engine.js'
import { buildConfig } from './core/config.js'
import type { ContextEngineConfig } from './core/types.js'

// ── ANSI Colors (zero dependencies) ──────────────────────────────────────────

const ESC = '\x1b['
const c = {
  reset:     `${ESC}0m`,
  bold:      `${ESC}1m`,
  dim:       `${ESC}2m`,
  red:       `${ESC}31m`,
  green:     `${ESC}32m`,
  yellow:    `${ESC}33m`,
  blue:      `${ESC}34m`,
  magenta:   `${ESC}35m`,
  cyan:      `${ESC}36m`,
  gray:      `${ESC}90m`,
}

const bold = (t: string) => `${c.bold}${t}${c.reset}`
const green = (t: string) => `${c.green}${t}${c.reset}`
const red = (t: string) => `${c.red}${t}${c.reset}`
const cyan = (t: string) => `${c.cyan}${t}${c.reset}`
const dim = (t: string) => `${c.dim}${t}${c.reset}`
const yellow = (t: string) => `${c.yellow}${t}${c.reset}`
const magenta = (t: string) => `${c.magenta}${t}${c.reset}`

// ── Helpers ──────────────────────────────────────────────────────────────────

function resolveHome(p: string): string {
  if (p.startsWith('~/') || p === '~') {
    return join(homedir(), p.slice(2))
  }
  return resolve(p)
}

async function collectFiles(dirPath: string): Promise<string[]> {
  const entries = await readdir(dirPath, { withFileTypes: true })
  const files: string[] = []
  for (const entry of entries) {
    const fullPath = join(dirPath, entry.name)
    if (entry.isDirectory()) {
      if (!entry.name.startsWith('.') && entry.name !== 'node_modules') {
        files.push(...(await collectFiles(fullPath)))
      }
    } else if (entry.isFile()) {
      files.push(fullPath)
    }
  }
  return files
}

function fmt(n: number): string {
  return n.toLocaleString('en-US')
}

// ── Config Loading ───────────────────────────────────────────────────────────

interface GlobalOptions {
  config?: string
  topK?: number
  model?: string
  json?: boolean
  method?: string
  concurrency?: number
}

async function createEngine(opts: GlobalOptions): Promise<ContextEngine> {
  const overrides: Partial<ContextEngineConfig> = {}

  if (opts.config) {
    const configPath = resolveHome(opts.config)
    try {
      const raw = await readFile(configPath, 'utf-8')
      const fileConfig = JSON.parse(raw) as Partial<ContextEngineConfig>
      Object.assign(overrides, fileConfig)
    } catch (err) {
      console.error(red(`Error: Could not read config file: ${configPath}`))
      if (err instanceof Error) console.error(dim(err.message))
      process.exit(1)
    }
  }

  if (opts.topK !== undefined) {
    overrides.retrieval = {
      ...(overrides.retrieval ?? {}),
      denseTopK: opts.topK,
      bm25TopK: opts.topK,
      rerankTopK: opts.topK,
    } as ContextEngineConfig['retrieval']
  }

  if (opts.model) {
    overrides.generation = {
      ...(overrides.generation ?? {}),
      model: opts.model,
    } as ContextEngineConfig['generation']
  }

  const config = await buildConfig(
    Object.keys(overrides).length > 0 ? overrides : undefined,
  )
  return new ContextEngine(config)
}

// ── CLI Definition ───────────────────────────────────────────────────────────

const program = new Command()
  .name('gz-context')
  .description(
    'Production-grade context engine for AI agents — hybrid RAG, reranking, token-aware assembly.',
  )
  .version('0.1.0')
  .option('--config <path>', 'Custom config file path')
  .option('--top-k <n>', 'Override top K for retrieval', (v: string) => parseInt(v, 10))
  .option('--model <name>', 'Override generation model')
  .option('--json', 'Output as JSON')
  .option('--method <method>', 'Search method: hybrid, dense, bm25')
  .option('-c, --concurrency <n>', 'Max concurrent eval cases', (v: string) => parseInt(v, 10))

/**
 * Read global options from the root program.
 * Commander binds `this` for zero-arg commands but passes it as a
 * parameter for commands with args. Using `program.opts()` avoids
 * this inconsistency entirely.
 */
function getOpts(): GlobalOptions {
  return program.opts() as GlobalOptions
}

// ── Command: ingest ──────────────────────────────────────────────────────────

program
  .command('ingest <path>')
  .description('Ingest a file or directory into the knowledge base')
  .action(async (inputPath: string) => {
    const opts = getOpts()
    const engine = await createEngine(opts)
    const json = opts.json === true

    try {
      const resolved = resolveHome(inputPath)
      const info = await stat(resolved)

      let files: string[] = []
      if (info.isDirectory()) {
        if (!json) console.log(dim(`Scanning directory: ${resolved}`))
        files = await collectFiles(resolved)
        if (files.length === 0) {
          if (json) {
            console.log(JSON.stringify({ files: 0, documents: 0 }))
          } else {
            console.log(yellow('No files found in directory.'))
          }
          return
        }
      } else {
        files = [resolved]
      }

      if (!json) {
        console.log(bold(`\nIngesting ${fmt(files.length)} file(s)...\n`))
      }

      const results: Array<{ source: string; docId: string; chunks: number }> = []

      for (const file of files) {
        try {
          const content = await readFile(file, 'utf-8')
          const doc = await engine.ingest(file, content)

          const store = await (engine as any).store.get()
          const chunks = await store.getChunksByDocumentId(doc.id)

          results.push({ source: file, docId: doc.id, chunks: chunks.length })

          if (!json) {
            console.log(
              `  ${green('✓')} ${file} → ${cyan(fmt(chunks.length))} chunks`,
            )
          }
        } catch (err) {
          if (!json) {
            console.error(
              `  ${red('✗')} ${file} — ${err instanceof Error ? err.message : err}`,
            )
          }
        }
      }

      if (json) {
        console.log(
          JSON.stringify(
            {
              files: files.length,
              documents: results.length,
              chunks: results.reduce((sum, r) => sum + r.chunks, 0),
              results,
            },
            null,
            2,
          ),
        )
      } else {
        const totalChunks = results.reduce((sum, r) => sum + r.chunks, 0)
        console.log(
          `\n${bold('Done:')} ${green(fmt(results.length))} documents, ` +
            `${cyan(fmt(totalChunks))} chunks created.`,
        )
      }
    } catch (err) {
      if (!json) {
        console.error(red(`\nError: ${err instanceof Error ? err.message : err}`))
      } else {
        console.log(JSON.stringify({ error: String(err) }))
      }
      process.exitCode = 1
    } finally {
      await engine.close()
    }
  })

// ── Command: search ──────────────────────────────────────────────────────────

program
  .command('search <query>')
  .description('Hybrid search across the knowledge base')
  .action(async (query: string) => {
    const opts = getOpts() as GlobalOptions
    const engine = await createEngine(opts)
    const json = opts.json === true

    try {
      const method = (opts.method ?? 'hybrid') as 'hybrid' | 'dense' | 'bm25'
      const topK = opts.topK ?? 5
      const results = await engine.search(query, { topK, method })

      if (json) {
        console.log(JSON.stringify(results, null, 2))
        return
      }

      if (results.length === 0) {
        console.log(yellow('No results found.'))
        return
      }

      console.log(bold(`\n${fmt(results.length)} results (${method}):\n`))

      for (let i = 0; i < results.length; i++) {
        const r = results[i]
        const score = Math.round(r.score * 1000) / 1000
        const preview = r.content.slice(0, 120).replace(/\n/g, ' ')

        console.log(`  ${magenta(`[${i + 1}]`)} score=${cyan(String(score))} method=${dim(r.method)}`)
        console.log(`      doc=${dim(r.documentId.slice(0, 12))}`)
        console.log(`      ${preview}${r.content.length > 120 ? '...' : ''}`)
        console.log()
      }
    } catch (err) {
      if (!json) {
        console.error(red(`\nError: ${err instanceof Error ? err.message : err}`))
      } else {
        console.log(JSON.stringify({ error: String(err) }))
      }
      process.exitCode = 1
    } finally {
      await engine.close()
    }
  })

// ── Command: ask ─────────────────────────────────────────────────────────────

program
  .command('ask <question>')
  .description('End-to-end RAG query: search + retrieve + generate')
  .action(async (question: string) => {
    const opts = getOpts()
    const engine = await createEngine(opts)
    const json = opts.json === true

    try {
      if (!json) {
        console.log(dim(`\nSearching and generating answer...\n`))
      }

      const response = await engine.query({
        query: question,
        context: [],
      })

      if (json) {
        console.log(JSON.stringify(response, null, 2))
        return
      }

      console.log(bold('Answer:'))
      console.log(response.text)
      console.log()

      if (response.citations.length > 0) {
        console.log(dim('Citations:'))
        for (const cite of response.citations) {
          console.log(`  ${dim('•')} ${cyan(cite)}`)
        }
      }

      console.log(
        dim(
          `\nModel: ${response.model} | Tokens: ${response.tokens.input}+${response.tokens.output} | Cost: $${response.cost.toFixed(6)} | ${response.latencyMs}ms`,
        ),
      )
    } catch (err) {
      if (!json) {
        console.error(red(`\nError: ${err instanceof Error ? err.message : err}`))
      } else {
        console.log(JSON.stringify({ error: String(err) }))
      }
      process.exitCode = 1
    } finally {
      await engine.close()
    }
  })

// ── Command: status ──────────────────────────────────────────────────────────

program
  .command('status')
  .description('Show engine status (config, chunk count, doc count)')
  .action(async () => {
    const opts = getOpts()
    const engine = await createEngine(opts)
    const json = opts.json === true

    try {
      const store = await (engine as any).store.get()
      const db = (store as any).db

      let docCount = 0
      let chunkCount = 0
      let embeddingCount = 0

      if (db) {
        docCount = db.prepare('SELECT COUNT(*) as cnt FROM documents').get()?.cnt ?? 0
        chunkCount = db.prepare('SELECT COUNT(*) as cnt FROM chunks').get()?.cnt ?? 0
        embeddingCount = db.prepare('SELECT COUNT(*) as cnt FROM embeddings').get()?.cnt ?? 0
      }

      const config = (engine as any).config

      const status = {
        documents: docCount,
        chunks: chunkCount,
        embeddings: embeddingCount,
        config: {
          embedding: { provider: config.embedding.provider, model: config.embedding.model },
          generation: { provider: config.generation.provider, model: config.generation.model },
          retrieval: config.retrieval,
          chunking: config.chunking,
        },
      }

      if (json) {
        console.log(JSON.stringify(status, null, 2))
        return
      }

      console.log(bold('\ngz-context-engine status\n'))
      console.log(`  ${bold('Documents:')}    ${cyan(fmt(docCount))}`)
      console.log(`  ${bold('Chunks:')}       ${cyan(fmt(chunkCount))}`)
      console.log(`  ${bold('Embeddings:')}   ${cyan(fmt(embeddingCount))}`)
      console.log()
      console.log(dim('Configuration:'))
      console.log(`  Embedding:  ${config.embedding.provider} / ${config.embedding.model}`)
      console.log(`  Generation: ${config.generation.provider} / ${config.generation.model}`)
      console.log(`  Retrieval:  dense=${config.retrieval.denseTopK} bm25=${config.retrieval.bm25TopK} rerank=${config.retrieval.rerankTopK}`)
      console.log(`  Chunking:   strategy=${config.chunking.strategy} max=${config.chunking.maxChunkTokens} overlap=${config.chunking.overlapTokens}`)
      console.log()
    } catch (err) {
      if (!json) {
        console.error(red(`\nError: ${err instanceof Error ? err.message : err}`))
      } else {
        console.log(JSON.stringify({ error: String(err) }))
      }
      process.exitCode = 1
    } finally {
      await engine.close()
    }
  })

// ── Command: eval ────────────────────────────────────────────────────────────

program
  .command('eval <suite>')
  .description('Run an evaluation suite from a JSON file')
  .option('-c, --concurrency <n>', 'Max concurrent cases', (v: string) => parseInt(v, 10))
  .action(async (suitePath: string) => {
    const opts = getOpts() as GlobalOptions
    const engine = await createEngine(opts)
    const json = opts.json === true

    try {
      const resolved = resolveHome(suitePath)
      const raw = await readFile(resolved, 'utf-8')
      const cases = JSON.parse(raw) as import('./core/types.js').EvalCase[]

      if (!Array.isArray(cases) || cases.length === 0) {
        if (!json) console.log(yellow('No test cases found in suite.'))
        return
      }

      const { EvalSuite } = await import('./eval/harness.js')
      const evalSuite = new EvalSuite(engine)

      if (!json) {
        console.log(bold(`\nRunning ${fmt(cases.length)} evaluation case(s)...\n`))
      }

      await evalSuite.load(cases)
      const report = await evalSuite.run({
        concurrency: opts.concurrency,
      })

      if (json) {
        console.log(JSON.stringify(report, null, 2))
        return
      }

      console.log(bold('Evaluation Report\n'))
      console.log(`  Cases:                 ${cyan(fmt(report.cases.length))}`)
      console.log(`  Mean Precision@K:      ${cyan(report.summary.meanPrecision.toFixed(4))}`)
      console.log(`  Mean Recall@K:         ${cyan(report.summary.meanRecall.toFixed(4))}`)
      console.log(`  Mean Faithfulness:     ${cyan(report.summary.meanFaithfulness.toFixed(4))}`)
      console.log(`  Mean Relevance:        ${cyan(report.summary.meanRelevance.toFixed(4))}`)
      console.log(`  p50 Latency:           ${cyan(fmt(Math.round(report.summary.p50LatencyMs)))}ms`)
      console.log(`  p99 Latency:           ${cyan(fmt(Math.round(report.summary.p99LatencyMs)))}ms`)
      console.log(`  Total Cost:            ${cyan(`$${report.summary.totalCost.toFixed(6)}`)}`)
      console.log()

      // Per-case breakdown
      for (const result of report.cases) {
        const status =
          result.metrics.precision > 0.5 ? green('PASS') : yellow('LOW')
        console.log(
          `  ${status} "${result.case.query.slice(0, 60)}${result.case.query.length > 60 ? '...' : ''}"`,
        )
        console.log(
          `      P=${result.metrics.precision.toFixed(2)} R=${result.metrics.recall.toFixed(2)} F=${result.metrics.faithfulness.toFixed(2)} Rl=${result.metrics.relevance.toFixed(2)} ${dim(`${result.latencyMs}ms`)}`,
        )
      }
      console.log()
    } catch (err) {
      if (!json) {
        console.error(red(`\nError: ${err instanceof Error ? err.message : err}`))
      } else {
        console.log(JSON.stringify({ error: String(err) }))
      }
      process.exitCode = 1
    } finally {
      await engine.close()
    }
  })

// ── Command: mcp ─────────────────────────────────────────────────────────────

program
  .command('mcp')
  .description('Start MCP server (stdio transport)')
  .action(async () => {
    const opts = getOpts()
    const engine = await createEngine(opts)

    try {
      const { startMcpServer } = await import('./server/mcp.js')
      await startMcpServer(engine)
    } catch (err) {
      console.error(red(`MCP server error: ${err instanceof Error ? err.message : err}`))
      await engine.close()
      process.exit(1)
    }
  })

// ── Parse & Execute ──────────────────────────────────────────────────────────

program.parse(process.argv)
