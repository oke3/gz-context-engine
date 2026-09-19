// Copyright (c) 2026 Ground Zero LLC. All rights reserved.

import { homedir } from 'node:os'
import { join } from 'node:path'
import { readFile, stat } from 'node:fs/promises'
import type { ContextEngineConfig } from './types.js'
import { DEFAULT_CONFIG } from './types.js'

/**
 * Resolve a path that may start with `~` to an absolute path.
 */
function resolveHomePath(p: string): string {
  if (p.startsWith('~/') || p === '~') {
    return join(homedir(), p.slice(2))
  }
  return p
}

/**
 * Deep-merge two config objects. Source values override target values.
 * Arrays are replaced, not merged.
 */
function deepMerge(
  target: Record<string, unknown>,
  source: Record<string, unknown>,
): Record<string, unknown> {
  const result = { ...target }
  for (const key of Object.keys(source)) {
    const srcVal = source[key]
    const tgtVal = target[key]
    if (
      srcVal !== undefined &&
      typeof srcVal === 'object' &&
      srcVal !== null &&
      !Array.isArray(srcVal) &&
      typeof tgtVal === 'object' &&
      tgtVal !== null &&
      !Array.isArray(tgtVal)
    ) {
      result[key] = deepMerge(
        tgtVal as Record<string, unknown>,
        srcVal as Record<string, unknown>,
      )
    } else if (srcVal !== undefined) {
      result[key] = srcVal
    }
  }
  return result
}

/**
 * Validate a resolved config. Throws a descriptive error on invalid values.
 */
function validateConfig(config: ContextEngineConfig): void {
  if (!config.dataDir || typeof config.dataDir !== 'string') {
    throw new Error('Config: dataDir must be a non-empty string')
  }
  if (config.embedding.dimensions <= 0) {
    throw new Error('Config: embedding.dimensions must be > 0')
  }
  if (config.retrieval.denseTopK <= 0 || config.retrieval.bm25TopK <= 0) {
    throw new Error('Config: retrieval.denseTopK and bm25TopK must be > 0')
  }
  if (config.retrieval.rerankTopK <= 0) {
    throw new Error('Config: retrieval.rerankTopK must be > 0')
  }
  if (config.retrieval.rrfK <= 0) {
    throw new Error('Config: retrieval.rrfK must be > 0')
  }
  if (config.generation.temperature < 0 || config.generation.temperature > 2) {
    throw new Error('Config: generation.temperature must be between 0 and 2')
  }
  if (config.generation.maxTokens <= 0) {
    throw new Error('Config: generation.maxTokens must be > 0')
  }
  if (config.chunking.maxChunkTokens <= 0) {
    throw new Error('Config: chunking.maxChunkTokens must be > 0')
  }
  if (config.chunking.overlapTokens < 0) {
    throw new Error('Config: chunking.overlapTokens must be >= 0')
  }
  if (config.chunking.overlapTokens >= config.chunking.maxChunkTokens) {
    throw new Error(
      'Config: chunking.overlapTokens must be < chunking.maxChunkTokens',
    )
  }
}

/**
 * Load config from a JSON file on disk. Returns null if file doesn't exist.
 */
async function loadConfigFile(
  filePath: string,
): Promise<Record<string, unknown> | null> {
  const resolved = resolveHomePath(filePath)
  try {
    const info = await stat(resolved)
    if (!info.isFile()) return null
  } catch {
    return null
  }
  const raw = await readFile(resolved, 'utf-8')
  return JSON.parse(raw) as Record<string, unknown>
}

/**
 * Build the final config by: defaults ← file ← overrides.
 * The default config file location is `~/.gz-context/config.json`.
 */
export async function buildConfig(
  overrides?: Partial<ContextEngineConfig>,
): Promise<ContextEngineConfig> {
  const defaultConfigPath = join(
    resolveHomePath(DEFAULT_CONFIG.dataDir),
    'config.json',
  )
  const fileConfig = await loadConfigFile(defaultConfigPath)
  let merged = deepMerge(
    DEFAULT_CONFIG as unknown as Record<string, unknown>,
    fileConfig ?? {},
  )
  if (overrides) {
    merged = deepMerge(merged, overrides as Record<string, unknown>)
  }
  const result = merged as unknown as ContextEngineConfig
  // Resolve all tilde paths
  result.dataDir = resolveHomePath(result.dataDir)
  result.store.dbPath = resolveHomePath(result.store.dbPath)
  validateConfig(result)
  return result
}

/**
 * Synchronous config builder for contexts where async isn't available.
 * Uses defaults + overrides only (no file loading).
 */
export function buildConfigSync(
  overrides?: Partial<ContextEngineConfig>,
): ContextEngineConfig {
  let merged = deepMerge(
    DEFAULT_CONFIG as unknown as Record<string, unknown>,
    (overrides ?? {}) as Record<string, unknown>,
  )
  const result = merged as unknown as ContextEngineConfig
  result.dataDir = resolveHomePath(result.dataDir)
  result.store.dbPath = resolveHomePath(result.store.dbPath)
  validateConfig(result)
  return result
}
