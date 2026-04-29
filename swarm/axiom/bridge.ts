import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

export const AXIOM_SWARM_WATERMARK = 'axiom.swarm.webwide.v1'

export type AxiomSourceUrl = {
  url: string
  domain: string
  reason: string
  seeded: boolean
  cached: boolean
}

export type AxiomCrawlPlan = {
  watermark: typeof AXIOM_SWARM_WATERMARK
  intent: 'web_search' | 'learn' | 'fetch'
  query: string
  worker_count: number
  requested_worker_count: number
  target_documents: number
  max_waves: number
  depth: number
  early_stop_score: number
  seed_domains: string[]
  source_urls: AxiomSourceUrl[]
  constraints: {
    one_worker_per_site: true
    no_duplicate_site_fetch: true
    no_external_search_engine: true
    default_worker_ceiling: number
    absolute_worker_limit: number
    lower_compute: string[]
  }
  origin: {
    kind: string
    task_type: string
    agent_name: string
    team_name: string
    message_count: number
  }
}

const DEFAULT_WORKERS = 10
const DEFAULT_MAX_WORKERS = 10
const ABSOLUTE_WORKER_LIMIT = 100
const DEFAULT_TARGET_DOCUMENTS = 12
const DEFAULT_MAX_WAVES = 16
const DEFAULT_EARLY_STOP_SCORE = 12
const DEFAULT_SOURCE_URL_CAP = 128
const DOMAIN_RE = /(?<!@)\b(?:https?:\/\/)?(?<domain>[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]{2,})+)\b/gi
const WORKER_RE = /\bswarm\s*-(?<workers>\d{1,4})\b/i
const DEPTH_RE = /\bdepth\s*-?(?<depth>\d{1,2})\b/i
const SWARM_HEAD_RE = /^\s*swarm(?:\s+-(?<workers>\d{1,4}))?\s*$/i
const DEPTH_SEGMENT_RE = /^\s*depth\s*-?(?<depth>\d{1,2})\s*$/i

type SourceProfile = {
  name?: string
  always?: boolean
  terms?: string[]
  domains?: string[]
}

type SourceConfig = {
  limits?: Record<string, unknown>
  profiles?: SourceProfile[]
  search_templates?: Record<string, string>
}

const fallbackSourceConfig: SourceConfig = {
  limits: {
    default_workers: DEFAULT_WORKERS,
    default_max_workers: DEFAULT_MAX_WORKERS,
    absolute_worker_limit: ABSOLUTE_WORKER_LIMIT,
    target_documents: DEFAULT_TARGET_DOCUMENTS,
    max_waves: DEFAULT_MAX_WAVES,
    early_stop_score: DEFAULT_EARLY_STOP_SCORE,
    max_search_sources: DEFAULT_SOURCE_URL_CAP,
  },
  profiles: [],
  search_templates: {},
}

function sourceConfig(): SourceConfig {
  const explicitPath = process.env.AXIOM_CRAWLER_SOURCE_CONFIG?.trim()
  const configPath = explicitPath || resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'config', 'crawler_sources.json')
  try {
    const parsed = JSON.parse(readFileSync(configPath, 'utf8')) as unknown
    if (!isRecord(parsed)) return fallbackSourceConfig
    return {
      limits: isRecord(parsed.limits) ? parsed.limits : fallbackSourceConfig.limits,
      profiles: Array.isArray(parsed.profiles) ? parsed.profiles.filter(isRecord).map(profile => ({
        name: typeof profile.name === 'string' ? profile.name : undefined,
        always: Boolean(profile.always),
        terms: Array.isArray(profile.terms) ? profile.terms.map(String) : [],
        domains: Array.isArray(profile.domains) ? profile.domains.map(String) : [],
      })) : fallbackSourceConfig.profiles,
      search_templates: isRecord(parsed.search_templates)
        ? Object.fromEntries(Object.entries(parsed.search_templates).map(([domain, template]) => [normalizeDomain(domain), String(template)]).filter(([domain]) => domain))
        : fallbackSourceConfig.search_templates,
    }
  } catch {
    return fallbackSourceConfig
  }
}

function sourceLimits(): Record<string, unknown> {
  return sourceConfig().limits ?? {}
}

function limitFromConfig(name: string, fallback: number): number {
  const value = Number(sourceLimits()[name])
  return Number.isFinite(value) ? value : fallback
}

function configuredInt(envName: string | undefined, limitName: string, fallback: number, low: number, high: number): number {
  const configured = envName ? process.env[envName] : undefined
  const value = configured?.trim() ? configured : limitFromConfig(limitName, fallback)
  return boundedInt(value, fallback, low, high)
}

function configuredFloat(envName: string | undefined, limitName: string, fallback: number, low: number, high: number): number {
  const configured = envName ? process.env[envName] : undefined
  const value = configured?.trim() ? configured : limitFromConfig(limitName, fallback)
  return boundedFloat(value, fallback, low, high)
}

function absoluteWorkerLimit(): number {
  return configuredInt(undefined, 'absolute_worker_limit', ABSOLUTE_WORKER_LIMIT, 1, 1000)
}

function maxWorkerCeiling(): number {
  return configuredInt('AXIOM_CRAWL_MAX_WORKERS', 'default_max_workers', DEFAULT_MAX_WORKERS, 1, absoluteWorkerLimit())
}

function defaultWorkerCount(): number {
  return configuredInt('AXIOM_CRAWL_WORKERS', 'default_workers', DEFAULT_WORKERS, 1, maxWorkerCeiling())
}

function targetDocumentCount(): number {
  return configuredInt('AXIOM_CRAWL_TARGET_DOCS', 'target_documents', DEFAULT_TARGET_DOCUMENTS, 1, 64)
}

function maxWavesLimit(): number {
  return configuredInt(undefined, 'max_waves', DEFAULT_MAX_WAVES, 1, 64)
}

function defaultWaveCount(): number {
  return configuredInt('AXIOM_CRAWL_WAVES', 'default_waves', 3, 1, maxWavesLimit())
}

function earlyStopScore(): number {
  return configuredFloat('AXIOM_CRAWL_EARLY_STOP_SCORE', 'early_stop_score', DEFAULT_EARLY_STOP_SCORE, 1, 1000)
}

function sourceUrlCap(): number {
  return configuredInt('AXIOM_MAX_SEARCH_SOURCES', 'max_search_sources', DEFAULT_SOURCE_URL_CAP, 1, 512)
}

export function toAxiomCrawlPlan(input: unknown): AxiomCrawlPlan {
  if (isPlanLike(input)) {
    return normalizePlan(input)
  }
  const query = normalizeQueryText(extractText(input).trim())
  const requestedWorkers = extractRequestedWorkers(input) ?? defaultWorkerCount()
  const depth = boundedInt(extractRequestedDepth(input), defaultWaveCount(), 1, maxWavesLimit())
  const seedDomains = uniqueDomains([
    ...extractDomains(query),
    ...pickSeedDomains(query),
  ])
  return {
    watermark: AXIOM_SWARM_WATERMARK,
    intent: inferIntent(query),
    query,
    worker_count: requestedWorkers,
    requested_worker_count: requestedWorkers,
    target_documents: targetDocumentCount(),
    max_waves: depth,
    depth,
    early_stop_score: earlyStopScore(),
    seed_domains: seedDomains,
    source_urls: sourceUrlsForDomains(query, seedDomains),
    constraints: {
      one_worker_per_site: true,
      no_duplicate_site_fetch: true,
      no_external_search_engine: true,
      default_worker_ceiling: maxWorkerCeiling(),
      absolute_worker_limit: absoluteWorkerLimit(),
      lower_compute: [
        'dedupe_urls',
        'dedupe_sites',
        'quality_early_stop',
        'link_expansion_after_wave',
      ],
    },
    origin: originContext(input),
  }
}

export function extractText(input: unknown): string {
  const parts: string[] = []
  collectText(input, parts, 0)
  return parts.filter(Boolean).join('\n')
}

export function extractDomains(text: string): string[] {
  const domains: string[] = []
  for (const match of text.matchAll(DOMAIN_RE)) {
    const domain = normalizeDomain(match.groups?.domain ?? '')
    if (domain) domains.push(domain)
  }
  return uniqueDomains(domains)
}

export function inferIntent(text: string): AxiomCrawlPlan['intent'] {
  const lowered = text.toLowerCase()
  if (lowered.startsWith('fetch ') || lowered.startsWith('open ') || lowered.startsWith('load ')) {
    return 'fetch'
  }
  if (lowered.startsWith('learn ') || lowered.startsWith('crawl ') || lowered.startsWith('index ')) {
    return 'learn'
  }
  return 'web_search'
}

export function pickSeedDomains(text: string): string[] {
  const lowered = text.toLowerCase()
  const matchedDomains: string[] = []
  const fallbackDomains: string[] = []
  for (const profile of sourceConfig().profiles ?? []) {
    const terms = Array.isArray(profile.terms) ? profile.terms.map(term => String(term).toLowerCase()).filter(Boolean) : []
    if (profile.always) {
      fallbackDomains.push(...(Array.isArray(profile.domains) ? profile.domains : []))
    } else if (terms.some(term => lowered.includes(term))) {
      matchedDomains.push(...(Array.isArray(profile.domains) ? profile.domains : []))
    }
  }
  return uniqueDomains([...matchedDomains, ...fallbackDomains])
}

function normalizePlan(input: Record<string, unknown>): AxiomCrawlPlan {
  const query = normalizeQueryText(String(input.query ?? extractText(input)).trim())
  const requestedWorkers = extractRequestedWorkers(input) ?? defaultWorkerCount()
  const depth = boundedInt(input.max_waves ?? input.depth ?? extractRequestedDepth(input), defaultWaveCount(), 1, maxWavesLimit())
  const seedDomains = uniqueDomains([
    ...normalizeDomainList(input.seed_domains),
    ...extractDomains(query),
    ...pickSeedDomains(query),
  ])
  return {
    watermark: AXIOM_SWARM_WATERMARK,
    intent: isIntent(input.intent) ? input.intent : inferIntent(query),
    query,
    worker_count: requestedWorkers,
    requested_worker_count: requestedWorkers,
    target_documents: boundedInt(input.target_documents, targetDocumentCount(), 1, 64),
    max_waves: depth,
    depth,
    early_stop_score: boundedFloat(input.early_stop_score, earlyStopScore(), 1, 1000),
    seed_domains: seedDomains,
    source_urls: uniqueSourceUrls([
      ...normalizeSourceUrls(input.source_urls),
      ...sourceUrlsForDomains(query, seedDomains),
    ]),
    constraints: {
      one_worker_per_site: true,
      no_duplicate_site_fetch: true,
      no_external_search_engine: true,
      default_worker_ceiling: maxWorkerCeiling(),
      absolute_worker_limit: absoluteWorkerLimit(),
      lower_compute: [
        'dedupe_urls',
        'dedupe_sites',
        'quality_early_stop',
        'link_expansion_after_wave',
      ],
    },
    origin: originContext(input),
  }
}

function collectText(value: unknown, parts: string[], depth: number): void {
  if (depth > 5 || value === null || value === undefined) return
  if (typeof value === 'string') {
    parts.push(value)
    return
  }
  if (Array.isArray(value)) {
    for (const item of value.slice(0, 32)) collectText(item, parts, depth + 1)
    return
  }
  if (typeof value !== 'object') return
  const record = value as Record<string, unknown>
  for (const key of ['query', 'content', 'text', 'prompt', 'payload', 'value', 'command', 'title', 'description']) {
    collectText(record[key], parts, depth + 1)
  }
  for (const key of ['task', 'message', 'messages', 'queue', 'hints', 'input']) {
    collectText(record[key], parts, depth + 1)
  }
}

function normalizeQueryText(text: string): string {
  const lines = text.split(/\r?\n/).map(line => line.replace(/\s+/g, ' ').trim()).filter(Boolean)
  if (lines.length === 0) return ''
  for (const line of lines) {
    const parsed = parseSwarmCommand(line)
    if (parsed) return parsed.query
  }
  for (const line of lines) {
    const lowered = line.toLowerCase()
    for (const prefix of [
      'search the whole internet for ',
      'search the internet for ',
      'search the web for ',
      'search for ',
    ]) {
      if (lowered.startsWith(prefix)) return line.slice(prefix.length).trim()
    }
  }
  return lines[0] ?? ''
}

function parseSwarmCommand(text: string): { query: string; workers?: number; depth?: number } | undefined {
  let segments = text.split('|').map(segment => segment.trim()).filter(Boolean)
  if (segments[0]?.toLowerCase() === 'search') segments = segments.slice(1)
  if (segments.length === 0) return undefined
  const swarmMatch = SWARM_HEAD_RE.exec(segments[0] ?? '')
  if (!swarmMatch) return undefined
  const querySegments: string[] = []
  let depth: number | undefined
  for (const segment of segments.slice(1)) {
    const depthMatch = DEPTH_SEGMENT_RE.exec(segment)
    if (depthMatch?.groups?.depth) {
      depth = boundedInt(depthMatch.groups.depth, defaultWaveCount(), 1, maxWavesLimit())
      continue
    }
    querySegments.push(segment)
  }
  return {
    query: querySegments.join(' | ').trim(),
    workers: coerceInt(swarmMatch.groups?.workers),
    depth,
  }
}

function extractRequestedWorkers(input: unknown): number | undefined {
  if (input && typeof input === 'object') {
    const record = input as Record<string, unknown>
    for (const key of ['requested_worker_count', 'requested_workers', 'worker_count', 'workers', 'parallelism', 'concurrency']) {
      const value = coerceInt(record[key])
      if (value !== undefined) return value
    }
    for (const key of ['constraints', 'hints', 'config', 'options', 'crawl']) {
      const nested = extractRequestedWorkers(record[key])
      if (nested !== undefined) return nested
    }
  }
  const text = extractText(input)
  const parsed = parseSwarmCommand(text)
  if (parsed?.workers !== undefined) return parsed.workers
  const match = WORKER_RE.exec(text)
  return coerceInt(match?.groups?.workers)
}

function extractRequestedDepth(input: unknown): number | undefined {
  if (input && typeof input === 'object') {
    const record = input as Record<string, unknown>
    for (const key of ['depth', 'crawl_depth', 'max_waves', 'waves']) {
      const value = coerceInt(record[key])
      if (value !== undefined) return boundedInt(value, defaultWaveCount(), 1, maxWavesLimit())
    }
    for (const key of ['constraints', 'hints', 'config', 'options', 'crawl']) {
      const nested = extractRequestedDepth(record[key])
      if (nested !== undefined) return nested
    }
  }
  const text = extractText(input)
  const parsed = parseSwarmCommand(text)
  if (parsed?.depth !== undefined) return parsed.depth
  const match = DEPTH_RE.exec(text)
  if (!match?.groups?.depth) return undefined
  return boundedInt(match.groups.depth, defaultWaveCount(), 1, maxWavesLimit())
}

function sourceUrlsForDomains(query: string, domains: string[]): AxiomSourceUrl[] {
  const encoded = encodeURIComponent(query)
  const sources: AxiomSourceUrl[] = []
  const templates = sourceConfig().search_templates ?? {}
  for (const domain of domains) {
    const searchUrl = templates[domain] ?? `https://${domain}/search?q={query}`
    sources.push({
      url: searchUrl.replace('{query}', encoded),
      domain,
      reason: 'swarm_bridge_seed',
      seeded: true,
      cached: false,
    })
    sources.push({
      url: `https://${domain}/`,
      domain,
      reason: 'swarm_bridge_seed_root',
      seeded: true,
      cached: false,
    })
  }
  return sources
}

function originContext(input: unknown): AxiomCrawlPlan['origin'] {
  if (!input || typeof input !== 'object') {
    return { kind: 'text', task_type: '', agent_name: '', team_name: '', message_count: extractText(input).trim() ? 1 : 0 }
  }
  const record = input as Record<string, unknown>
  const task = isRecord(record.task) ? record.task : {}
  const identity = isRecord(record.identity) ? record.identity : {}
  return {
    kind: String(record.kind ?? record.type ?? 'generic_swarm_talk'),
    task_type: String(task.type ?? record.task_type ?? ''),
    agent_name: String(identity.agentName ?? record.agent_name ?? ''),
    team_name: String(identity.teamName ?? record.team_name ?? ''),
    message_count: Array.isArray(record.messages) ? record.messages.length : 0,
  }
}

function normalizeDomainList(input: unknown): string[] {
  if (typeof input === 'string') return uniqueDomains(input.split(/[\s,;]+/))
  if (Array.isArray(input)) return uniqueDomains(input.map(item => String(item)))
  return []
}

function normalizeSourceUrls(input: unknown): AxiomSourceUrl[] {
  if (!Array.isArray(input)) return []
  return input.flatMap(item => {
    if (!isRecord(item)) return []
    const url = String(item.url ?? '').trim()
    const domain = normalizeDomain(String(item.domain ?? url))
    if (!url || !domain) return []
    return [{
      url,
      domain,
      reason: String(item.reason ?? 'swarm_bridge_source'),
      seeded: Boolean(item.seeded ?? true),
      cached: Boolean(item.cached ?? false),
    }]
  })
}

function normalizeDomain(raw: string): string {
  const trimmed = raw.trim().toLowerCase()
  if (!trimmed) return ''
  try {
    const withScheme = trimmed.includes('://') ? trimmed : `https://${trimmed}`
    const host = new URL(withScheme).hostname.replace(/\.$/, '')
    return host.includes('.') ? host : ''
  } catch {
    const bare = trimmed.split('/')[0]?.replace(/\.$/, '') ?? ''
    return bare.includes('.') && !/\s/.test(bare) ? bare : ''
  }
}

function uniqueDomains(domains: string[]): string[] {
  const seen = new Set<string>()
  const unique: string[] = []
  for (const raw of domains) {
    const domain = normalizeDomain(raw)
    if (!domain || seen.has(domain)) continue
    seen.add(domain)
    unique.push(domain)
  }
  return unique
}

function uniqueSourceUrls(sources: AxiomSourceUrl[]): AxiomSourceUrl[] {
  const seen = new Set<string>()
  const unique: AxiomSourceUrl[] = []
  for (const source of sources) {
    if (!source.url || seen.has(source.url)) continue
    seen.add(source.url)
    unique.push(source)
  }
  return unique.slice(0, sourceUrlCap())
}

function isPlanLike(input: unknown): input is Record<string, unknown> {
  return isRecord(input) && (input.watermark === AXIOM_SWARM_WATERMARK || 'seed_domains' in input || 'source_urls' in input)
}

function isRecord(input: unknown): input is Record<string, unknown> {
  return Boolean(input) && typeof input === 'object' && !Array.isArray(input)
}

function isIntent(input: unknown): input is AxiomCrawlPlan['intent'] {
  return input === 'web_search' || input === 'learn' || input === 'fetch'
}

function coerceInt(input: unknown): number | undefined {
  if (input === null || input === undefined || input === '') return undefined
  const value = Number(input)
  return Number.isInteger(value) ? value : undefined
}

function boundedInt(input: unknown, fallback: number, low: number, high: number): number {
  const value = coerceInt(input)
  if (value === undefined) return fallback
  return Math.max(low, Math.min(high, value))
}

function boundedFloat(input: unknown, fallback: number, low: number, high: number): number {
  const value = Number(input)
  if (!Number.isFinite(value)) return fallback
  return Math.max(low, Math.min(high, value))
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = []
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  return Buffer.concat(chunks).toString('utf8')
}

async function main(): Promise<void> {
  const raw = (await readStdin()).trim()
  const input = raw ? parseInput(raw) : ''
  process.stdout.write(`${JSON.stringify(toAxiomCrawlPlan(input), null, 2)}\n`)
}

function parseInput(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown
  } catch {
    return raw
  }
}

const invokedPath = process.argv[1] ? pathToFileURL(process.argv[1]).href : ''
if (import.meta.url === invokedPath) {
  main().catch(error => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    process.exitCode = 1
  })
}
