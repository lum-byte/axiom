import { createHash, randomUUID } from 'node:crypto'
import { createRequire } from 'node:module'
import { performance } from 'node:perf_hooks'
import type {
  AxiomRegisteredTool,
  AxiomToolExecution,
  AxiomToolResult,
} from './types.js'
import { AnthropicProviderAdapter } from './anthropic.js'
import { normalizeAlpineStripRequest } from '../AlpineStripTool/AlpineStripTool.js'

export type RunnerRequest = {
  tool: string
  input?: Record<string, unknown>
  mode?: 'selective' | 'manual' | 'diagnostic' | 'snapshot'
  run_id?: string
}

const nodeRequire = createRequire(import.meta.url)

export class AxiomToolRunner {
  private readonly registry: Map<string, AxiomRegisteredTool>

  constructor(tools: AxiomRegisteredTool[]) {
    this.registry = new Map(tools.map(tool => [tool.name, tool]))
  }

  list(): AxiomRegisteredTool[] {
    return [...this.registry.values()].sort((a, b) => a.name.localeCompare(b.name))
  }

  health(): Array<AxiomRegisteredTool & { dependencyStatus: Record<string, boolean> }> {
    return this.list().map(tool => ({
      ...tool,
      dependencyStatus: Object.fromEntries(tool.dependencies.map(dep => [dep, dependencyAvailable(dep)])),
    }))
  }

  async call(request: RunnerRequest): Promise<AxiomToolExecution> {
    const started = performance.now()
    const invocationId = randomUUID()
    const tool = this.registry.get(request.tool)
    if (!tool) {
      return this.error(request.tool, invocationId, started, 'ToolNotRegistered', `Unknown tool ${request.tool}`)
    }
    if (tool.status !== 'ready') {
      return this.error(tool.name, invocationId, started, 'ToolUnavailable', `Tool ${tool.name} is ${tool.status}`)
    }
    try {
      const data = await this.dispatch(tool, request.input ?? {})
      const resultBlock = this.resultBlock(JSON.stringify(data), invocationId)
      return {
        ok: true,
        toolName: tool.name,
        invocationId,
        durationMs: performance.now() - started,
        data,
        resultBlock,
      }
    } catch (error) {
      return this.error(
        tool.name,
        invocationId,
        started,
        error instanceof Error ? error.name : 'ToolExecutionError',
        error instanceof Error ? error.message : String(error),
      )
    }
  }

  private async dispatch(tool: AxiomRegisteredTool, input: Record<string, unknown>): Promise<unknown> {
    if (tool.name === 'AlpineStripTool') {
      const request = normalizeAlpineStripRequest(input)
      return {
        adapter: 'AlpineStripTool',
        status: 'planned',
        native: 'alpine_strip/tool_strip_accelerator.c',
        request,
        queue_line: buildQueueLine(request),
      }
    }
    if (tool.name === 'WebFetchTool') {
      return {
        adapter: 'WebFetchTool',
        status: 'planned',
        url: input.url,
        note: 'Runtime bridge will call the existing WebFetchTool implementation when the full tool app state is present.',
      }
    }
    if (tool.name === 'WebSearchTool') {
      return {
        adapter: 'WebSearchTool',
        status: 'registered',
        query: input.query,
        note: 'Available for explicit diagnostic tool calls; AXIOM TAG search remains primary.',
      }
    }
    if (tool.name === 'SuggestBackgroundPRTool') {
      return suggestBackgroundPR(input)
    }
    if (tool.name === 'TungstenTool') {
      return tungstenSnapshot(input)
    }
    if (tool.name === 'VerifyPlanExecutionTool') {
      return verifyPlanExecution(input)
    }
    if (tool.name === 'WorkflowTool') {
      return buildWorkflow(input)
    }
    return {
      adapter: tool.name,
      status: 'executed_metadata_adapter',
      inputHash: hashJSON(input),
      permissionClass: tool.permissionClass,
      adapterKind: tool.adapterKind,
    }
  }

  private resultBlock(content: string, invocationId: string): AxiomToolResult {
    return AnthropicProviderAdapter.toToolResultBlock(content, invocationId)
  }

  private error(toolName: string, invocationId: string, started: number, type: string, message: string): AxiomToolExecution {
    return {
      ok: false,
      toolName,
      invocationId,
      durationMs: performance.now() - started,
      error: { type, message },
      resultBlock: AnthropicProviderAdapter.toToolResultBlock(message, invocationId, true),
    }
  }
}

function buildQueueLine(request: ReturnType<typeof normalizeAlpineStripRequest>): string | undefined {
  if (!request.input_path || !request.output_path) {
    return undefined
  }
  return JSON.stringify({
    url: request.url ?? '',
    slot_idx: request.slot_idx ?? 0,
    input_path: request.input_path,
    output_path: request.output_path,
  })
}

function stringArray(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(item => String(item)).filter(Boolean)
  if (typeof value === 'string' && value.trim()) {
    return value.split(/\r?\n|,/).map(item => item.trim()).filter(Boolean)
  }
  return []
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim() ? value.trim() : fallback
}

function suggestBackgroundPR(input: Record<string, unknown>): Record<string, unknown> {
  const changedFiles = stringArray(input.changedFiles ?? input.changed_files ?? input.files)
  const goals = stringArray(input.goals ?? input.goal ?? input.summary)
  const tests = inferTests(changedFiles)
  const risk = riskFromFiles(changedFiles)
  const titleSeed = goals[0] ?? (changedFiles[0] ? `Update ${changedFiles[0].split(/[\\/]/)[0]}` : 'AXIOM runtime update')
  return {
    adapter: 'SuggestBackgroundPRTool',
    status: 'ready',
    branch: stringValue(input.branch ?? input.branchName, 'codex/axiom-runtime-hardening'),
    base: stringValue(input.base ?? input.baseBranch, 'main'),
    title: `[AXIOM] ${titleSeed}`.slice(0, 120),
    summary: goals.length > 0 ? goals : ['Harden AXIOM runtime behavior behind the existing contract seam.'],
    risk,
    tests,
    changedFiles,
    fileSummary: summarizeFileGroups(changedFiles),
  }
}

function tungstenSnapshot(input: Record<string, unknown>): Record<string, unknown> {
  const memory = process.memoryUsage()
  const load = typeof process.cpuUsage === 'function' ? process.cpuUsage() : { user: 0, system: 0 }
  return {
    adapter: 'TungstenTool',
    status: 'ready',
    platform: process.platform,
    arch: process.arch,
    pid: process.pid,
    node: process.version,
    cwd: stringValue(input.cwd ?? input.root, process.cwd()),
    processUptimeSeconds: Math.round(process.uptime()),
    memory,
    cpuUsageMicros: load,
    checkedAt: new Date().toISOString(),
  }
}

function verifyPlanExecution(input: Record<string, unknown>): Record<string, unknown> {
  const steps = normalizePlan(input.plan ?? input.steps ?? input.checklist)
  const tests = stringArray(input.tests ?? input.testOutput ?? input.test_output)
  const changedFiles = stringArray(input.changedFiles ?? input.changed_files ?? input.files)
  const completed = steps.filter(step => step.status === 'completed')
  const blocked = steps.filter(step => step.status === 'blocked')
  const failedTests = tests.filter(test => /fail|error|nonzero|timeout/i.test(test))
  const testScore = tests.length === 0 ? 0 : Math.max(0, 1 - failedTests.length / tests.length)
  const planScore = steps.length === 0 ? 0 : completed.length / steps.length
  const fileScore = changedFiles.length > 0 ? 1 : 0
  const score = Number(((planScore * 0.55) + (testScore * 0.35) + (fileScore * 0.10)).toFixed(4))
  const missingEvidence = [
    ...(steps.length === 0 ? ['plan'] : []),
    ...(completed.length !== steps.length ? ['completed_steps'] : []),
    ...(tests.length === 0 ? ['tests'] : []),
    ...(changedFiles.length === 0 ? ['changed_files'] : []),
  ]
  return {
    adapter: 'VerifyPlanExecutionTool',
    status: blocked.length > 0 || failedTests.length > 0 ? 'failed' : missingEvidence.length === 0 && score >= 0.95 ? 'verified' : 'incomplete',
    score,
    completedSteps: completed.length,
    totalSteps: steps.length,
    pendingSteps: steps.filter(step => step.status !== 'completed'),
    blockedSteps: blocked,
    failedTests,
    changedFiles,
    missingEvidence,
    checkedAt: new Date().toISOString(),
  }
}

function buildWorkflow(input: Record<string, unknown>): Record<string, unknown> {
  const goal = stringValue(input.goal ?? input.task ?? input.prompt, 'AXIOM workflow')
  const files = stringArray(input.files ?? input.changedFiles ?? input.changed_files)
  const lower = `${goal} ${files.join(' ')}`.toLowerCase()
  const commands = new Set<string>()
  if (lower.includes('tools')) commands.add('cd tools && npm run typecheck && npm run build')
  if (lower.includes('preparser') || lower.includes('.go')) commands.add('go test ./preparser/...')
  if (lower.includes('offline') || lower.includes('.cu')) commands.add('sh ./run_cuda_tests.sh')
  if (lower.includes('daemon') || lower.includes('alpine_strip') || lower.includes('.c')) commands.add('sh ./run_c_tests.sh')
  if (lower.includes('axiom_tui') || lower.includes('.rs')) commands.add('cd axiom_tui && cargo test')
  if (lower.includes('python') || lower.includes('tag/') || lower.includes('signal_kernel')) commands.add('python -m pytest tests -q')
  if (commands.size === 0) commands.add('python -m pytest tests -q')
  return {
    adapter: 'WorkflowTool',
    status: 'ready',
    goal,
    files,
    stages: [
      { id: 'context', title: 'Read local contracts and docs', commands: [] },
      { id: 'implementation', title: 'Apply scoped implementation changes', commands: [] },
      { id: 'verification', title: 'Run focused verification', commands: [...commands] },
    ],
    acceptance: [
      'contract-visible payloads stay canonical',
      'generated artifacts stay outside durable store files unless owned by the component',
      'focused tests pass before integration tests',
    ],
  }
}

function inferTests(changedFiles: string[]): string[] {
  const tests = new Set<string>()
  for (const file of changedFiles) {
    const lower = file.toLowerCase()
    if (lower.endsWith('.py')) tests.add('python -m pytest tests -q')
    if (lower.endsWith('.ts') || lower.endsWith('.tsx') || lower.includes('tools/')) tests.add('cd tools && npm run typecheck && npm run build')
    if (lower.endsWith('.go') || lower.includes('preparser/')) tests.add('go test ./preparser/...')
    if (lower.endsWith('.c') || lower.includes('daemons/') || lower.includes('alpine_strip/')) tests.add('sh ./run_c_tests.sh')
    if (lower.endsWith('.cu') || lower.includes('offline/')) tests.add('sh ./run_cuda_tests.sh')
    if (lower.endsWith('.rs') || lower.includes('axiom_tui/')) tests.add('cd axiom_tui && cargo test')
  }
  if (tests.size === 0) tests.add('python -m pytest tests -q')
  return [...tests]
}

function riskFromFiles(changedFiles: string[]): Record<string, string> {
  const highRisk = changedFiles.some(file => /contracts|crawler_bus|\.c$|\.cu$/i.test(file))
  const mediumRisk = changedFiles.some(file => /interface|daemon|offline|\.go$|\.rs$/i.test(file))
  if (highRisk) return { level: 'high', reason: 'Contract, bus, native, or GPU surfaces changed.' }
  if (mediumRisk) return { level: 'medium', reason: 'Runtime implementation changed and needs focused regression.' }
  if (changedFiles.length > 0) return { level: 'low', reason: 'Change set avoids core runtime seams.' }
  return { level: 'unknown', reason: 'No changed files were supplied.' }
}

function summarizeFileGroups(changedFiles: string[]): Array<{ group: string; count: number }> {
  const groups = new Map<string, number>()
  for (const file of changedFiles) {
    const group = file.split(/[\\/]/)[0] || 'root'
    groups.set(group, (groups.get(group) ?? 0) + 1)
  }
  return [...groups.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([group, count]) => ({ group, count }))
}

type NormalizedStep = { id: string; text: string; status: 'completed' | 'in_progress' | 'blocked' | 'pending' }

function normalizePlan(value: unknown): NormalizedStep[] {
  const raw = Array.isArray(value) ? value : stringArray(value)
  return raw.map((item, index): NormalizedStep => {
    if (typeof item === 'string') {
      return { id: `step-${index + 1}`, text: item, status: 'pending' }
    }
    const record = item as Record<string, unknown>
    return {
      id: stringValue(record.id ?? record.step_id, `step-${index + 1}`),
      text: stringValue(record.text ?? record.step ?? record.title, ''),
      status: normalizeStepStatus(record.status),
    }
  })
}

function normalizeStepStatus(value: unknown): NormalizedStep['status'] {
  const text = String(value ?? '').toLowerCase()
  if (['done', 'complete', 'completed', 'pass', 'passed'].includes(text)) return 'completed'
  if (['doing', 'active', 'in_progress', 'running'].includes(text)) return 'in_progress'
  if (['blocked', 'fail', 'failed', 'error'].includes(text)) return 'blocked'
  return 'pending'
}

export function hashJSON(value: unknown): string {
  return createHash('sha256').update(stableJSONStringify(value)).digest('hex')
}

export function stableJSONStringify(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableJSONStringify).join(',')}]`
  }
  const record = value as Record<string, unknown>
  return `{${Object.keys(record).sort().map(key => `${JSON.stringify(key)}:${stableJSONStringify(record[key])}`).join(',')}}`
}

function dependencyAvailable(dep: string): boolean {
  if (dep === 'bun:bundle') {
    return typeof process !== 'undefined'
  }
  try {
    nodeRequire.resolve(dep)
    return true
  } catch {
    return false
  }
}
