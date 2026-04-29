import { createInterface } from 'node:readline'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { discoverTools } from './registry.js'
import { AxiomToolRunner, hashJSON } from './runner.js'

const currentDir = dirname(fileURLToPath(import.meta.url))
const toolsRoot = process.env.AXIOM_TOOLS_ROOT ?? resolve(currentDir, '..', '..')
const runner = new AxiomToolRunner(discoverTools(toolsRoot))

type SidecarRequest =
  | { op: 'list' }
  | { op: 'health' }
  | { op: 'call'; tool: string; input?: Record<string, unknown>; run_id?: string; mode?: 'selective' | 'manual' | 'diagnostic' | 'snapshot' }

async function handle(raw: SidecarRequest): Promise<Record<string, unknown>> {
  if (raw.op === 'list') {
    return { ok: true, tools: runner.list() }
  }
  if (raw.op === 'health') {
    return { ok: true, tools: runner.health() }
  }
  if (raw.op === 'call') {
    const result = await runner.call(raw)
    return {
      ok: result.ok,
      result,
      input_hash: hashJSON(raw.input ?? {}),
    }
  }
  return { ok: false, error_type: 'UnknownOperation', error: `unknown op ${(raw as { op?: unknown }).op}` }
}

async function main(): Promise<void> {
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity })
  for await (const line of rl) {
    if (!line.trim()) continue
    try {
      const request = JSON.parse(line) as SidecarRequest
      const response = await handle(request)
      process.stdout.write(`${JSON.stringify(response)}\n`)
    } catch (error) {
      process.stdout.write(JSON.stringify({
        ok: false,
        error_type: error instanceof Error ? error.name : 'SidecarError',
        error: error instanceof Error ? error.message : String(error),
      }) + '\n')
    }
  }
}

main().catch(error => {
  process.stderr.write(`${error instanceof Error ? error.stack ?? error.message : String(error)}\n`)
  process.exitCode = 1
})
