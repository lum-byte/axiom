import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import type { AxiomRegisteredTool } from './types.js'

export const AXIOM_TOOL_BRIDGE_VERSION = '0.1.0'

const INFRASTRUCTURE_DIRS = new Set([
  'axiom-sdk',
  'dist',
  'node_modules',
])

const NETWORK_TOOLS = new Set([
  'RemoteTriggerTool',
  'WebFetchTool',
  'WebSearchTool',
])

const WRITE_TEMP_TOOLS = new Set([
  'BriefTool',
  'FileReadTool',
])

const WRITE_REPO_TOOLS = new Set([
  'FileEditTool',
  'FileWriteTool',
  'NotebookEditTool',
])

const ORCHESTRATION_TOOLS = new Set([
  'AgentTool',
  'AskUserQuestionTool',
  'EnterPlanModeTool',
  'EnterWorktreeTool',
  'ExitPlanModeTool',
  'ExitWorktreeTool',
  'ScheduleCronTool',
  'SendMessageTool',
  'TaskCreateTool',
  'TaskGetTool',
  'TaskListTool',
  'TaskOutputTool',
  'TaskStopTool',
  'TaskUpdateTool',
  'TeamCreateTool',
  'TeamDeleteTool',
  'TodoWriteTool',
])

const EXTERNAL_DEPENDENCIES: Record<string, string[]> = {
  AlpineStripTool: [],
  AgentTool: ['zod/v4', 'react', 'bun:bundle'],
  AskUserQuestionTool: ['zod/v4', 'react', 'bun:bundle'],
  BashTool: ['zod/v4'],
  BriefTool: ['zod/v4', 'axios'],
  ConfigTool: ['zod/v4', 'bun:bundle'],
  FileEditTool: ['diff', 'zod/v4'],
  FileReadTool: ['zod/v4'],
  FileWriteTool: ['diff', 'zod/v4'],
  LSPTool: ['zod/v4'],
  MCPTool: ['zod/v4', 'react'],
  McpAuthTool: ['lodash-es/reject.js', 'zod/v4'],
  PowerShellTool: ['zod/v4'],
  SkillTool: ['lodash-es', 'zod/v4'],
  SyntheticOutputTool: ['ajv', 'zod/v4'],
  ToolSearchTool: ['lodash-es/memoize.js', 'zod/v4'],
  WebFetchTool: ['axios', 'lru-cache', 'zod/v4'],
  WebSearchTool: ['zod/v4'],
}

export function discoverTools(toolsRoot: string): AxiomRegisteredTool[] {
  const entries = readdirSync(toolsRoot, { withFileTypes: true })
  return entries
    .filter(entry => entry.isDirectory())
    .filter(entry => !INFRASTRUCTURE_DIRS.has(entry.name))
    .map(entry => buildToolRecord(entry.name, join(toolsRoot, entry.name)))
    .sort((a, b) => a.name.localeCompare(b.name))
}

export function buildToolRecord(name: string, sourcePath: string): AxiomRegisteredTool {
  const dependencies = EXTERNAL_DEPENDENCIES[name] ?? ['zod/v4']
  const hasFiles = directoryHasImplementation(sourcePath)
  const permissionClass = permissionClassForTool(name)
  const adapterKind = adapterKindForTool(name)
  const status = hasFiles ? 'ready' : 'disabled'
  return {
    name,
    adapterKind,
    permissionClass,
    sourcePath,
    status,
    dependencies,
  }
}

export function permissionClassForTool(name: string): AxiomRegisteredTool['permissionClass'] {
  if (name === 'AlpineStripTool') return 'read_only'
  if (NETWORK_TOOLS.has(name)) return 'network'
  if (WRITE_REPO_TOOLS.has(name)) return 'write_repo'
  if (WRITE_TEMP_TOOLS.has(name)) return 'write_temp'
  if (name === 'WorkflowTool' || name === 'SuggestBackgroundPRTool' || name === 'VerifyPlanExecutionTool') return 'orchestration'
  if (ORCHESTRATION_TOOLS.has(name)) return 'orchestration'
  return 'read_only'
}

export function adapterKindForTool(name: string): string {
  if (name === 'AlpineStripTool') return 'native_strip'
  if (name === 'TungstenTool') return 'runtime_monitor'
  if (name === 'VerifyPlanExecutionTool') return 'verification'
  if (name === 'WorkflowTool') return 'workflow'
  if (name === 'SuggestBackgroundPRTool') return 'orchestration'
  if (name.includes('Web')) return 'web'
  if (name.includes('File') || name.includes('Notebook') || name === 'BriefTool') return 'artifact'
  if (name.includes('Bash') || name.includes('PowerShell')) return 'shell_guard'
  if (name.includes('MCP') || name.includes('Mcp') || name.includes('Resource')) return 'connector'
  if (name.includes('Task') || name.includes('Team') || name.includes('Agent') || name.includes('Todo')) return 'orchestration'
  return 'utility'
}

function directoryHasImplementation(path: string): boolean {
  try {
    const files = readdirSync(path)
    return files.some(file => {
      const full = join(path, file)
      const stat = statSync(full)
      return stat.isFile() && /\.(ts|tsx|js)$/.test(file) && stat.size > 128
    })
  } catch {
    return false
  }
}
