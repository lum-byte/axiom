import { WORKFLOW_TOOL_NAME } from './constants.js'

function asArray(value) {
  if (Array.isArray(value)) return value.map(String).filter(Boolean)
  if (typeof value === 'string' && value.trim()) return value.split(/\r?\n|,/).map(item => item.trim()).filter(Boolean)
  return []
}

function inferStages(goal, files) {
  const stages = [
    { id: 'context', title: 'Read local contracts and docs', commands: [] },
    { id: 'implementation', title: 'Apply scoped implementation changes', commands: [] },
    { id: 'verification', title: 'Run focused verification', commands: [] },
  ]
  const lower = `${goal} ${files.join(' ')}`.toLowerCase()
  if (lower.includes('tools')) stages[2].commands.push('cd tools && npm run typecheck && npm run build')
  if (lower.includes('preparser') || lower.includes('.go')) stages[2].commands.push('go test ./preparser/...')
  if (lower.includes('offline') || lower.includes('.cu')) stages[2].commands.push('sh ./run_cuda_tests.sh')
  if (lower.includes('daemon') || lower.includes('alpine_strip') || lower.includes('.c')) stages[2].commands.push('sh ./run_c_tests.sh')
  if (lower.includes('axiom_tui') || lower.includes('.rs')) stages[2].commands.push('cd axiom_tui && cargo test')
  if (lower.includes('python') || lower.includes('tag/') || lower.includes('signal_kernel')) stages[2].commands.push('python -m pytest tests -q')
  if (stages[2].commands.length === 0) stages[2].commands.push('python -m pytest tests -q')
  return stages
}

export class WorkflowTool {
  constructor(options = {}) {
    this.name = WORKFLOW_TOOL_NAME
    this.options = options
  }

  async call(input = {}) {
    const goal = String(input.goal || input.task || input.prompt || 'AXIOM workflow')
    const files = asArray(input.files || input.changedFiles || input.changed_files)
    const constraints = asArray(input.constraints || input.rules)
    const stages = inferStages(goal, files)
    return {
      status: 'ready',
      workflow: WORKFLOW_TOOL_NAME,
      goal,
      files,
      constraints,
      stages,
      acceptance: [
        'contract-visible payloads stay canonical',
        'generated artifacts stay outside durable store files unless owned by the component',
        'focused tests pass before integration tests',
      ],
      created_at: new Date().toISOString(),
    }
  }
}

export default WorkflowTool
