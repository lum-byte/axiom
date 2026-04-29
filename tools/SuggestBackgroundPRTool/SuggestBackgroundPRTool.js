const DEFAULT_TESTS = [
  'python -m pytest tests -q',
  'npm run typecheck',
  'cargo test',
]

function asArray(value) {
  if (Array.isArray(value)) return value.map(String).filter(Boolean)
  if (typeof value === 'string' && value.trim()) return value.split(/\r?\n|,/).map(item => item.trim()).filter(Boolean)
  return []
}

function riskForFile(path) {
  const lower = String(path).toLowerCase()
  if (lower.includes('contracts') || lower.includes('crawler_bus') || lower.endsWith('.c') || lower.endsWith('.cu')) return 3
  if (lower.includes('interface') || lower.includes('daemon') || lower.includes('offline') || lower.endsWith('.rs') || lower.endsWith('.go')) return 2
  if (lower.includes('test') || lower.includes('docs') || lower.endsWith('.md')) return 1
  return 2
}

function buildRisk(changedFiles) {
  const score = changedFiles.reduce((max, file) => Math.max(max, riskForFile(file)), 0)
  if (score >= 3) return { level: 'high', reason: 'Contract, bus, native, or GPU surfaces changed.' }
  if (score === 2) return { level: 'medium', reason: 'Runtime implementation changed and needs focused regression.' }
  if (score === 1) return { level: 'low', reason: 'Test/documentation-weighted change set.' }
  return { level: 'unknown', reason: 'No changed files were supplied.' }
}

function inferTests(changedFiles) {
  const tests = new Set()
  for (const file of changedFiles) {
    const lower = file.toLowerCase()
    if (lower.endsWith('.py')) tests.add('python -m pytest tests -q')
    if (lower.endsWith('.ts') || lower.endsWith('.tsx') || lower.includes('tools/')) {
      tests.add('cd tools && npm run typecheck && npm run build')
    }
    if (lower.endsWith('.go') || lower.includes('preparser/')) tests.add('go test ./preparser/...')
    if (lower.endsWith('.c') || lower.includes('daemons/') || lower.includes('alpine_strip/')) tests.add('sh ./run_c_tests.sh')
    if (lower.endsWith('.cu') || lower.includes('offline/')) tests.add('sh ./run_cuda_tests.sh')
    if (lower.endsWith('.rs') || lower.includes('axiom_tui/')) tests.add('cd axiom_tui && cargo test')
  }
  if (tests.size === 0) DEFAULT_TESTS.forEach(test => tests.add(test))
  return [...tests]
}

function summarizeFiles(changedFiles) {
  const groups = new Map()
  for (const file of changedFiles) {
    const group = String(file).split(/[\\/]/)[0] || 'root'
    groups.set(group, (groups.get(group) || 0) + 1)
  }
  return [...groups.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([group, count]) => ({ group, count }))
}

export class SuggestBackgroundPRTool {
  constructor(options = {}) {
    this.name = 'suggest_background_pr'
    this.options = options
  }

  async call(input = {}) {
    const changedFiles = asArray(input.changedFiles || input.changed_files || input.files)
    const goals = asArray(input.goals || input.goal || input.summary)
    const findings = asArray(input.findings || input.risks || input.notes)
    const branch = String(input.branch || input.branchName || 'codex/axiom-runtime-hardening')
    const base = String(input.base || input.baseBranch || 'main')
    const risk = buildRisk(changedFiles)
    const tests = inferTests(changedFiles)
    const titleSubject = goals[0] || (changedFiles[0] ? `Update ${changedFiles[0].split(/[\\/]/)[0]}` : 'AXIOM runtime update')
    const title = `[AXIOM] ${titleSubject}`.slice(0, 120)
    const bodyLines = [
      '## Summary',
      ...((goals.length ? goals : ['Harden AXIOM runtime behavior behind the existing contract seam.']).map(item => `- ${item}`)),
      '',
      '## Risk',
      `- ${risk.level}: ${risk.reason}`,
      ...findings.map(item => `- ${item}`),
      '',
      '## Test Plan',
      ...tests.map(item => `- \`${item}\``),
    ]
    return {
      status: 'ready',
      branch,
      base,
      title,
      body: bodyLines.join('\n'),
      risk,
      tests,
      file_summary: summarizeFiles(changedFiles),
      changed_files: changedFiles,
    }
  }
}

export default SuggestBackgroundPRTool
