function asArray(value) {
  if (Array.isArray(value)) return value.map(String).filter(Boolean)
  if (typeof value === 'string' && value.trim()) return value.split(/\r?\n/).map(item => item.trim()).filter(Boolean)
  return []
}

function normalizeStatus(value) {
  const text = String(value || '').toLowerCase()
  if (['done', 'complete', 'completed', 'pass', 'passed'].includes(text)) return 'completed'
  if (['doing', 'active', 'in_progress', 'running'].includes(text)) return 'in_progress'
  if (['blocked', 'fail', 'failed', 'error'].includes(text)) return 'blocked'
  return 'pending'
}

function normalizePlanItem(item, index) {
  if (typeof item === 'string') {
    return { id: `step-${index + 1}`, text: item, status: 'pending' }
  }
  const record = item && typeof item === 'object' ? item : {}
  return {
    id: String(record.id || record.step_id || `step-${index + 1}`),
    text: String(record.text || record.step || record.title || ''),
    status: normalizeStatus(record.status),
  }
}

function scoreTests(tests) {
  if (tests.length === 0) return { score: 0, status: 'missing', failed: [] }
  const failed = tests.filter(test => /fail|error|nonzero|timeout/i.test(test))
  return {
    score: failed.length === 0 ? 1 : Math.max(0, 1 - failed.length / tests.length),
    status: failed.length === 0 ? 'passed' : 'failed',
    failed,
  }
}

export class VerifyPlanExecutionTool {
  constructor(options = {}) {
    this.name = 'verify_plan_execution'
    this.options = options
  }

  async call(input = {}) {
    const planItems = asArray(input.plan || input.steps || input.checklist).map(normalizePlanItem)
    const changedFiles = asArray(input.changedFiles || input.changed_files || input.files)
    const tests = asArray(input.tests || input.testOutput || input.test_output)
    const completed = planItems.filter(item => item.status === 'completed')
    const blocked = planItems.filter(item => item.status === 'blocked')
    const pending = planItems.filter(item => item.status !== 'completed')
    const testScore = scoreTests(tests)
    const planScore = planItems.length === 0 ? 0 : completed.length / planItems.length
    const fileScore = changedFiles.length > 0 ? 1 : 0
    const score = Number(((planScore * 0.55) + (testScore.score * 0.35) + (fileScore * 0.10)).toFixed(4))
    const missing = []
    if (planItems.length === 0) missing.push('plan')
    if (pending.length > 0) missing.push('completed_steps')
    if (tests.length === 0) missing.push('tests')
    if (changedFiles.length === 0) missing.push('changed_files')
    const status = blocked.length > 0 || testScore.status === 'failed'
      ? 'failed'
      : missing.length === 0 && score >= 0.95
        ? 'verified'
        : 'incomplete'
    return {
      status,
      score,
      completed_steps: completed.length,
      total_steps: planItems.length,
      pending_steps: pending,
      blocked_steps: blocked,
      test_status: testScore.status,
      failed_tests: testScore.failed,
      changed_files: changedFiles,
      missing_evidence: missing,
      checked_at: new Date().toISOString(),
    }
  }
}

export default VerifyPlanExecutionTool
