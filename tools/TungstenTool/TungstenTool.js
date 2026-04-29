import os from 'node:os'
import fs from 'node:fs'
import path from 'node:path'

function readNumber(file) {
  try {
    const text = fs.readFileSync(file, 'utf8').trim()
    const value = Number(text)
    return Number.isFinite(value) ? value : null
  } catch {
    return null
  }
}

function diskSnapshot(root) {
  const resolved = path.resolve(root || process.cwd())
  const exists = fs.existsSync(resolved)
  let files = 0
  let bytes = 0
  if (exists) {
    const stack = [resolved]
    while (stack.length && files < 5000) {
      const current = stack.pop()
      let entries = []
      try {
        entries = fs.readdirSync(current, { withFileTypes: true })
      } catch {
        continue
      }
      for (const entry of entries) {
        const full = path.join(current, entry.name)
        if (entry.isDirectory() && !['.git', 'node_modules', 'dist', '__pycache__'].includes(entry.name)) {
          stack.push(full)
        } else if (entry.isFile()) {
          files += 1
          try {
            bytes += fs.statSync(full).size
          } catch {
            // Best effort diagnostics; unreadable files do not fail monitoring.
          }
        }
      }
    }
  }
  return { root: resolved, exists, sampled_files: files, sampled_bytes: bytes }
}

function thermalSnapshot() {
  const linuxZone = '/sys/class/thermal/thermal_zone0/temp'
  const millidegrees = readNumber(linuxZone)
  if (millidegrees === null) return { available: false }
  return { available: true, celsius: Math.round((millidegrees / 1000) * 10) / 10 }
}

export class TungstenTool {
  constructor(options = {}) {
    this.name = 'tungsten'
    this.options = options
  }

  async call(input = {}) {
    const sampleRoot = input.root || input.cwd || process.cwd()
    const load = os.loadavg()
    const cpus = os.cpus()
    const freeMem = os.freemem()
    const totalMem = os.totalmem()
    const memoryPressure = totalMem > 0 ? 1 - freeMem / totalMem : 0
    const snapshot = {
      status: 'ready',
      platform: process.platform,
      arch: process.arch,
      uptime_seconds: Math.round(os.uptime()),
      process_uptime_seconds: Math.round(process.uptime()),
      pid: process.pid,
      cpu: {
        cores: cpus.length,
        model: cpus[0] ? cpus[0].model : 'unknown',
        load_1m: load[0],
        load_5m: load[1],
        load_15m: load[2],
      },
      memory: {
        free_bytes: freeMem,
        total_bytes: totalMem,
        pressure: Number(memoryPressure.toFixed(4)),
      },
      thermal: thermalSnapshot(),
      disk_sample: diskSnapshot(sampleRoot),
      checked_at: new Date().toISOString(),
    }
    snapshot.health = memoryPressure > 0.92 ? 'memory_pressure' : 'nominal'
    return snapshot
  }
}

export default TungstenTool
