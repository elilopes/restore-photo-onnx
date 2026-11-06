import path from 'node:path'

const env = process.env

function int(name: string, fallback: number): number {
  const raw = env[name]?.trim()
  if (!raw) return fallback
  const n = Number(raw)
  if (!Number.isFinite(n)) throw new Error(`variável ${name} inválida: ${raw}`)
  return n
}

const retentionHours = int('RETENTION_HOURS', 24)

export const config = {
  port: int('PORT', 3001),
  host: env.HOST?.trim() || '0.0.0.0',
  redisUrl: env.REDIS_URL?.trim() || 'redis://localhost:6379',
  workerUrl: (env.WORKER_URL?.trim() || 'http://localhost:8000').replace(/\/+$/, ''),
  storageDir: path.resolve(env.STORAGE_DIR?.trim() || '../storage'),
  corsOrigin: env.CORS_ORIGIN?.trim() || 'http://localhost:5173',
  maxUploadBytes: int('MAX_UPLOAD_MB', 25) * 1024 * 1024,
  queueName: 'restore-photo',
  queueConcurrency: int('QUEUE_CONCURRENCY', 1),
  workerTimeoutMs: int('WORKER_TIMEOUT_MS', 10 * 60_000),
  progressPollMs: int('PROGRESS_POLL_MS', 1000),
  retentionHours,
  retentionSeconds: Math.round(retentionHours * 3600),
} as const
