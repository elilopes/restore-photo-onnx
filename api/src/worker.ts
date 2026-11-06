/**
 * Worker BullMQ: consome a fila e delega o processamento ao worker Python (FastAPI).
 * Roda como processo separado da API para escalar de forma independente
 * (QUEUE_CONCURRENCY jobs por processo; vários processos podem consumir a mesma fila).
 */
import { UnrecoverableError, Worker } from 'bullmq'
import { startCleanup } from './cleanup.js'
import { config } from './config.js'
import { createRedis } from './redis.js'
import type { RestoreJobData, RestoreJobResult } from './schemas.js'
import { LocalStorage } from './storage.js'
import { fetchProgress, processJob, WorkerHttpError } from './workerClient.js'

const log = (level: 'info' | 'warn' | 'error', msg: string, extra: Record<string, unknown> = {}) =>
  console.log(JSON.stringify({ level, time: new Date().toISOString(), msg, ...extra }))

const storage = new LocalStorage(config.storageDir)
await storage.init()
const stopCleanup = startCleanup(storage, (m) => log('info', m))

const worker = new Worker<RestoreJobData, RestoreJobResult>(
  config.queueName,
  async (job) => {
    // O Python expõe o progresso por etapa; repassamos para o job (o frontend lê via GET /api/jobs/:id).
    const poll = setInterval(() => {
      void fetchProgress(job.data.jobId)
        .then((p) => (p ? job.updateProgress({ percent: Math.round(p.progress * 100), step: p.step }) : undefined))
        .catch(() => undefined) // lock perdido/Redis instável: o próximo tick tenta de novo
    }, config.progressPollMs)

    try {
      await job.updateProgress({ percent: 1, step: 'start' })
      return await processJob(job.data)
    } catch (err) {
      // 4xx do Python (imagem inválida, parâmetros) não melhora com retry.
      if (err instanceof WorkerHttpError && !err.retryable) throw new UnrecoverableError(err.message)
      throw err
    } finally {
      clearInterval(poll)
    }
  },
  { connection: createRedis(), concurrency: config.queueConcurrency },
)

worker.on('completed', (job, result) =>
  log('info', 'job concluído', { jobId: job.id, totalMs: result.totalMs, steps: result.steps.map((s) => `${s.name}:${s.status}`) }))
worker.on('failed', (job, err) =>
  log('error', 'job falhou', { jobId: job?.id, attempt: job?.attemptsMade, error: err.message }))
worker.on('error', (err) => log('error', 'erro no worker', { error: err.message }))

log('info', 'worker iniciado', { queue: config.queueName, concurrency: config.queueConcurrency, pythonWorker: config.workerUrl })

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.once(signal, () => {
    log('info', `${signal} recebido, aguardando jobs em andamento`)
    stopCleanup()
    worker.close().then(() => process.exit(0), () => process.exit(1))
  })
}
