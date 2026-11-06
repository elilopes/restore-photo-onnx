/**
 * Fila de jobs em memória — substitui BullMQ + Redis.
 * Processa um job por vez (concurrency=1); jobs ficam em memória até
 * serem removidos ou o servidor reiniciar.
 */
import { config } from './config.js'
import type { RestoreJobData, RestoreJobResult } from './schemas.js'
import { fetchProgress, processJob, WorkerHttpError } from './workerClient.js'

export type JobState = 'queued' | 'active' | 'completed' | 'failed'

export interface InMemoryJob {
  data: RestoreJobData
  getState(): Promise<JobState>
  progress: { percent: number; step: string }
  returnvalue: RestoreJobResult | null
  failedReason: string
  remove(): Promise<void>
}

interface JobEntry {
  data: RestoreJobData
  state: JobState
  progress: { percent: number; step: string }
  returnvalue: RestoreJobResult | null
  failedReason: string
}

export class JobStore {
  private jobs = new Map<string, JobEntry>()
  /** Fila de jobIds aguardando processamento */
  private queue: string[] = []
  /** Promise do job em andamento (garante 1 job por vez) */
  private running: Promise<void> = Promise.resolve()

  add(data: RestoreJobData): void {
    const entry: JobEntry = {
      data,
      state: 'queued',
      progress: { percent: 0, step: 'queued' },
      returnvalue: null,
      failedReason: '',
    }
    this.jobs.set(data.jobId, entry)
    this.queue.push(data.jobId)
    // Encadeia na fila serial (sem sobreposição)
    this.running = this.running.then(() => this._processNext())
  }

  getJob(jobId: string): InMemoryJob | null {
    const entry = this.jobs.get(jobId)
    if (!entry) return null
    return this._wrap(jobId, entry)
  }

  private _wrap(jobId: string, entry: JobEntry): InMemoryJob {
    return {
      data: entry.data,
      getState: async () => entry.state,
      get progress() { return entry.progress },
      get returnvalue() { return entry.returnvalue },
      get failedReason() { return entry.failedReason },
      remove: async () => { this.jobs.delete(jobId) },
    }
  }

  private async _processNext(): Promise<void> {
    const jobId = this.queue.shift()
    if (!jobId) return

    const entry = this.jobs.get(jobId)
    if (!entry) return // foi removido enquanto aguardava

    entry.state = 'active'
    entry.progress = { percent: 1, step: 'start' }

    // Polling de progresso do worker Python
    const poll = setInterval(() => {
      fetchProgress(jobId)
        .then((p) => {
          if (p && entry.state === 'active') {
            entry.progress = { percent: Math.round(p.progress * 100), step: p.step }
          }
        })
        .catch(() => undefined)
    }, config.progressPollMs)

    try {
      const result = await processJob(entry.data)
      entry.returnvalue = result
      entry.state = 'completed'
      entry.progress = { percent: 100, step: 'done' }
    } catch (err) {
      const msg = err instanceof WorkerHttpError
        ? err.message
        : err instanceof Error ? err.message : 'falha desconhecida'
      entry.failedReason = msg
      entry.state = 'failed'
    } finally {
      clearInterval(poll)
    }

    // Auto-limpar após retention configurado
    setTimeout(
      () => this.jobs.delete(jobId),
      config.retentionSeconds * 1000,
    ).unref()
  }
}
