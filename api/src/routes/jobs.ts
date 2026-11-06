import { randomUUID } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import type { FastifyPluginAsync } from 'fastify'
import { z } from 'zod'
import { HttpError } from '../errors.js'
import type { JobStore, InMemoryJob } from '../jobStore.js'
import {
  JOB_ID_RE,
  optionsSchema,
  type FaceAnalysisResult,
  type JobProgress,
  type JobStatus,
  type JobView,
  type RestoreJobData,
} from '../schemas.js'
import { mimeFromName, type LocalStorage } from '../storage.js'
import { analyzeImage } from '../workerClient.js'

export interface JobRoutesOptions {
  queue: JobStore
  storage: LocalStorage
}

const FILE_KINDS = ['original', 'result', 'mask'] as const
type FileKind = (typeof FILE_KINDS)[number]

function mapStatus(state: string): JobStatus {
  switch (state) {
    case 'completed': return 'done'
    case 'failed':    return 'failed'
    case 'active':    return 'processing'
    default:          return 'queued'
  }
}

function normalizeProgress(raw: unknown): JobProgress {
  if (typeof raw === 'number') return { percent: Math.min(100, Math.max(0, raw)), step: '' }
  if (typeof raw === 'object' && raw !== null && 'percent' in raw) {
    const { percent, step } = raw as { percent: unknown; step?: unknown }
    return { percent: Math.min(100, Math.max(0, Number(percent) || 0)), step: typeof step === 'string' ? step : '' }
  }
  return { percent: 0, step: 'queued' }
}

async function toView(job: InMemoryJob): Promise<JobView> {
  const id = job.data.jobId
  const status = mapStatus(await job.getState())
  const view: JobView = {
    id,
    status,
    progress: status === 'done' ? { percent: 100, step: 'done' } : normalizeProgress(job.progress),
    options: job.data.options,
  }
  if (status === 'done' && job.returnvalue) {
    const { resultName: _r, maskName, ...rest } = job.returnvalue
    view.result = {
      ...rest,
      urls: {
        original: `/api/jobs/${id}/original`,
        result:   `/api/jobs/${id}/result`,
        mask:     maskName ? `/api/jobs/${id}/mask` : null,
      },
    }
  }
  if (status === 'failed') view.error = job.failedReason || 'falha desconhecida'
  return view
}

export const jobRoutes: FastifyPluginAsync<JobRoutesOptions> = async (app, { queue, storage }) => {
  function requireJob(id: string): InMemoryJob {
    if (!JOB_ID_RE.test(id)) throw new HttpError(400, 'id inválido')
    const job = queue.getJob(id)
    if (!job) throw new HttpError(404, 'job não encontrado')
    return job
  }

  // Criar job: multipart com `file` (imagem) e `options` (JSON opcional).
  app.post('/api/jobs', async (req, reply) => {
    if (!req.isMultipart()) throw new HttpError(415, 'envie multipart/form-data com os campos file e options')

    const jobId = randomUUID()
    let rawOptions: unknown = {}
    let sawFile = false
    let upload: { name: string; mime: string } | null = null

    try {
      for await (const part of req.parts()) {
        if (part.type === 'file') {
          if (part.fieldname !== 'file' || sawFile) { part.file.resume(); continue }
          sawFile = true
          upload = await storage.saveUpload(jobId, part.file)
        } else if (part.fieldname === 'options' && typeof part.value === 'string' && part.value.trim()) {
          try { rawOptions = JSON.parse(part.value) }
          catch { throw new HttpError(400, 'o campo options precisa ser um JSON válido') }
        }
      }

      if (!sawFile) throw new HttpError(400, 'campo file é obrigatório')
      if (!upload)  throw new HttpError(415, 'formato não suportado (use JPEG, PNG, WebP, TIFF ou BMP)')

      const parsed = optionsSchema.safeParse(rawOptions)
      if (!parsed.success) throw new HttpError(400, `opções inválidas: ${z.prettifyError(parsed.error)}`)

      const data: RestoreJobData = { jobId, inputName: upload.name, options: parsed.data }
      queue.add(data)
    } catch (err) {
      await storage.removeJob(jobId)
      throw err
    }

    return reply.code(202).send({ id: jobId, status: 'queued' satisfies JobStatus })
  })

  app.get<{ Params: { id: string } }>('/api/jobs/:id', async (req) => toView(requireJob(req.params.id)))

  app.get<{ Params: { id: string; kind: string }; Querystring: { download?: string } }>(
    '/api/jobs/:id/:kind',
    async (req, reply) => {
      const kind = req.params.kind as FileKind
      if (!FILE_KINDS.includes(kind)) throw new HttpError(404, 'recurso desconhecido')
      const job = requireJob(req.params.id)
      const id = job.data.jobId

      let file: string
      let name: string
      if (kind === 'original') {
        name = job.data.inputName
        file = storage.uploadPath(name)
      } else {
        const state = await job.getState()
        const rv = state === 'completed' ? job.returnvalue : null
        if (!rv) throw new HttpError(409, 'resultado ainda não disponível')
        const resultFile = kind === 'result' ? rv.resultName : rv.maskName
        if (!resultFile) throw new HttpError(404, 'este job não gerou máscara')
        name = resultFile
        file = storage.resultPath(id, name)
      }

      const info = await stat(file).catch(() => null)
      if (!info?.isFile()) throw new HttpError(404, 'arquivo não encontrado (expirado?)')

      reply
        .header('content-type', mimeFromName(name))
        .header('content-length', info.size)
        .header('cache-control', 'private, max-age=3600')
      if (req.query.download) {
        const ext = name.split('.').pop()
        reply.header('content-disposition', `attachment; filename="${kind === 'result' ? 'restaurada' : kind}-${id.slice(0, 8)}.${ext}"`)
      }
      return reply.send(createReadStream(file))
    },
  )

  // Analisa rostos na imagem (após upload) sem processar o job.
  app.get<{ Params: { id: string } }>('/api/jobs/:id/analyze', async (req): Promise<FaceAnalysisResult> => {
    const job = requireJob(req.params.id)
    const result = await analyzeImage(job.data.inputName)
    if (!result) return { faceCount: 0, faceCoverage: 0, suggestFaceRestore: false }
    return result
  })

  app.delete<{ Params: { id: string } }>('/api/jobs/:id', async (req, reply) => {
    const job = requireJob(req.params.id)
    if ((await job.getState()) === 'active') throw new HttpError(409, 'job em processamento; aguarde terminar')
    await job.remove()
    await storage.removeJob(job.data.jobId)
    return reply.code(204).send()
  })
}
