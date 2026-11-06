import { mkdir, mkdtemp, readdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import type { FastifyInstance } from 'fastify'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  process.env.MAX_UPLOAD_MB = '1'
})

import { buildApp } from '../src/app.js'
import type { RestoreJobData, RestoreJobResult } from '../src/schemas.js'
import { LocalStorage } from '../src/storage.js'

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.alloc(64, 7)])
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

class FakeJob {
  state = 'waiting'
  progress: unknown = 0
  returnvalue: RestoreJobResult | null = null
  failedReason = ''
  removed = false
  constructor(public data: RestoreJobData) {}
  async getState() { return this.state }
  async remove() { this.removed = true }
}

class FakeQueue {
  jobs = new Map<string, FakeJob>()
  add(data: RestoreJobData) {
    const job = new FakeJob(data)
    this.jobs.set(data.jobId, job)
  }
  getJob(id: string) { return this.jobs.get(id) ?? null }
}

async function multipart(file: Buffer | null, options?: unknown, filename = 'foto.png') {
  const fd = new FormData()
  if (file) fd.append('file', new Blob([new Uint8Array(file)], { type: 'image/png' }), filename)
  if (options !== undefined) fd.append('options', typeof options === 'string' ? options : JSON.stringify(options))
  const req = new Request('http://local/x', { method: 'POST', body: fd })
  return { payload: Buffer.from(await req.arrayBuffer()), headers: { 'content-type': req.headers.get('content-type')! } }
}

describe('API de jobs', () => {
  let app: FastifyInstance
  let queue: FakeQueue
  let storage: LocalStorage

  beforeEach(async () => {
    queue = new FakeQueue()
    storage = new LocalStorage(await mkdtemp(path.join(tmpdir(), 'rp-api-')))
    await storage.init()
    app = await buildApp({ queue: queue as never, storage })
  })
  afterEach(() => app.close())

  const post = async (file: Buffer | null, options?: unknown, filename?: string) =>
    app.inject({ method: 'POST', url: '/api/jobs', ...(await multipart(file, options, filename)) })

  it('POST /api/jobs enfileira com opções padrão e grava o upload', async () => {
    const res = await post(PNG)
    expect(res.statusCode).toBe(202)
    const { id, status } = res.json()
    expect(id).toMatch(UUID_RE)
    expect(status).toBe('queued')
    expect(await readdir(storage.uploadsDir)).toEqual([`${id}.png`])
    const job = queue.jobs.get(id)!
    expect(job.data).toMatchObject({ jobId: id, inputName: `${id}.png`, options: { repair: true, upscale: 1, outputFormat: 'png', colorize: 'auto' } })
  })

  it('aceita opções customizadas', async () => {
    const res = await post(PNG, { upscale: 2, colorize: 'on', sharpenStrength: 0.9, outputFormat: 'jpg', repair: false })
    expect(res.statusCode).toBe(202)
    expect(queue.jobs.get(res.json().id)!.data.options).toMatchObject({
      upscale: 2, colorize: 'on', sharpenStrength: 0.9, outputFormat: 'jpg', repair: false, enhanceColors: true,
    })
  })

  it.each([
    ['upscale inválido', { upscale: 3 }],
    ['força fora da faixa', { sharpenStrength: 2 }],
    ['colorize desconhecido', { colorize: 'talvez' }],
    ['JSON quebrado', '{nao-json'],
  ])('rejeita %s com 400 e não deixa o upload em disco', async (_name, options) => {
    const res = await post(PNG, options)
    expect(res.statusCode).toBe(400)
    expect(queue.jobs.size).toBe(0)
    expect(await readdir(storage.uploadsDir)).toEqual([])
  })

  it('rejeita arquivo que não é imagem (415) e requisição sem arquivo (400)', async () => {
    const bad = await post(Buffer.from('%PDF-1.7 ...'), undefined, 'foto.png')
    expect(bad.statusCode).toBe(415)
    expect(await readdir(storage.uploadsDir)).toEqual([])
    expect((await post(null, {})).statusCode).toBe(400)
    expect((await app.inject({ method: 'POST', url: '/api/jobs', payload: { a: 1 } })).statusCode).toBe(415)
  })

  it('rejeita upload acima do limite (413) e limpa o arquivo parcial', async () => {
    const big = Buffer.concat([PNG, Buffer.alloc(1.2 * 1024 * 1024)])
    const res = await post(big)
    expect(res.statusCode).toBe(413)
    expect(await readdir(storage.uploadsDir)).toEqual([])
  })

  it('GET /api/jobs/:id reflete os estados da fila', async () => {
    const { id } = (await post(PNG)).json()
    const job = queue.jobs.get(id)!

    expect((await app.inject(`/api/jobs/${id}`)).json()).toMatchObject({ id, status: 'queued', progress: { percent: 0 } })

    job.state = 'active'
    job.progress = { percent: 45, step: 'colorize' }
    expect((await app.inject(`/api/jobs/${id}`)).json()).toMatchObject({ status: 'processing', progress: { percent: 45, step: 'colorize' } })

    job.state = 'failed'
    job.failedReason = 'boom'
    expect((await app.inject(`/api/jobs/${id}`)).json()).toMatchObject({ status: 'failed', error: 'boom' })

    job.state = 'completed'
    job.returnvalue = {
      resultName: 'result.png', maskName: 'mask.png', inputWidth: 10, inputHeight: 10, width: 20, height: 20,
      wasMonochrome: false, totalMs: 5, steps: [{ name: 'load', status: 'done', engine: null, detail: null, ms: 1 }],
    }
    const done = (await app.inject(`/api/jobs/${id}`)).json()
    expect(done).toMatchObject({ status: 'done', progress: { percent: 100 }, result: { width: 20, urls: { result: `/api/jobs/${id}/result`, mask: `/api/jobs/${id}/mask` } } })
    expect(done.result.resultName).toBeUndefined() // detalhe interno não vaza
  })

  it('valida id e devolve 404 para job desconhecido', async () => {
    expect((await app.inject('/api/jobs/nao-e-uuid')).statusCode).toBe(400)
    expect((await app.inject('/api/jobs/99999999-9999-4999-8999-999999999999')).statusCode).toBe(404)
    expect((await app.inject('/api/jobs/..%2F..%2Fetc%2Fpasswd')).statusCode).toBe(400)
  })

  it('serve original, resultado e máscara; 409 antes de concluir; download com attachment', async () => {
    const { id } = (await post(PNG)).json()
    const job = queue.jobs.get(id)!

    const original = await app.inject(`/api/jobs/${id}/original`)
    expect(original.statusCode).toBe(200)
    expect(original.headers['content-type']).toBe('image/png')
    expect(original.rawPayload.equals(PNG)).toBe(true)

    expect((await app.inject(`/api/jobs/${id}/result`)).statusCode).toBe(409)

    await mkdir(path.join(storage.resultsDir, id))
    await writeFile(path.join(storage.resultsDir, id, 'result.jpg'), Buffer.from('JPEGDATA'))
    job.state = 'completed'
    job.returnvalue = {
      resultName: 'result.jpg', maskName: null, inputWidth: 1, inputHeight: 1, width: 1, height: 1,
      wasMonochrome: false, totalMs: 1, steps: [],
    }
    const result = await app.inject(`/api/jobs/${id}/result?download=1`)
    expect(result.statusCode).toBe(200)
    expect(result.headers['content-type']).toBe('image/jpeg')
    expect(result.headers['content-disposition']).toContain(`restaurada-${id.slice(0, 8)}.jpg`)
    expect(result.body).toBe('JPEGDATA')

    expect((await app.inject(`/api/jobs/${id}/mask`)).statusCode).toBe(404)
    expect((await app.inject(`/api/jobs/${id}/segredo`)).statusCode).toBe(404)
  })

  it('DELETE remove job e arquivos; job ativo dá 409', async () => {
    const { id } = (await post(PNG)).json()
    const job = queue.jobs.get(id)!

    job.state = 'active'
    expect((await app.inject({ method: 'DELETE', url: `/api/jobs/${id}` })).statusCode).toBe(409)

    job.state = 'completed'
    expect((await app.inject({ method: 'DELETE', url: `/api/jobs/${id}` })).statusCode).toBe(204)
    expect(job.removed).toBe(true)
    expect(await readdir(storage.uploadsDir)).toEqual([])
  })

  it('/api/health responde 200 com worker ok', async () => {
    const res = await app.inject('/api/health')
    expect(res.json()).toMatchObject({ api: 'ok' })
  })
})
