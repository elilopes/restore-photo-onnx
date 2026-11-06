import { createServer, type Server } from 'node:http'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'

vi.hoisted(() => {
  process.env.WORKER_URL = 'http://127.0.0.1:18765'
})

import type { RestoreJobData } from '../src/schemas.js'
import { optionsSchema } from '../src/schemas.js'
import { fetchProgress, processJob, toWorkerPayload, WorkerHttpError, workerHealth } from '../src/workerClient.js'

const JOB = '11111111-1111-4111-8111-111111111111'
const data: RestoreJobData = { jobId: JOB, inputName: `${JOB}.png`, options: optionsSchema.parse({ upscale: 2, repairSensitivity: 0.7 }) }

let server: Server
let lastBody: any
let mode: 'ok' | 'bad-image' | 'crash' = 'ok'

beforeAll(async () => {
  server = createServer((req, res) => {
    const chunks: Buffer[] = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      res.setHeader('content-type', 'application/json')
      if (req.url === '/health') return void res.end('{"status":"ok"}')
      if (req.url === `/progress/${JOB}`) return void res.end('{"progress":0.45,"step":"colorize"}')
      if (req.url?.startsWith('/progress/')) return void res.writeHead(404).end('{}')
      if (req.url === '/process') {
        lastBody = JSON.parse(Buffer.concat(chunks).toString())
        if (mode === 'bad-image') return void res.writeHead(422).end('{"detail":"não foi possível decodificar"}')
        if (mode === 'crash') return void res.writeHead(500).end('{"detail":"RuntimeError: x"}')
        return void res.end(JSON.stringify({
          job_id: JOB, result_name: 'result.png', mask_name: null, input_width: 10, input_height: 8, width: 20, height: 16,
          was_monochrome: true, total_ms: 12,
          steps: [{ name: 'load', status: 'done', engine: null, detail: '10x8', ms: 3 }, { name: 'sharpen', status: 'done', ms: 1 }],
        }))
      }
      res.writeHead(404).end('{}')
    })
  })
  await new Promise<void>((r) => server.listen(18765, '127.0.0.1', r))
})
afterAll(() => new Promise<void>((r) => server.close(() => r())))

describe('workerClient', () => {
  it('toWorkerPayload converte camelCase -> snake_case', () => {
    expect(toWorkerPayload(data)).toEqual({
      job_id: JOB,
      input_name: `${JOB}.png`,
      options: {
        repair: true, repair_sensitivity: 0.7, colorize: 'auto', enhance_colors: true, enhance_strength: 0.5,
        sharpen: true, sharpen_strength: 0.5, upscale: 2, output_format: 'png', use_models: true,
      },
    })
  })

  it('processJob envia o payload e converte a resposta', async () => {
    mode = 'ok'
    const result = await processJob(data)
    expect(lastBody.options.repair_sensitivity).toBe(0.7)
    expect(result).toMatchObject({ resultName: 'result.png', maskName: null, width: 20, height: 16, wasMonochrome: true, totalMs: 12 })
    expect(result.steps[1]).toEqual({ name: 'sharpen', status: 'done', engine: null, detail: null, ms: 1 })
  })

  it('4xx do worker não é "retryable"; 5xx é', async () => {
    mode = 'bad-image'
    const err = await processJob(data).catch((e) => e)
    expect(err).toBeInstanceOf(WorkerHttpError)
    expect(err.status).toBe(422)
    expect(err.retryable).toBe(false)

    mode = 'crash'
    const err2 = await processJob(data).catch((e) => e)
    expect(err2.retryable).toBe(true)
  })

  it('fetchProgress e workerHealth', async () => {
    expect(await fetchProgress(JOB)).toEqual({ progress: 0.45, step: 'colorize' })
    expect(await fetchProgress('22222222-2222-4222-8222-222222222222')).toBeNull()
    expect(await workerHealth()).toBe(true)
  })
})
