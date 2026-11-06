import { Agent, fetch } from 'undici'
import { config } from './config.js'
import type { FaceAnalysisResult, RestoreJobData, RestoreJobResult, StepReport } from './schemas.js'

// O processamento é síncrono e pode levar minutos: os timeouts padrão do undici (5 min) não servem.
const dispatcher = new Agent({
  connectTimeout: 5_000,
  headersTimeout: config.workerTimeoutMs,
  bodyTimeout: config.workerTimeoutMs,
})

/** Erro HTTP do worker Python. 4xx (arquivo inválido, parâmetros) não adianta repetir. */
export class WorkerHttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'WorkerHttpError'
  }
  get retryable(): boolean {
    return this.status >= 500
  }
}

/** Contrato do worker Python (snake_case) — espelho de worker/app/schemas.py. */
interface WorkerResponse {
  result_name: string
  mask_name: string | null
  input_width: number
  input_height: number
  width: number
  height: number
  was_monochrome: boolean
  steps: Array<Omit<StepReport, 'engine' | 'detail'> & { engine?: string | null; detail?: string | null }>
  total_ms: number
}

export function toWorkerPayload(data: RestoreJobData) {
  const o = data.options
  return {
    job_id: data.jobId,
    input_name: data.inputName,
    options: {
      repair: o.repair,
      repair_sensitivity: o.repairSensitivity,
      colorize: o.colorize,
      enhance_colors: o.enhanceColors,
      enhance_strength: o.enhanceStrength,
      sharpen: o.sharpen,
      sharpen_strength: o.sharpenStrength,
      upscale: o.upscale,
      output_format: o.outputFormat,
      use_models: o.useModels,
      low_light: o.lowLight,
      face_restore: o.faceRestore,
      face_restore_model: o.faceRestoreModel,
      style_transfer: o.styleTransfer,
      style_name: o.styleName,
      style_strength: o.styleStrength,
    },
  }
}

export function fromWorkerResponse(r: WorkerResponse): RestoreJobResult {
  return {
    resultName: r.result_name,
    maskName: r.mask_name,
    inputWidth: r.input_width,
    inputHeight: r.input_height,
    width: r.width,
    height: r.height,
    wasMonochrome: r.was_monochrome,
    steps: r.steps.map((s) => ({ ...s, engine: s.engine ?? null, detail: s.detail ?? null })),
    totalMs: r.total_ms,
  }
}

export async function processJob(data: RestoreJobData): Promise<RestoreJobResult> {
  const res = await fetch(`${config.workerUrl}/process`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(toWorkerPayload(data)),
    dispatcher,
  })
  if (!res.ok) {
    throw new WorkerHttpError(res.status, `worker respondeu ${res.status}: ${(await res.text()).slice(0, 500)}`)
  }
  return fromWorkerResponse((await res.json()) as WorkerResponse)
}

export async function fetchProgress(jobId: string): Promise<{ progress: number; step: string } | null> {
  try {
    const res = await fetch(`${config.workerUrl}/progress/${jobId}`, { signal: AbortSignal.timeout(2_000) })
    if (!res.ok) {
      await res.body?.cancel()
      return null
    }
    return (await res.json()) as { progress: number; step: string }
  } catch {
    return null
  }
}

export async function workerHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${config.workerUrl}/health`, { signal: AbortSignal.timeout(2_000) })
    await res.body?.cancel()
    return res.ok
  } catch {
    return false
  }
}

export async function workerModels(): Promise<unknown | null> {
  try {
    const res = await fetch(`${config.workerUrl}/models`, { signal: AbortSignal.timeout(2_000) })
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}

export async function startModelDownload(only?: string): Promise<{ status: string } | null> {
  try {
    const url = only
      ? `${config.workerUrl}/models/download?only=${encodeURIComponent(only)}`
      : `${config.workerUrl}/models/download`
    const res = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(5_000) })
    return res.ok ? (await res.json()) as { status: string } : null
  } catch {
    return null
  }
}

export async function modelDownloadStatus(): Promise<unknown | null> {
  try {
    const res = await fetch(`${config.workerUrl}/models/download/status`, { signal: AbortSignal.timeout(2_000) })
    return res.ok ? await res.json() : null
  } catch {
    return null
  }
}

export async function analyzeImage(inputName: string): Promise<FaceAnalysisResult | null> {
  try {
    const res = await fetch(`${config.workerUrl}/analyze`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ input_name: inputName }),
      signal: AbortSignal.timeout(10_000),
    })
    if (!res.ok) return null
    const data = await res.json() as { face_count: number; face_coverage: number; suggest_face_restore: boolean }
    return {
      faceCount: data.face_count,
      faceCoverage: data.face_coverage,
      suggestFaceRestore: data.suggest_face_restore,
    }
  } catch {
    return null
  }
}
