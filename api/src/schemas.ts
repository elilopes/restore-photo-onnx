import { z } from 'zod'

/** Opções que o frontend envia (camelCase). O worker Python recebe a versão snake_case (ver workerClient.ts). */
export const optionsSchema = z.object({
  repair: z.boolean().default(true),
  repairSensitivity: z.number().min(0).max(1).default(0.5),
  colorize: z.enum(['auto', 'on', 'off']).default('auto'),
  enhanceColors: z.boolean().default(true),
  enhanceStrength: z.number().min(0).max(1).default(0.5),
  sharpen: z.boolean().default(true),
  sharpenStrength: z.number().min(0).max(1).default(0.5),
  upscale: z.union([z.literal(1), z.literal(2), z.literal(4)]).default(1),
  outputFormat: z.enum(['png', 'jpg']).default('png'),
  useModels: z.boolean().default(true),
  // Novos modelos
  lowLight: z.boolean().default(false),
  faceRestore: z.boolean().default(false),
  faceRestoreModel: z.enum(['gfpgan', 'codeformer']).default('gfpgan'),
  styleTransfer: z.boolean().default(false),
  styleName: z.string().default('candy'),
  styleStrength: z.number().min(0).max(1).default(0.7),
})

/** Resultado da análise de rostos (espelho de AnalyzeFaceResponse do worker). */
export interface FaceAnalysisResult {
  faceCount: number
  faceCoverage: number
  suggestFaceRestore: boolean
}
export type RestoreOptions = z.infer<typeof optionsSchema>

export interface RestoreJobData {
  jobId: string
  inputName: string
  options: RestoreOptions
}

export interface StepReport {
  name: string
  status: 'done' | 'skipped' | 'failed'
  engine: string | null
  detail: string | null
  ms: number
}

export interface RestoreJobResult {
  resultName: string
  maskName: string | null
  inputWidth: number
  inputHeight: number
  width: number
  height: number
  wasMonochrome: boolean
  steps: StepReport[]
  totalMs: number
}

export interface JobProgress {
  percent: number
  step: string
}

/** Mesmo padrão do worker Python (UUID minúsculo). */
export const JOB_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/

export type JobStatus = 'queued' | 'processing' | 'done' | 'failed'

export interface JobView {
  id: string
  status: JobStatus
  progress: JobProgress
  options: RestoreOptions
  result?: Omit<RestoreJobResult, 'resultName' | 'maskName'> & {
    urls: { original: string; result: string; mask: string | null }
  }
  error?: string
}
