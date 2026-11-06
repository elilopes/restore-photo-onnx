// Espelho dos tipos de api/src/schemas.ts (camelCase, como a API devolve).

export type Colorize = 'auto' | 'on' | 'off'
export type Upscale = 1 | 2 | 4
export type OutputFormat = 'png' | 'jpg'
export type FaceRestoreModel = 'gfpgan' | 'codeformer'
export type StyleName = 'candy' | 'mosaic' | 'rain_princess' | 'udnie'

export interface RestoreOptions {
  repair: boolean
  repairSensitivity: number
  colorize: Colorize
  enhanceColors: boolean
  enhanceStrength: number
  sharpen: boolean
  sharpenStrength: number
  upscale: Upscale
  outputFormat: OutputFormat
  useModels: boolean
  // Novos modelos
  lowLight: boolean
  faceRestore: boolean
  faceRestoreModel: FaceRestoreModel
  styleTransfer: boolean
  styleName: StyleName
  styleStrength: number
}

export interface FaceAnalysis {
  faceCount: number
  faceCoverage: number
  suggestFaceRestore: boolean
}

export interface StepReport {
  name: string
  status: 'done' | 'skipped' | 'failed'
  engine: string | null
  detail: string | null
  ms: number
}

export type JobStatus = 'queued' | 'processing' | 'done' | 'failed'

export interface JobView {
  id: string
  status: JobStatus
  progress: { percent: number; step: string }
  options: RestoreOptions
  result?: {
    inputWidth: number
    inputHeight: number
    width: number
    height: number
    wasMonochrome: boolean
    totalMs: number
    steps: StepReport[]
    urls: { original: string; result: string; mask: string | null }
  }
  error?: string
}
