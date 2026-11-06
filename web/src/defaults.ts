import type { RestoreOptions } from './types'

export const DEFAULT_OPTIONS: RestoreOptions = {
  repair: true,
  repairSensitivity: 0.5,
  colorize: 'auto',
  enhanceColors: true,
  enhanceStrength: 0.5,
  sharpen: true,
  sharpenStrength: 0.5,
  upscale: 1,
  outputFormat: 'png',
  useModels: true,
  lowLight: false,
  faceRestore: false,
  faceRestoreModel: 'gfpgan',
  styleTransfer: false,
  styleName: 'candy',
  styleStrength: 0.7,
}

const STEP_LABELS: Record<string, string> = {
  start: 'Na fila',
  load: 'Lendo a foto',
  low_light: 'Realçando a iluminação',
  repair: 'Removendo rasgos e riscos',
  colorize: 'Colorindo',
  enhance_colors: 'Realçando as cores',
  face_restore: 'Restaurando rostos',
  upscale: 'Aumentando a resolução',
  sharpen: 'Aumentando a nitidez',
  style: 'Aplicando estilo artístico',
  save: 'Salvando o resultado',
  done: 'Concluído',
}

/** Nome amigável de uma etapa (vem do worker Python). */
export function stepLabel(step: string): string {
  return STEP_LABELS[step] ?? 'Processando'
}

export const STATUS_LABELS = {
  done: 'feito',
  skipped: 'não foi preciso',
  failed: 'falhou',
} as const

/** Traduz o identificador técnico do motor em linguagem de usuário. */
export function describeEngine(engine: string | null): string | null {
  if (!engine) return null
  if (engine.startsWith('onnx:')) return 'modelo de IA'
  if (engine === 'opencv-telea') return 'preenchimento de imagem'
  return 'processamento clássico'
}
