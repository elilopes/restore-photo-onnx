import type { JobView, RestoreOptions } from './types'

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

type FetchFn = typeof fetch

async function readJson(res: Response): Promise<Record<string, unknown>> {
  try {
    return (await res.json()) as Record<string, unknown>
  } catch {
    return {}
  }
}

async function failIfNotOk(res: Response): Promise<Record<string, unknown>> {
  const data = await readJson(res)
  if (!res.ok) {
    const message = typeof data.error === 'string' ? data.error : `Erro ${res.status} do servidor.`
    throw new ApiError(res.status, message)
  }
  return data
}

export function buildForm(file: File, options: RestoreOptions): FormData {
  const form = new FormData()
  form.append('file', file, file.name)
  form.append('options', JSON.stringify(options))
  return form
}

export async function createJob(
  file: File,
  options: RestoreOptions,
  fetchFn: FetchFn = fetch,
): Promise<{ id: string }> {
  const res = await fetchFn('/api/jobs', { method: 'POST', body: buildForm(file, options) })
  const data = await failIfNotOk(res)
  if (typeof data.id !== 'string') throw new ApiError(res.status, 'Resposta inesperada do servidor.')
  return { id: data.id }
}

export async function getJob(id: string, fetchFn: FetchFn = fetch): Promise<JobView> {
  const res = await fetchFn(`/api/jobs/${encodeURIComponent(id)}`)
  return (await failIfNotOk(res)) as unknown as JobView
}

export type FileKind = 'original' | 'result' | 'mask'

export function fileUrl(id: string, kind: FileKind, download = false): string {
  return `/api/jobs/${encodeURIComponent(id)}/${kind}${download ? '?download=1' : ''}`
}
