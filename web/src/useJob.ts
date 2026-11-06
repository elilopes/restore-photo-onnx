import { useEffect, useState } from 'react'
import { ApiError, getJob } from './api'
import type { JobView } from './types'

/**
 * Consulta o job a cada `pollMs` até terminar. Erros de rede são tentados de novo;
 * erros 4xx (ex.: job removido) param o polling.
 */
export function useJob(jobId: string | null, pollMs = 1000): { job: JobView | null; error: string | null } {
  const [state, setState] = useState<{ id: string; view: JobView | null; error: string | null } | null>(null)

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    const tick = async () => {
      try {
        const view = await getJob(jobId)
        if (cancelled) return
        setState({ id: jobId, view, error: null })
        if (view.status === 'done' || view.status === 'failed') return
      } catch (err) {
        if (cancelled) return
        const message = err instanceof Error ? err.message : 'Falha de conexão.'
        setState({ id: jobId, view: null, error: message })
        if (err instanceof ApiError && err.status < 500) return
      }
      timer = setTimeout(tick, pollMs)
    }

    void tick()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [jobId, pollMs])

  // Só expõe o estado do job atual (não o de um job anterior ainda na memória).
  if (!jobId || !state || state.id !== jobId) return { job: null, error: null }
  return { job: state.view, error: state.error }
}
