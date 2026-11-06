import { config } from './config.js'
import type { LocalStorage } from './storage.js'

const HOUR_MS = 3_600_000

/** Remove uploads/resultados mais antigos que RETENTION_HOURS (na partida e a cada hora). */
export function startCleanup(storage: LocalStorage, log: (msg: string) => void = console.log): () => void {
  const run = async () => {
    try {
      const removed = await storage.purgeOlderThan(config.retentionHours * HOUR_MS)
      if (removed > 0) log(`limpeza: ${removed} itens antigos removidos`)
    } catch (err) {
      log(`limpeza falhou: ${(err as Error).message}`)
    }
  }
  void run()
  const timer = setInterval(run, HOUR_MS)
  timer.unref()
  return () => clearInterval(timer)
}
