import { buildApp } from './app.js'
import { config } from './config.js'
import { JobStore } from './jobStore.js'
import { LocalStorage } from './storage.js'

const queue = new JobStore()
const storage = new LocalStorage(config.storageDir)
await storage.init()

const app = await buildApp({ queue, storage })

for (const signal of ['SIGINT', 'SIGTERM'] as const) {
  process.once(signal, () => {
    app.log.info(`${signal} recebido, encerrando`)
    app.close().then(() => process.exit(0), () => process.exit(1))
  })
}

await app.listen({ port: config.port, host: config.host })
