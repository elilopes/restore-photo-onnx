import type { FastifyPluginAsync } from 'fastify'
import { modelDownloadStatus, startModelDownload, workerHealth, workerModels } from '../workerClient.js'

export const systemRoutes: FastifyPluginAsync = async (app) => {
  // Saúde do sistema: só o worker Python importa agora (sem Redis).
  app.get('/api/health', async (_req, reply) => {
    const worker = await workerHealth()
    return reply.code(worker ? 200 : 503).send({
      api: 'ok',
      worker: worker ? 'ok' : 'down',
    })
  })

  // Quais modelos ONNX o worker enxerga/carregou.
  app.get('/api/models', async (_req, reply) => {
    const models = await workerModels()
    if (!models) return reply.code(502).send({ error: 'worker indisponível' })
    return models
  })

  // Inicia o download dos modelos ONNX em background no worker Python.
  app.post('/api/models/download', async (req, reply) => {
    const { only } = (req.query ?? {}) as { only?: string }
    const result = await startModelDownload(only)
    if (!result) return reply.code(502).send({ error: 'worker indisponível' })
    return reply.code(202).send(result)
  })

  // Progresso do download atual.
  app.get('/api/models/download/status', async (_req, reply) => {
    const status = await modelDownloadStatus()
    if (!status) return reply.code(502).send({ error: 'worker indisponível' })
    return status
  })
}
