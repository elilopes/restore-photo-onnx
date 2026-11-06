import cors from '@fastify/cors'
import multipart from '@fastify/multipart'
import Fastify, { type FastifyInstance } from 'fastify'
import { config } from './config.js'
import { jobRoutes, type JobRoutesOptions } from './routes/jobs.js'
import { systemRoutes } from './routes/system.js'

export type AppDeps = JobRoutesOptions

export async function buildApp(deps: AppDeps): Promise<FastifyInstance> {
  const app = Fastify({ logger: true, bodyLimit: 1024 * 1024 })

  await app.register(cors, {
    origin: config.corsOrigin.split(',').map((o) => o.trim()),
    methods: ['GET', 'POST', 'DELETE', 'OPTIONS'],
  })
  await app.register(multipart, {
    limits: { fileSize: config.maxUploadBytes, files: 1, fields: 4, fieldSize: 16 * 1024 },
  })

  app.setErrorHandler((err: Error & { statusCode?: number }, req, reply) => {
    const status = err.statusCode && err.statusCode >= 400 ? err.statusCode : 500
    if (status >= 500) req.log.error(err)
    const message = status === 413
      ? `arquivo maior que ${Math.round(config.maxUploadBytes / 1024 / 1024)} MB`
      : err.message
    reply.code(status).send({ error: status >= 500 ? 'erro interno' : message })
  })

  await app.register(jobRoutes, deps)
  await app.register(systemRoutes)
  return app
}
