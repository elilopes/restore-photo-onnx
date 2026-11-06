import { Queue } from 'bullmq'
import type { Redis } from 'ioredis'
import { config } from './config.js'
import type { RestoreJobData, RestoreJobResult } from './schemas.js'

export type RestoreQueue = Queue<RestoreJobData, RestoreJobResult>

export function createQueue(connection: Redis): RestoreQueue {
  return new Queue<RestoreJobData, RestoreJobResult>(config.queueName, {
    connection,
    defaultJobOptions: {
      attempts: 2,
      backoff: { type: 'exponential', delay: 5_000 },
      removeOnComplete: { age: config.retentionSeconds },
      removeOnFail: { age: config.retentionSeconds },
    },
  })
}
