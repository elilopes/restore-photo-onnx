import { Redis } from 'ioredis'
import { config } from './config.js'

/** `maxRetriesPerRequest: null` é exigido pelo BullMQ para conexões de Worker (comandos bloqueantes). */
export function createRedis(): Redis {
  return new Redis(config.redisUrl, { maxRetriesPerRequest: null })
}

/** PING com timeout (com `maxRetriesPerRequest: null` um Redis fora do ar deixaria o comando pendente para sempre). */
export async function pingRedis(redis: Redis, timeoutMs = 1500): Promise<boolean> {
  const timeout = new Promise<boolean>((resolve) => setTimeout(() => resolve(false), timeoutMs).unref())
  return Promise.race([redis.ping().then((r) => r === 'PONG', () => false), timeout])
}
