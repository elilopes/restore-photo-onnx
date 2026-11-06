/** Erro com status HTTP; o error handler do Fastify devolve `message` para 4xx. */
export class HttpError extends Error {
  constructor(readonly statusCode: number, message: string) {
    super(message)
    this.name = 'HttpError'
  }
}
