import { describe, expect, it, vi } from 'vitest'
import { ApiError, buildForm, createJob, fileUrl, getJob } from './api'
import { DEFAULT_OPTIONS } from './defaults'

const photo = new File(['png-bytes'], 'avo.png', { type: 'image/png' })

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

describe('api', () => {
  it('buildForm envia o arquivo e as opções como JSON', async () => {
    const form = buildForm(photo, { ...DEFAULT_OPTIONS, upscale: 2 })
    const file = form.get('file') as File
    expect(file.name).toBe('avo.png')
    expect(JSON.parse(form.get('options') as string)).toMatchObject({ upscale: 2, repair: true })
  })

  it('createJob devolve o id do job', async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ id: 'abc' }, 202))
    await expect(createJob(photo, DEFAULT_OPTIONS, fetchFn)).resolves.toEqual({ id: 'abc' })
    expect(fetchFn).toHaveBeenCalledWith('/api/jobs', expect.objectContaining({ method: 'POST' }))
  })

  it('createJob mostra a mensagem de erro que a API devolveu', async () => {
    const fetchFn = vi.fn().mockResolvedValue(jsonResponse({ error: 'formato não suportado' }, 415))
    const err = await createJob(photo, DEFAULT_OPTIONS, fetchFn).catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(415)
    expect(err.message).toBe('formato não suportado')
  })

  it('getJob trata resposta sem JSON sem quebrar', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response('<html>502</html>', { status: 502 }))
    await expect(getJob('abc', fetchFn)).rejects.toMatchObject({ status: 502, message: 'Erro 502 do servidor.' })
  })

  it('fileUrl monta as rotas de arquivo', () => {
    expect(fileUrl('abc', 'result')).toBe('/api/jobs/abc/result')
    expect(fileUrl('abc', 'mask', true)).toBe('/api/jobs/abc/mask?download=1')
  })
})
