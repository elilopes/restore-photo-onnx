import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

const JOB = '11111111-1111-4111-8111-111111111111'

const doneView = {
  id: JOB,
  status: 'done',
  progress: { percent: 100, step: 'done' },
  options: {},
  result: {
    inputWidth: 800,
    inputHeight: 600,
    width: 1600,
    height: 1200,
    wasMonochrome: true,
    totalMs: 4200,
    steps: [
      { name: 'load', status: 'done', engine: null, detail: '800x600', ms: 5 },
      { name: 'repair', status: 'done', engine: 'opencv-telea', detail: '3 defeitos, 0.8% da imagem', ms: 80 },
      { name: 'colorize', status: 'skipped', engine: null, detail: 'modelo indisponível', ms: 0 },
    ],
    urls: { original: `/api/jobs/${JOB}/original`, result: `/api/jobs/${JOB}/result`, mask: `/api/jobs/${JOB}/mask` },
  },
}

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
}

/** Respostas padrão do ModelManager — sempre retorna estado vazio para não interferir nos testes. */
const MM_MODELS = { providers: ['CPUExecutionProvider'], models: [] }
const MM_DL_STATUS = { running: false, error: null, models: [] }

function makeFetch(jobSequence: Response[]) {
  let jobIdx = 0
  return vi.fn().mockImplementation((url: string) => {
    if (url === '/api/models') return Promise.resolve(jsonResponse(MM_MODELS))
    if (url === '/api/models/download/status') return Promise.resolve(jsonResponse(MM_DL_STATUS))
    // Chamadas relacionadas a job (/api/jobs, /api/jobs/:id)
    const res = jobSequence[jobIdx] ?? jobSequence[jobSequence.length - 1]!
    jobIdx++
    return Promise.resolve(res)
  })
}

describe('App', () => {
  it('envia a foto, acompanha o progresso e mostra o antes/depois', async () => {
    const processing = { id: JOB, status: 'processing', progress: { percent: 40, step: 'repair' }, options: {} }
    const fetchMock = makeFetch([
      jsonResponse({ id: JOB, status: 'queued' }, 202),
      jsonResponse(processing),
      jsonResponse(doneView),
    ])
    vi.stubGlobal('fetch', fetchMock)

    const { container } = render(<App pollMs={5} />)
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['png'], 'avo.png', { type: 'image/png' })] } })

    const button = await screen.findByRole('button', { name: 'Restaurar foto' })
    await userEvent.click(button)

    expect(await screen.findByAltText('Foto restaurada')).toBeInTheDocument()
    expect(screen.getByAltText('Foto original')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Baixar restaurada' })).toHaveAttribute('href', `/api/jobs/${JOB}/result?download=1`)
    expect(screen.getByText('Preenchimento de imagem', { exact: false })).toBeInTheDocument()
    expect(screen.getAllByText('feito', { selector: '.step-status' })).toHaveLength(2)
    expect(screen.getByText('não foi preciso')).toBeInTheDocument()

    const post = fetchMock.mock.calls.find(([url, init]) => url === '/api/jobs' && (init as RequestInit)?.method === 'POST')!
    expect(post).toBeDefined()
    expect((post[1] as RequestInit).body).toBeInstanceOf(FormData)
  })

  it('mostra a mensagem do servidor quando o envio é recusado', async () => {
    vi.stubGlobal('fetch', makeFetch([jsonResponse({ error: 'formato não suportado' }, 415)]))
    const { container } = render(<App pollMs={5} />)
    fireEvent.change(container.querySelector('input[type="file"]')!, {
      target: { files: [new File(['png'], 'avo.png', { type: 'image/png' })] },
    })
    await userEvent.click(await screen.findByRole('button', { name: 'Restaurar foto' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('formato não suportado')
  })

  it('mostra o erro do job quando o processamento falha', async () => {
    const failed = { id: JOB, status: 'failed', progress: { percent: 10, step: 'load' }, options: {}, error: 'worker respondeu 500' }
    vi.stubGlobal(
      'fetch',
      makeFetch([jsonResponse({ id: JOB, status: 'queued' }, 202), jsonResponse(failed)]),
    )
    const { container } = render(<App pollMs={5} />)
    fireEvent.change(container.querySelector('input[type="file"]')!, {
      target: { files: [new File(['png'], 'avo.png', { type: 'image/png' })] },
    })
    await userEvent.click(await screen.findByRole('button', { name: 'Restaurar foto' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('worker respondeu 500'))
  })
})
