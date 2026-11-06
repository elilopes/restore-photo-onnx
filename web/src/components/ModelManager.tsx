import { useEffect, useRef, useState } from 'react'

// ---------------------------------------------------------------------------
// Tipos
// ---------------------------------------------------------------------------

interface ModelStatus {
  name: string
  role: string
  loaded: boolean
  present: boolean
  error: string | null
}

interface ModelsResponse {
  providers: string[]
  models: ModelStatus[]
}

interface DownloadModelState {
  name: string
  status: 'pending' | 'downloading' | 'done' | 'skipped' | 'error'
  size_total: number
  size_done: number
  percent: number
  error: string | null
}

interface DownloadStatus {
  running: boolean
  error: string | null
  models: DownloadModelState[]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ROLE_LABEL: Record<string, string> = {
  inpaint: 'Remoção de rasgos (LaMa)',
  colorize: 'Colorização P&B (DDColor)',
  superres: 'Ampliação 4× (Real-ESRGAN)',
  superres_fb: 'Ampliação 4× — fallback (SwinIR)',
  face_gfpgan: 'Restauração facial (GFPGAN)',
  face_codeformer: 'Restauração facial — alternativo (CodeFormer)',
  lowlight: 'Realce de iluminação (SCI)',
  style_candy: 'Estilo Candy',
  style_mosaic: 'Estilo Mosaico',
  style_rain_princess: 'Estilo Rain Princess',
  style_udnie: 'Estilo Udnie',
}

const STATUS_ICON: Record<string, string> = {
  pending: '⏳',
  downloading: '⬇',
  done: '✓',
  skipped: '✓',
  error: '✗',
}

function fmt(bytes: number) {
  if (bytes <= 0) return ''
  return bytes >= 1_000_000 ? `${(bytes / 1_000_000).toFixed(0)} MB` : `${Math.round(bytes / 1000)} KB`
}

// ---------------------------------------------------------------------------
// Componente
// ---------------------------------------------------------------------------

interface Props {
  /** Ocultar se todos os modelos estiverem carregados e nenhum download ativo */
  hideWhenOk?: boolean
}

export function ModelManager({ hideWhenOk = false }: Props) {
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [dl, setDl] = useState<DownloadStatus | null>(null)
  const [starting, setStarting] = useState(false)
  const [guideOpen, setGuideOpen] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Busca status dos modelos
  async function fetchModels() {
    try {
      const r = await fetch('/api/models')
      if (r.ok) setModels((await r.json()) as ModelsResponse)
    } catch { /* worker fora do ar — ignorar */ }
  }

  // Busca progresso do download
  async function fetchDl() {
    try {
      const r = await fetch('/api/models/download/status')
      if (r.ok) {
        const data = (await r.json()) as DownloadStatus
        setDl(data)
        if (!data.running) {
          // Download terminou — atualizar lista de modelos
          fetchModels()
          stopPoll()
        }
      }
    } catch { /* ignorar */ }
  }

  function startPoll() {
    if (pollRef.current) return
    pollRef.current = setInterval(fetchDl, 1500)
  }

  function stopPoll() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    fetchModels()
    // Verifica se já há download em andamento
    fetchDl()
    return stopPoll
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Inicia o polling quando o download é iniciado ou já estava rodando
  useEffect(() => {
    if (dl?.running) startPoll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dl?.running])

  async function handleDownload() {
    setStarting(true)
    try {
      const r = await fetch('/api/models/download', { method: 'POST' })
      if (r.ok) {
        await fetchDl()
        startPoll()
      }
    } finally {
      setStarting(false)
    }
  }

  // ---------------------------------------------------------------------------
  // Ocultar quando não há nada para mostrar
  // ---------------------------------------------------------------------------
  const allLoaded = models?.models.every((m) => m.loaded) ?? false
  const dlActive = dl?.running ?? false

  if (hideWhenOk && allLoaded && !dlActive) return null
  if (!models && !dl) return null // ainda carregando

  // ---------------------------------------------------------------------------
  // Renderização
  // ---------------------------------------------------------------------------
  const missingCount = models?.models.filter((m) => !m.present).length ?? 0

  return (
    <section className="model-manager" aria-labelledby="mm-heading">
      <h2 id="mm-heading">Modelos de IA</h2>

      {/* Lista de modelos */}
      {models && (
        <ul className="model-list">
          {models.models.map((m) => (
            <li key={m.name} className={`model-item model-item--${m.loaded ? 'ok' : m.present ? 'warn' : 'missing'}`}>
              <span className="model-item__icon" aria-hidden="true">
                {m.loaded ? '●' : m.present ? '○' : '○'}
              </span>
              <span className="model-item__label">{ROLE_LABEL[m.role] ?? m.name}</span>
              <span className="model-item__status muted">
                {m.loaded ? 'carregado' : m.present ? 'presente (não carregado)' : 'não encontrado'}
              </span>
            </li>
          ))}
        </ul>
      )}

      {/* Progresso do download */}
      {dl && (dl.running || dl.models.length > 0) && (
        <div className="dl-progress" role="status">
          {dl.models.map((m) => (
            <div key={m.name} className="dl-row">
              <span className={`dl-icon dl-icon--${m.status}`} aria-hidden="true">
                {STATUS_ICON[m.status] ?? '·'}
              </span>
              <span className="dl-name">{m.name}</span>
              {m.status === 'downloading' && (
                <>
                  <progress className="dl-bar" max={100} value={m.percent} aria-label={`Baixando ${m.name}`} />
                  <span className="dl-pct muted">
                    {m.percent}%{m.size_total > 0 ? ` · ${fmt(m.size_done)} / ${fmt(m.size_total)}` : ''}
                  </span>
                </>
              )}
              {(m.status === 'done' || m.status === 'skipped') && (
                <span className="dl-pct muted">{m.status === 'skipped' ? 'já existe' : `${fmt(m.size_total)}`}</span>
              )}
              {m.status === 'error' && <span className="dl-pct error">{m.error}</span>}
            </div>
          ))}
          {dl.error && <p className="error">{dl.error}</p>}
        </div>
      )}

      {/* Botão de download automático */}
      {!dlActive && missingCount > 0 && (
        <div className="actions" style={{ marginTop: '16px' }}>
          <button type="button" className="button" disabled={starting} onClick={handleDownload}>
            {starting ? 'Iniciando…' : `Baixar ${missingCount} modelo${missingCount > 1 ? 's' : ''} automaticamente`}
          </button>
        </div>
      )}

      {dlActive && (
        <p className="muted" style={{ marginTop: '12px', fontSize: '0.9rem' }}>
          Download em andamento… não feche a janela do worker Python.
        </p>
      )}

      {/* Guia de download manual */}
      <div className="guide-toggle" style={{ marginTop: '20px' }}>
        <button
          type="button"
          className="button-ghost"
          aria-expanded={guideOpen}
          onClick={() => setGuideOpen((v) => !v)}
        >
          {guideOpen ? '▲' : '▼'} Como baixar manualmente
        </button>

        {guideOpen && (
          <div className="guide" aria-live="polite">
            <p>
              Os modelos são arquivos <code>.onnx</code> que devem ficar na pasta <code>models\</code> dentro do
              projeto. Os nomes dos arquivos precisam ser exatos:
            </p>

            <table className="guide-table">
              <thead>
                <tr>
                  <th>Arquivo</th>
                  <th>Função</th>
                  <th>Tamanho aprox.</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>lama_fp32.onnx</code></td>
                  <td>Remove rasgos e riscos</td>
                  <td>~200 MB</td>
                </tr>
                <tr>
                  <td><code>ddcolor-tiny-fp16.onnx</code></td>
                  <td>Colorização de fotos P&amp;B</td>
                  <td>~60 MB</td>
                </tr>
                <tr>
                  <td><code>realesrgan_x4plus.onnx</code></td>
                  <td>Ampliação 4× com detalhe real</td>
                  <td>~65 MB</td>
                </tr>
              </tbody>
            </table>

            <p>Fontes no Hugging Face (abrir no navegador e clicar em "download"):</p>
            <ul>
              <li>
                <a href="https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx" target="_blank" rel="noreferrer">
                  lama_fp32.onnx
                </a>
              </li>
              <li>
                <a href="https://huggingface.co/edgetools/ddcolor/resolve/main/ddcolor-tiny-fp16.onnx" target="_blank" rel="noreferrer">
                  ddcolor-tiny-fp16.onnx
                </a>
              </li>
              <li>
                <a href="https://huggingface.co/Qualcomm/Real-ESRGAN-x4plus/resolve/main/Real-ESRGAN-x4plus.onnx" target="_blank" rel="noreferrer">
                  realesrgan_x4plus.onnx
                </a>
              </li>
            </ul>

            <p>Depois de baixar, copie os arquivos para:</p>
            <pre><code>C:\restore-photo-onnx\models\</code></pre>
            <p className="muted" style={{ fontSize: '0.85rem' }}>
              O sistema carrega os modelos ao iniciar o worker Python. Reinicie o worker após copiar os arquivos.
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
