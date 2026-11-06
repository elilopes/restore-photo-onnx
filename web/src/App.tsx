import { useState } from 'react'
import { ApiError, createJob, fileUrl } from './api'
import { Compare } from './components/Compare'
import { Dropzone } from './components/Dropzone'
import { ModelManager } from './components/ModelManager'
import { OptionsPanel } from './components/OptionsPanel'
import { StepList } from './components/StepList'
import { DEFAULT_OPTIONS, stepLabel } from './defaults'
import { useJob } from './useJob'
import type { FaceAnalysis, RestoreOptions } from './types'

interface AppProps {
  /** Intervalo de consulta do progresso (ms). Exposto para os testes. */
  pollMs?: number
}

export default function App({ pollMs = 1000 }: AppProps) {
  const [file, setFile] = useState<File | null>(null)
  const [options, setOptions] = useState<RestoreOptions>(DEFAULT_OPTIONS)
  const [jobId, setJobId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [faceAnalysis, setFaceAnalysis] = useState<FaceAnalysis | null>(null)
  const [faceHintDismissed, setFaceHintDismissed] = useState(false)
  const { job, error: pollError } = useJob(jobId, pollMs)

  const running = submitting || job?.status === 'queued' || job?.status === 'processing'
  const done = job?.status === 'done' && job.result ? job : null

  async function start() {
    if (!file) return
    setSubmitting(true)
    setSubmitError(null)
    setJobId(null)
    setFaceAnalysis(null)
    setFaceHintDismissed(false)
    try {
      const { id } = await createJob(file, options)
      setJobId(id)
      // Analisar rostos em background após envio
      analyzeJobFaces(id)
    } catch (err) {
      setSubmitError(err instanceof ApiError || err instanceof Error ? err.message : 'Falha ao enviar.')
    } finally {
      setSubmitting(false)
    }
  }

  async function analyzeJobFaces(id: string) {
    try {
      const res = await fetch(`/api/jobs/${encodeURIComponent(id)}/analyze`)
      if (!res.ok) return
      const data = await res.json() as { faceCount: number; faceCoverage: number; suggestFaceRestore: boolean }
      setFaceAnalysis({ faceCount: data.faceCount, faceCoverage: data.faceCoverage, suggestFaceRestore: data.suggestFaceRestore })
    } catch {
      // Silencioso — análise é opcional
    }
  }

  function newPhoto() {
    setJobId(null)
    setFile(null)
    setSubmitError(null)
    setFaceAnalysis(null)
    setFaceHintDismissed(false)
  }

  const progress = job?.progress
  const failure = job?.status === 'failed' ? job.error ?? 'Erro desconhecido.' : null
  const problem = submitError ?? failure ?? (pollError && !job ? pollError : null)

  return (
    <main className="page">
      <header className="masthead">
        <h1>Devolva a cor e a nitidez à sua foto antiga</h1>
        <p className="lede">
          Remova rasgos e riscos, recupere cores desbotadas, colorize fotos em preto e branco e
          aumente a nitidez. O processamento acontece no servidor, e sua foto original não é alterada.
        </p>
      </header>

      <div className="workbench">
        <section className="print" aria-labelledby="photo-heading">
          <h2 id="photo-heading" className="visually-hidden">
            Foto
          </h2>

          <Dropzone file={file} onFile={(f) => { setFile(f); setJobId(null) }} disabled={running} />

          {faceAnalysis?.suggestFaceRestore && !options.faceRestore && !faceHintDismissed && (
            <div className="face-hint" role="note">
              <span>
                {faceAnalysis.faceCount > 2
                  ? `${faceAnalysis.faceCount} rostos detectados`
                  : 'Rosto(s) detectado(s)'}
                {' '}— ative a <strong>Restauração Facial</strong> para melhor resultado.
                <br /><small>⚠ Esse modelo é mais demorado</small>
              </span>
              <div className="face-hint-actions">
                <button
                  type="button"
                  className="button"
                  style={{ fontSize: '0.875rem', padding: '6px 14px' }}
                  onClick={() => setOptions(o => ({ ...o, faceRestore: true }))}
                >
                  Ativar
                </button>
                <button
                  type="button"
                  className="button-ghost"
                  style={{ fontSize: '0.875rem', padding: '6px 14px' }}
                  onClick={() => setFaceHintDismissed(true)}
                >
                  Ignorar
                </button>
              </div>
            </div>
          )}

          {running && (
            <div className="progress" role="status">
              <progress max={100} value={progress?.percent ?? 0} aria-label="Progresso da restauração" />
              <p>
                {stepLabel(progress?.step ?? 'start')}
                <span className="muted"> · {progress?.percent ?? 0}%</span>
              </p>
            </div>
          )}

          {problem && (
            <p role="alert" className="error">
              {problem}
            </p>
          )}

          {done?.result && (
            <>
              <Compare originalUrl={fileUrl(done.id, 'original')} resultUrl={fileUrl(done.id, 'result')} />
              <div className="actions">
                <a className="button" href={fileUrl(done.id, 'result', true)} download>
                  Baixar restaurada
                </a>
                {done.result.urls.mask && (
                  <a className="button-ghost" href={fileUrl(done.id, 'mask', true)} download>
                    Baixar máscara de defeitos
                  </a>
                )}
                <button type="button" className="button-ghost" onClick={newPhoto}>
                  Nova foto
                </button>
              </div>
            </>
          )}

          {!done && !running && (
            <div className="actions">
              <button type="button" className="button" disabled={!file} onClick={start}>
                Restaurar foto
              </button>
            </div>
          )}
        </section>

        <aside className="controls" aria-label="Ajustes">
          <OptionsPanel value={options} onChange={setOptions} disabled={running} />
        </aside>
      </div>

      {done?.result && (
        <section className="report" aria-labelledby="report-heading">
          <h2 id="report-heading">O que foi feito</h2>
          <p className="muted">
            {done.result.inputWidth} × {done.result.inputHeight} → {done.result.width} × {done.result.height} px
            {done.result.wasMonochrome ? ' · foto em preto e branco' : ''} · {(done.result.totalMs / 1000).toFixed(1)} s
          </p>
          <StepList steps={done.result.steps} />
        </section>
      )}

      {/* Gerenciador de modelos: visível quando há modelos faltando ou download ativo */}
      <ModelManager hideWhenOk />
    </main>
  )
}
