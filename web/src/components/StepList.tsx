import { describeEngine, STATUS_LABELS, stepLabel } from '../defaults'
import type { StepReport } from '../types'

export function StepList({ steps }: { steps: StepReport[] }) {
  return (
    <ol className="steps">
      {steps.map((s) => {
        const engine = describeEngine(s.engine)
        // O detalhe da leitura é técnico demais para o usuário; o das outras etapas é útil.
        const detail = s.name === 'load' ? null : s.detail
        return (
          <li key={s.name} className={`step is-${s.status}`}>
            <div className="step-head">
              <span>{stepLabel(s.name)}</span>
              <span className="step-status">{STATUS_LABELS[s.status]}</span>
            </div>
            {(engine || detail) && (
              <p className="muted step-detail">
                {[engine, detail].filter(Boolean).join(' · ')}
              </p>
            )}
          </li>
        )
      })}
    </ol>
  )
}
