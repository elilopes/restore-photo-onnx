import { useId, type ReactNode } from 'react'
import type { Colorize, FaceRestoreModel, OutputFormat, RestoreOptions, StyleName, Upscale } from '../types'

interface Props {
  value: RestoreOptions
  onChange: (next: RestoreOptions) => void
  disabled?: boolean
}

export function OptionsPanel({ value, onChange, disabled }: Props) {
  const set = <K extends keyof RestoreOptions>(key: K, v: RestoreOptions[K]) => onChange({ ...value, [key]: v })

  return (
    <fieldset className="options" disabled={disabled}>
      <legend className="visually-hidden">Opções de restauração</legend>

      <Section title="Reparo">
        <Switch
          label="Remover rasgos e riscos"
          checked={value.repair}
          onChange={(v) => set('repair', v)}
        />
        {value.repair && (
          <Slider
            label="Sensibilidade"
            hint="Mais alto encontra riscos mais finos, mas pode marcar detalhes da foto."
            value={value.repairSensitivity}
            onChange={(v) => set('repairSensitivity', v)}
          />
        )}
      </Section>

      <Section title="Cor">
        <Choice<Colorize>
          legend="Colorir"
          value={value.colorize}
          onChange={(v) => set('colorize', v)}
          options={[
            { value: 'auto', label: 'Só se for preto e branco' },
            { value: 'on', label: 'Sempre' },
            { value: 'off', label: 'Nunca' },
          ]}
        />
        <Switch
          label="Realçar cores desbotadas"
          checked={value.enhanceColors}
          onChange={(v) => set('enhanceColors', v)}
        />
        {value.enhanceColors && (
          <Slider label="Intensidade" value={value.enhanceStrength} onChange={(v) => set('enhanceStrength', v)} />
        )}
      </Section>

      <Section title="Nitidez">
        <Switch label="Aumentar a nitidez" checked={value.sharpen} onChange={(v) => set('sharpen', v)} />
        {value.sharpen && (
          <Slider label="Intensidade" value={value.sharpenStrength} onChange={(v) => set('sharpenStrength', v)} />
        )}
      </Section>

      <Section title="Pré-processamento">
        <Switch
          label="Realçar iluminação (SCI)"
          hint="Melhora fotos muito escuras antes das demais etapas."
          checked={value.lowLight}
          onChange={(v) => set('lowLight', v)}
        />
      </Section>

      <Section title="Restauração Facial">
        <Switch
          label="Restaurar rostos"
          hint="Use em fotos de família ou retratos com rostos danificados."
          checked={value.faceRestore}
          onChange={(v) => set('faceRestore', v)}
        />
        {value.faceRestore && (
          <>
            <div className="warning-banner" role="note">
              ⚠ Esse modelo é mais demorado
            </div>
            <Choice<FaceRestoreModel>
              legend="Modelo de restauração"
              value={value.faceRestoreModel}
              onChange={(v) => set('faceRestoreModel', v)}
              options={[
                { value: 'gfpgan', label: 'GFPGAN', hint: 'Rápido, bons resultados gerais' },
                { value: 'codeformer', label: 'CodeFormer', hint: 'Mais detalhado, especialmente em rostos antigos' },
              ]}
            />
          </>
        )}
      </Section>

      <Section title="Estilo Artístico">
        <Switch
          label="Aplicar estilo artístico"
          hint="Opção criativa — não é restauração. Altera o visual da foto."
          checked={value.styleTransfer}
          onChange={(v) => set('styleTransfer', v)}
        />
        {value.styleTransfer && (
          <>
            <Choice<StyleName>
              legend="Estilo"
              value={value.styleName}
              onChange={(v) => set('styleName', v)}
              options={[
                { value: 'candy', label: 'Candy', hint: 'Cores vibrantes' },
                { value: 'mosaic', label: 'Mosaico', hint: 'Estilo cubista' },
                { value: 'rain_princess', label: 'Rain Princess', hint: 'Impressionista' },
                { value: 'udnie', label: 'Udnie', hint: 'Abstrato moderno' },
              ]}
            />
            <Slider
              label="Intensidade do estilo"
              hint="Mais alto = estilo mais forte, menos foto original."
              value={value.styleStrength}
              onChange={(v) => set('styleStrength', v)}
            />
          </>
        )}
      </Section>

      <Section title="Saída">
        <Choice<Upscale>
          legend="Tamanho"
          value={value.upscale}
          onChange={(v) => set('upscale', v)}
          options={[
            { value: 1, label: 'Mesmo tamanho' },
            { value: 2, label: 'Dobro', hint: 'Mais lento' },
            { value: 4, label: 'Quádruplo', hint: 'Bem mais lento e arquivo grande' },
          ]}
        />
        <Choice<OutputFormat>
          legend="Formato"
          value={value.outputFormat}
          onChange={(v) => set('outputFormat', v)}
          options={[
            { value: 'png', label: 'PNG', hint: 'Sem perda de qualidade' },
            { value: 'jpg', label: 'JPG', hint: 'Arquivo menor, com leve perda' },
          ]}
        />
        <Switch
          label="Usar modelos de IA"
          hint="Desligue para usar só o processamento clássico. Sem os modelos, a restauração é mais simples."
          checked={value.useModels}
          onChange={(v) => set('useModels', v)}
        />
      </Section>
    </fieldset>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="options-section">
      <h2>{title}</h2>
      {children}
    </section>
  )
}

function Switch({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint?: string
  checked: boolean
  onChange: (v: boolean) => void
}) {
  const id = useId()
  const hintId = useId()
  return (
    <div className="field">
      <label className="switch" htmlFor={id}>
        <input
          id={id}
          type="checkbox"
          role="switch"
          checked={checked}
          aria-describedby={hint ? hintId : undefined}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="switch-track" aria-hidden="true" />
        <span>{label}</span>
      </label>
      {hint && (
        <p id={hintId} className="hint">
          {hint}
        </p>
      )}
    </div>
  )
}

function Slider({
  label,
  hint,
  value,
  onChange,
}: {
  label: string
  hint?: string
  value: number
  onChange: (v: number) => void
}) {
  const id = useId()
  const hintId = useId()
  const percent = Math.round(value * 100)
  return (
    <div className="field field-slider">
      <label htmlFor={id}>
        {label} <output htmlFor={id}>{percent}%</output>
      </label>
      <input
        id={id}
        type="range"
        min={0}
        max={100}
        step={5}
        value={percent}
        aria-describedby={hint ? hintId : undefined}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
      />
      {hint && (
        <p id={hintId} className="hint">
          {hint}
        </p>
      )}
    </div>
  )
}

interface Option<T> {
  value: T
  label: string
  hint?: string
}

function Choice<T extends string | number>({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string
  options: Option<T>[]
  value: T
  onChange: (v: T) => void
}) {
  const name = useId()
  return (
    <fieldset className="choice">
      <legend>{legend}</legend>
      {options.map((o) => (
        <label key={String(o.value)} className="choice-item">
          <input
            type="radio"
            name={name}
            value={String(o.value)}
            checked={value === o.value}
            onChange={() => onChange(o.value)}
          />
          <span>
            <span className="choice-label">{o.label}</span>
            {o.hint && <small className="hint">{o.hint}</small>}
          </span>
        </label>
      ))}
    </fieldset>
  )
}
