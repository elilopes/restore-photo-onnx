import { useEffect, useState } from 'react'

interface Props {
  originalUrl: string
  resultUrl: string
}

/**
 * Comparador antes/depois. Ao abrir, a divisória sai da borda direita e para
 * no meio: é o único movimento automático da tela. Depois disso, o controle
 * responde só ao usuário.
 */
export function Compare({ originalUrl, resultUrl }: Props) {
  const [split, setSplit] = useState(100) // % da foto original visível (da esquerda)
  const [settling, setSettling] = useState(true)

  useEffect(() => {
    const open = window.setTimeout(() => setSplit(50), 80)
    const done = window.setTimeout(() => setSettling(false), 1100)
    return () => {
      window.clearTimeout(open)
      window.clearTimeout(done)
    }
  }, [])

  return (
    <figure className="compare">
      <div className={`compare-stage${settling ? ' is-settling' : ''}`}>
        <img className="compare-result" src={resultUrl} alt="Foto restaurada" />
        <img
          className="compare-original"
          src={originalUrl}
          alt="Foto original"
          style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
        />
        <div className="compare-handle" style={{ left: `${split}%` }} aria-hidden="true" />
      </div>
      <figcaption>
        <span>Original</span>
        <input
          type="range"
          min={0}
          max={100}
          value={split}
          aria-label="Arraste para comparar a foto original com a restaurada"
          onChange={(e) => {
            setSettling(false)
            setSplit(Number(e.target.value))
          }}
        />
        <span>Restaurada</span>
      </figcaption>
    </figure>
  )
}
