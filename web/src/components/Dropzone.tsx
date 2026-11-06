import { useEffect, useId, useRef, useState, type DragEvent } from 'react'
import { validateFile } from '../validation'

interface Props {
  file: File | null
  onFile: (file: File) => void
  disabled?: boolean
}

export function Dropzone({ file, onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const errorId = useId()
  const [over, setOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function accept(candidate: File | undefined) {
    if (!candidate) return
    const problem = validateFile(candidate)
    setError(problem)
    if (!problem) onFile(candidate)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setOver(false)
    if (!disabled) accept(e.dataTransfer.files[0])
  }

  return (
    <div
      className={`dropzone${over ? ' is-over' : ''}${file ? ' has-file' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={handleDrop}
    >
      {previewUrl ? (
        <img className="dropzone-preview" src={previewUrl} alt={`Foto selecionada: ${file?.name}`} />
      ) : (
        <p className="dropzone-empty">
          Arraste a foto para cá
          <span className="muted">JPEG, PNG, WebP, TIFF ou BMP, até 25 MB</span>
        </p>
      )}

      <button
        type="button"
        className="button-ghost"
        disabled={disabled}
        aria-describedby={error ? errorId : undefined}
        onClick={() => inputRef.current?.click()}
      >
        {file ? 'Escolher outra foto' : 'Escolher arquivo'}
      </button>

      <input
        ref={inputRef}
        type="file"
        hidden
        accept=".jpg,.jpeg,.png,.webp,.tif,.tiff,.bmp,image/*"
        onChange={(e) => {
          accept(e.target.files?.[0])
          e.target.value = '' // permite escolher o mesmo arquivo de novo
        }}
      />

      {error && (
        <p id={errorId} role="alert" className="error">
          {error}
        </p>
      )}
    </div>
  )
}
