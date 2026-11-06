import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { Dropzone } from './Dropzone'

describe('Dropzone', () => {
  it('mostra erro e não envia arquivo de formato inválido', () => {
    const onFile = vi.fn()
    const { container } = render(<Dropzone file={null} onFile={onFile} />)
    const input = container.querySelector('input[type="file"]')!
    fireEvent.change(input, { target: { files: [new File(['x'], 'nota.txt', { type: 'text/plain' })] } })

    expect(screen.getByRole('alert')).toHaveTextContent('Formato não suportado')
    expect(onFile).not.toHaveBeenCalled()
  })

  it('envia a foto válida recebida por arrastar e soltar', () => {
    const onFile = vi.fn()
    render(<Dropzone file={null} onFile={onFile} />)
    const png = new File(['png'], 'avo.png', { type: 'image/png' })
    fireEvent.drop(screen.getByText('Arraste a foto para cá'), { dataTransfer: { files: [png] } })

    expect(onFile).toHaveBeenCalledWith(png)
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('desabilita o botão enquanto ocupado', () => {
    render(<Dropzone file={null} onFile={vi.fn()} disabled />)
    expect(screen.getByRole('button', { name: 'Escolher arquivo' })).toBeDisabled()
  })
})
