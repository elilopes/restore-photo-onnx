import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { DEFAULT_OPTIONS } from '../defaults'
import { OptionsPanel } from './OptionsPanel'

describe('OptionsPanel', () => {
  it('desligar o reparo emite a mudança e esconde a sensibilidade', () => {
    const onChange = vi.fn()
    const { rerender } = render(<OptionsPanel value={DEFAULT_OPTIONS} onChange={onChange} />)
    expect(screen.getByRole('slider', { name: /Sensibilidade/ })).toBeInTheDocument()

    fireEvent.click(screen.getByRole('switch', { name: 'Remover rasgos e riscos' }))
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_OPTIONS, repair: false })

    rerender(<OptionsPanel value={{ ...DEFAULT_OPTIONS, repair: false }} onChange={onChange} />)
    expect(screen.queryByRole('slider', { name: /Sensibilidade/ })).toBeNull()
  })

  it('escolher 2x em tamanho emite upscale como número', () => {
    const onChange = vi.fn()
    render(<OptionsPanel value={DEFAULT_OPTIONS} onChange={onChange} />)
    fireEvent.click(screen.getByRole('radio', { name: /Dobro/ }))
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_OPTIONS, upscale: 2 })
  })

  it('slider converte percentual para 0..1', () => {
    const onChange = vi.fn()
    render(<OptionsPanel value={DEFAULT_OPTIONS} onChange={onChange} />)
    fireEvent.change(screen.getByRole('slider', { name: /Sensibilidade/ }), { target: { value: '80' } })
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_OPTIONS, repairSensitivity: 0.8 })
  })

  it('fica desabilitado durante o processamento', () => {
    render(<OptionsPanel value={DEFAULT_OPTIONS} onChange={vi.fn()} disabled />)
    expect(screen.getByRole('switch', { name: 'Remover rasgos e riscos' })).toBeDisabled()
  })
})
