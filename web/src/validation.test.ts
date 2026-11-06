import { describe, expect, it } from 'vitest'
import { MAX_UPLOAD_BYTES, validateFile } from './validation'

const f = (name: string, type: string, size = 1000) => ({ name, type, size })

describe('validateFile', () => {
  it('aceita os formatos suportados pelo tipo MIME', () => {
    expect(validateFile(f('a.png', 'image/png'))).toBeNull()
    expect(validateFile(f('a.jpg', 'image/jpeg'))).toBeNull()
    expect(validateFile(f('a.webp', 'image/webp'))).toBeNull()
  })

  it('aceita TIFF/BMP sem tipo MIME quando a extensão é conhecida', () => {
    expect(validateFile(f('scan.TIFF', ''))).toBeNull()
    expect(validateFile(f('scan.bmp', ''))).toBeNull()
  })

  it('rejeita outros formatos e arquivos vazios', () => {
    expect(validateFile(f('doc.pdf', 'application/pdf'))).toMatch(/Formato não suportado/)
    expect(validateFile(f('nota.txt', ''))).toMatch(/Formato não suportado/)
    expect(validateFile(f('a.png', 'image/png', 0))).toMatch(/vazio/)
  })

  it('rejeita arquivo acima do limite', () => {
    expect(validateFile(f('a.png', 'image/png', MAX_UPLOAD_BYTES + 1))).toMatch(/25 MB/)
    expect(validateFile(f('a.png', 'image/png', MAX_UPLOAD_BYTES))).toBeNull()
  })
})
