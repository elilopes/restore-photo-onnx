export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024

const ACCEPTED_MIME = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/bmp']
const ACCEPTED_EXT = ['jpg', 'jpeg', 'png', 'webp', 'tif', 'tiff', 'bmp']

/** Validação rápida no navegador. A API confere os bytes do arquivo de novo. */
export function validateFile(file: Pick<File, 'name' | 'type' | 'size'>): string | null {
  const ext = file.name.split('.').pop()?.toLowerCase() ?? ''
  const typeOk = ACCEPTED_MIME.includes(file.type)
  // Alguns sistemas deixam o tipo vazio para TIFF/BMP: aí vale a extensão.
  const extOk = file.type === '' && ACCEPTED_EXT.includes(ext)
  if (!typeOk && !extOk) return 'Formato não suportado. Use JPEG, PNG, WebP, TIFF ou BMP.'
  if (file.size === 0) return 'O arquivo está vazio.'
  if (file.size > MAX_UPLOAD_BYTES) return 'A foto passa de 25 MB. Reduza o tamanho e tente de novo.'
  return null
}
