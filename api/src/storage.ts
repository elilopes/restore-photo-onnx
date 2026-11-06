import { createWriteStream } from 'node:fs'
import { mkdir, open, readdir, rename, rm, stat } from 'node:fs/promises'
import path from 'node:path'
import type { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { JOB_ID_RE } from './schemas.js'

export interface ImageType {
  ext: 'jpg' | 'png' | 'webp' | 'tif' | 'bmp'
  mime: string
}

const RESULT_FILE_RE = /^(result\.(png|jpg)|mask\.png)$/

/** Identifica o formato pelos bytes iniciais (não confia na extensão nem no Content-Type do cliente). */
export function sniffImage(head: Buffer): ImageType | null {
  const startsWith = (...bytes: number[]) => bytes.every((b, i) => head[i] === b)
  if (startsWith(0xff, 0xd8, 0xff)) return { ext: 'jpg', mime: 'image/jpeg' }
  if (startsWith(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)) return { ext: 'png', mime: 'image/png' }
  if (head.length >= 12 && head.toString('latin1', 0, 4) === 'RIFF' && head.toString('latin1', 8, 12) === 'WEBP') {
    return { ext: 'webp', mime: 'image/webp' }
  }
  if (startsWith(0x49, 0x49, 0x2a, 0x00) || startsWith(0x4d, 0x4d, 0x00, 0x2a)) return { ext: 'tif', mime: 'image/tiff' }
  if (startsWith(0x42, 0x4d)) return { ext: 'bmp', mime: 'image/bmp' }
  return null
}

const MIME_BY_EXT: Record<string, string> = {
  jpg: 'image/jpeg', png: 'image/png', webp: 'image/webp', tif: 'image/tiff', bmp: 'image/bmp',
}

export function mimeFromName(name: string): string {
  return MIME_BY_EXT[path.extname(name).slice(1).toLowerCase()] ?? 'application/octet-stream'
}

/**
 * Armazenamento em disco local (volume compartilhado com o worker Python).
 *   uploads/<jobId>.<ext>        entrada
 *   results/<jobId>/result.png   saída (escrita pelo Python)
 *   results/<jobId>/mask.png     máscara de defeitos (debug/visualização)
 * Trocar por S3/MinIO = reimplementar esta classe e expor URLs assinadas.
 */
export class LocalStorage {
  readonly uploadsDir: string
  readonly resultsDir: string

  constructor(readonly root: string) {
    this.uploadsDir = path.join(root, 'uploads')
    this.resultsDir = path.join(root, 'results')
  }

  async init(): Promise<void> {
    await mkdir(this.uploadsDir, { recursive: true })
    await mkdir(this.resultsDir, { recursive: true })
  }

  uploadPath(name: string): string {
    if (path.basename(name) !== name) throw new Error('nome de upload inválido')
    return path.join(this.uploadsDir, name)
  }

  /** Caminho de um arquivo de resultado; só aceita nomes da lista permitida. */
  resultPath(jobId: string, name: string): string {
    if (!JOB_ID_RE.test(jobId) || !RESULT_FILE_RE.test(name)) throw new Error('arquivo de resultado inválido')
    return path.join(this.resultsDir, jobId, name)
  }

  /** Grava o upload em disco e valida os bytes iniciais. Devolve null se não for uma imagem suportada. */
  async saveUpload(jobId: string, stream: Readable): Promise<{ name: string; mime: string } | null> {
    if (!JOB_ID_RE.test(jobId)) throw new Error('jobId inválido')
    const tmp = path.join(this.uploadsDir, `${jobId}.part`)
    try {
      await pipeline(stream, createWriteStream(tmp, { flags: 'wx' }))
      const fh = await open(tmp, 'r')
      const head = Buffer.alloc(16)
      try {
        await fh.read(head, 0, 16, 0)
      } finally {
        await fh.close()
      }
      const type = sniffImage(head)
      if (!type) {
        await rm(tmp, { force: true })
        return null
      }
      const name = `${jobId}.${type.ext}`
      await rename(tmp, path.join(this.uploadsDir, name))
      return { name, mime: type.mime }
    } catch (err) {
      await rm(tmp, { force: true })
      throw err
    }
  }

  async removeJob(jobId: string): Promise<void> {
    if (!JOB_ID_RE.test(jobId)) return
    const names = await readdir(this.uploadsDir).catch(() => [] as string[])
    await Promise.all(names.filter((n) => n.startsWith(`${jobId}.`)).map((n) => rm(path.join(this.uploadsDir, n), { force: true })))
    await rm(path.join(this.resultsDir, jobId), { recursive: true, force: true })
  }

  /** Apaga uploads e resultados mais antigos que `maxAgeMs`. Retorna quantos itens foram removidos. */
  async purgeOlderThan(maxAgeMs: number, now = Date.now()): Promise<number> {
    let removed = 0
    for (const dir of [this.uploadsDir, this.resultsDir]) {
      for (const name of await readdir(dir).catch(() => [] as string[])) {
        const full = path.join(dir, name)
        const info = await stat(full).catch(() => null)
        if (info && now - info.mtimeMs > maxAgeMs) {
          await rm(full, { recursive: true, force: true })
          removed++
        }
      }
    }
    return removed
  }
}
