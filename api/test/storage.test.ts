import { mkdtemp, readdir, stat, utimes, writeFile, mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import { beforeEach, describe, expect, it } from 'vitest'
import { LocalStorage, mimeFromName, sniffImage } from '../src/storage.js'

const JOB = '11111111-1111-4111-8111-111111111111'
const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13])

describe('sniffImage', () => {
  it('reconhece os formatos suportados pelos bytes iniciais', () => {
    expect(sniffImage(Buffer.from([0xff, 0xd8, 0xff, 0xe0]))?.ext).toBe('jpg')
    expect(sniffImage(PNG)?.ext).toBe('png')
    expect(sniffImage(Buffer.from('RIFF\0\0\0\0WEBPVP8 '))?.ext).toBe('webp')
    expect(sniffImage(Buffer.from([0x49, 0x49, 0x2a, 0x00]))?.ext).toBe('tif')
    expect(sniffImage(Buffer.from([0x42, 0x4d, 0, 0]))?.ext).toBe('bmp')
  })
  it('rejeita o que não é imagem (mesmo com extensão enganosa)', () => {
    expect(sniffImage(Buffer.from('<?php echo 1; ?>'))).toBeNull()
    expect(sniffImage(Buffer.from('%PDF-1.7'))).toBeNull()
    expect(sniffImage(Buffer.alloc(0))).toBeNull()
  })
  it('mimeFromName', () => {
    expect(mimeFromName('result.png')).toBe('image/png')
    expect(mimeFromName('x.JPG')).toBe('image/jpeg')
    expect(mimeFromName('x.bin')).toBe('application/octet-stream')
  })
})

describe('LocalStorage', () => {
  let storage: LocalStorage
  beforeEach(async () => {
    storage = new LocalStorage(await mkdtemp(path.join(tmpdir(), 'rp-')))
    await storage.init()
  })

  it('saveUpload grava com a extensão detectada e rejeita não-imagens sem deixar lixo', async () => {
    const ok = await storage.saveUpload(JOB, Readable.from([PNG]))
    expect(ok).toEqual({ name: `${JOB}.png`, mime: 'image/png' })
    expect(await readdir(storage.uploadsDir)).toEqual([`${JOB}.png`])

    const other = '22222222-2222-4222-8222-222222222222'
    expect(await storage.saveUpload(other, Readable.from([Buffer.from('nao sou imagem')]))).toBeNull()
    expect(await readdir(storage.uploadsDir)).toEqual([`${JOB}.png`])
  })

  it('bloqueia path traversal e nomes fora da lista', () => {
    expect(() => storage.uploadPath('../x.png')).toThrow()
    expect(() => storage.resultPath('../../etc', 'result.png')).toThrow()
    expect(() => storage.resultPath(JOB, '../../passwd')).toThrow()
    expect(() => storage.resultPath(JOB, 'segredo.txt')).toThrow()
    expect(storage.resultPath(JOB, 'result.png')).toBe(path.join(storage.resultsDir, JOB, 'result.png'))
    expect(storage.resultPath(JOB, 'mask.png')).toContain('mask.png')
  })

  it('removeJob apaga upload e pasta de resultados', async () => {
    await storage.saveUpload(JOB, Readable.from([PNG]))
    await mkdir(path.join(storage.resultsDir, JOB))
    await writeFile(path.join(storage.resultsDir, JOB, 'result.png'), PNG)
    await storage.removeJob(JOB)
    expect(await readdir(storage.uploadsDir)).toEqual([])
    expect(await readdir(storage.resultsDir)).toEqual([])
  })

  it('purgeOlderThan remove só o que passou da retenção', async () => {
    await storage.saveUpload(JOB, Readable.from([PNG]))
    const fresh = '33333333-3333-4333-8333-333333333333'
    await storage.saveUpload(fresh, Readable.from([PNG]))
    const old = new Date(Date.now() - 48 * 3_600_000)
    await utimes(path.join(storage.uploadsDir, `${JOB}.png`), old, old)

    expect(await storage.purgeOlderThan(24 * 3_600_000)).toBe(1)
    expect(await readdir(storage.uploadsDir)).toEqual([`${fresh}.png`])
    expect((await stat(storage.uploadsDir)).isDirectory()).toBe(true)
  })
})
