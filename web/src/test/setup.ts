import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// jsdom não implementa object URLs.
URL.createObjectURL = () => 'blob:mock'
URL.revokeObjectURL = () => {}
