"use client"

import { useState, useCallback } from "react"

interface ScanResult {
  status: string
  confidence: string
  isFake: boolean
}

interface UseScannerReturn {
  isScanning: boolean
  result: ScanResult | null
  error: string | null
  startScan: (file: File) => Promise<void>
  reset: () => void
}

export function useScanner(): UseScannerReturn {
  const [isScanning, setIsScanning] = useState(false)
  const [result, setResult] = useState<ScanResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const startScan = useCallback(async (file: File) => {
    setIsScanning(true)
    setResult(null)
    setError(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`)
      }

      const data = await res.json()

      setResult({
        status: data.is_fake ? "Fake" : "Real",
        confidence: (data.confidence * 100).toFixed(1),
        isFake: data.is_fake,
      })
    } catch (e) {
      console.error("[Scanner] Error:", e)
      // Demo fallback when backend is not available
      setError("Backend not connected - showing demo result")
      const isFake = Math.random() > 0.5
      setResult({
        status: isFake ? "Fake" : "Real",
        confidence: (85 + Math.random() * 14).toFixed(1),
        isFake: isFake,
      })
    } finally {
      setIsScanning(false)
    }
  }, [])

  const reset = useCallback(() => {
    setIsScanning(false)
    setResult(null)
    setError(null)
  }, [])

  return { isScanning, result, error, startScan, reset }
}
