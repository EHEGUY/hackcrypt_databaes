"use client"

import { useRef, useEffect, useState } from "react"
import { XCircle, Play } from "lucide-react"

interface MediaPreviewProps {
  url: string
  type: "video" | "image"
  fileName: string
  mimeType: string
}

export function MediaPreview({ url, type, fileName, mimeType }: MediaPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState(false)
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    if (type === "video" && videoRef.current) {
      const video = videoRef.current
      setError(false)
      setIsLoaded(false)

      // Force reload the video
      video.load()
    }
  }, [url, type])

  if (type === "image") {
    return (
      <div className="rounded-2xl overflow-hidden border border-border bg-card">
        <img
          src={url || "/placeholder.svg"}
          alt="Preview"
          className="w-full h-auto max-h-[50vh] object-contain bg-black"
          onError={() => setError(true)}
        />
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-card">
            <div className="text-center p-6">
              <XCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
              <p className="text-muted-foreground">Image preview unavailable</p>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-2xl overflow-hidden border border-border bg-black relative">
      {!isLoaded && !error && (
        <div className="absolute inset-0 flex items-center justify-center bg-card z-10">
          <div className="text-center">
            <div className="w-16 h-16 rounded-full border-2 border-muted-foreground/30 border-t-foreground animate-spin mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">Loading video...</p>
          </div>
        </div>
      )}

      <video
        ref={videoRef}
        key={url}
        controls
        playsInline
        preload="auto"
        muted
        className="w-full h-auto max-h-[50vh] bg-black"
        onLoadedData={() => setIsLoaded(true)}
        onCanPlay={() => setIsLoaded(true)}
        onError={() => {
          setError(true)
          setIsLoaded(true)
        }}
      >
        <source src={url} type={mimeType || "video/mp4"} />
        <source src={url} type="video/mp4" />
        <source src={url} type="video/webm" />
        <source src={url} type="video/quicktime" />
      </video>

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-card">
          <div className="text-center p-6">
            <div className="w-20 h-20 rounded-2xl bg-secondary flex items-center justify-center mx-auto mb-4">
              <Play className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-foreground font-medium mb-1">Video loaded</p>
            <p className="text-sm text-muted-foreground">{fileName}</p>
            <p className="text-xs text-muted-foreground mt-2">Preview may not display - scan will still work</p>
          </div>
        </div>
      )}
    </div>
  )
}
