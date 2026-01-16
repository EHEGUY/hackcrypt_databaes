"use client"

import type React from "react"
import { useRef } from "react"
import { Upload } from "lucide-react"

interface UploadBoxProps {
  onFileSelect: (file: File) => void
}

export function UploadBox({ onFileSelect }: UploadBoxProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && (file.type.startsWith("video") || file.type.startsWith("image"))) {
      onFileSelect(file)
    }
  }

  return (
    <div className="text-center">
      <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-4">Upload Media</h1>
      <p className="text-muted-foreground mb-10 text-lg">Drop an image or video to analyze for deepfake manipulation</p>

      <div
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="border border-border rounded-2xl p-16 cursor-pointer hover:border-muted-foreground/50 hover:bg-card/30 transition-all group"
      >
        <div className="w-20 h-20 rounded-2xl bg-secondary flex items-center justify-center mx-auto mb-6 group-hover:scale-105 transition-transform">
          <Upload className="w-10 h-10 text-foreground" strokeWidth={1.5} />
        </div>
        <p className="text-foreground font-medium mb-2 text-lg">Click to upload or drag and drop</p>
        <p className="text-sm text-muted-foreground">MP4, MOV, AVI, JPG, PNG up to 100MB</p>
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,image/jpeg,image/png,image/webp"
          hidden
        />
      </div>
    </div>
  )
}
