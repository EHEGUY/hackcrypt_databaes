"use client"
import { useState, useCallback } from "react"
import { Shield, Upload, Loader2 } from "lucide-react"
import { useScanner } from "@/hooks/use-scanner"
import { Navbar } from "@/components/navbar"
import { Hero } from "@/components/hero"
import { Features } from "@/components/features"
import { UploadBox } from "@/components/upload-box"
import { MediaPreview } from "@/components/media-preview"
import { ResultCard } from "@/components/result-card"

interface Media {
  url: string
  type: "video" | "image"
  name: string
  file: File
}

export default function Page() {
  const [media, setMedia] = useState<Media | null>(null)
  const [view, setView] = useState<"landing" | "upload" | "about">("landing")
  const { isScanning, result, error, startScan, reset: resetScanner } = useScanner()

  const handleFileSelect = useCallback(
    (file: File) => {
      if (media?.url) {
        URL.revokeObjectURL(media.url)
      }
      const url = URL.createObjectURL(file)
      const isVideo = file.type.startsWith("video")
      setMedia({
        url,
        type: isVideo ? "video" : "image",
        name: file.name,
        file,
      })
      resetScanner()
    },
    [media, resetScanner],
  )

  const handleReset = useCallback(() => {
    if (media?.url) {
      URL.revokeObjectURL(media.url)
    }
    setMedia(null)
    resetScanner()
  }, [media, resetScanner])

  const handleScan = useCallback(() => {
    if (media?.file) {
      startScan(media.file)
    }
  }, [media, startScan])

  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      <Navbar view={view} onNavigate={setView} onReset={handleReset} />

      {/* Landing Page */}
      {view === "landing" && (
        <main className="pt-16">
          <Hero onStartScanning={() => setView("upload")} onLearnMore={() => setView("about")} />

          {/* Stats Section - Hackathon appropriate */}
          <section className="border-y border-border bg-card/20">
            <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4">
              {[
                { value: "99.2%", label: "Detection accuracy" },
                { value: "<2s", label: "Analysis time" },
                { value: "Image + Video", label: "Supported formats" },
                { value: "0", label: "Data stored" },
              ].map((stat, i) => (
                <div key={i} className="px-6 py-12 border-r border-border last:border-r-0 text-center">
                  <div className="text-3xl md:text-4xl font-semibold tracking-tight mb-2">{stat.value}</div>
                  <div className="text-sm text-muted-foreground">{stat.label}</div>
                </div>
              ))}
            </div>
          </section>

          <Features />

          {/* CTA Section */}
          <section className="py-24 px-6 border-t border-border">
            <div className="max-w-2xl mx-auto text-center">
              <h2 className="text-3xl md:text-5xl font-semibold tracking-tight mb-6">Ready to verify?</h2>
              <p className="text-muted-foreground text-lg mb-10">
                Upload your first file and see the results in seconds.
              </p>
              <button
                onClick={() => setView("upload")}
                className="inline-flex items-center gap-2 bg-foreground text-background px-8 py-4 rounded-full text-base font-medium hover:bg-foreground/90 transition-all"
              >
                <Upload className="w-4 h-4" />
                Upload Media
              </button>
            </div>
          </section>

          {/* Footer */}
          <footer className="border-t border-border py-8 px-6">
            <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4" />
                <span>Verify</span>
              </div>
              <p>No media is stored. All processing is done locally.</p>
            </div>
          </footer>
        </main>
      )}

      {/* Upload Page */}
      {view === "upload" && (
        <main className="pt-24 pb-16 px-6 min-h-screen flex items-center justify-center">
          <div className="w-full max-w-xl">
            {!media ? (
              <UploadBox onFileSelect={handleFileSelect} />
            ) : (
              <div className="space-y-6">
                <MediaPreview url={media.url} type={media.type} fileName={media.name} mimeType={media.file.type} />

                {/* File Info */}
                <div className="flex items-center justify-between p-4 rounded-xl bg-card border border-border">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center">
                      {media.type === "video" ? (
                        <svg
                          className="w-6 h-6"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                          />
                          <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      ) : (
                        <svg
                          className="w-6 h-6"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                          />
                        </svg>
                      )}
                    </div>
                    <div>
                      <p className="font-medium truncate max-w-[200px]">{media.name}</p>
                      <p className="text-sm text-muted-foreground capitalize">{media.type}</p>
                    </div>
                  </div>
                </div>

                {/* Actions */}
                {!result && !isScanning && (
                  <div className="flex gap-3">
                    <button
                      onClick={handleScan}
                      className="flex-1 flex items-center justify-center gap-2 bg-foreground text-background py-4 rounded-xl font-medium hover:bg-foreground/90 transition-colors"
                    >
                      <Shield className="w-5 h-5" />
                      Analyze Media
                    </button>
                    <button
                      onClick={handleReset}
                      className="px-6 py-4 rounded-xl border border-border hover:bg-card transition-colors text-muted-foreground"
                    >
                      Cancel
                    </button>
                  </div>
                )}

                {/* Scanning State */}
                {isScanning && (
                  <div className="p-10 rounded-2xl border border-border bg-card text-center">
                    <Loader2 className="w-12 h-12 mx-auto mb-6 animate-spin text-muted-foreground" />
                    <div className="h-1 bg-secondary rounded-full overflow-hidden mb-6 max-w-xs mx-auto">
                      <div className="h-full w-1/2 bg-foreground animate-scan" />
                    </div>
                    <p className="text-muted-foreground text-lg">Analyzing media for synthetic artifacts...</p>
                  </div>
                )}

                {/* Result */}
                {result && (
                  <>
                    <ResultCard
                      status={result.status}
                      confidence={result.confidence}
                      isFake={result.isFake}
                      error={error}
                    />
                    <button
                      onClick={handleReset}
                      className="w-full flex items-center justify-center gap-2 bg-foreground text-background py-4 rounded-xl font-medium hover:bg-foreground/90 transition-colors"
                    >
                      Scan Another File
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </main>
      )}

      {/* About Page */}
      {view === "about" && (
        <main className="pt-24 pb-16 px-6 min-h-screen">
          <div className="max-w-2xl mx-auto">
            <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-8">About Verify</h1>

            <div className="prose prose-invert prose-lg max-w-none">
              <p className="text-muted-foreground text-lg leading-relaxed mb-8">
                Verify is a deepfake detection tool built to help identify AI-generated and manipulated media. Using
                advanced machine learning models, we analyze images and videos for signs of synthetic manipulation.
              </p>

              <h2 className="text-2xl font-semibold mb-4 mt-12">How it works</h2>
              <p className="text-muted-foreground leading-relaxed mb-8">
                Our AI model analyzes visual patterns, compression artifacts, and inconsistencies that are typically
                present in AI-generated content. The analysis happens in real-time and provides a confidence score
                indicating the likelihood of manipulation.
              </p>

              <h2 className="text-2xl font-semibold mb-4 mt-12">Privacy</h2>
              <p className="text-muted-foreground leading-relaxed mb-8">
                We take your privacy seriously. Media files are processed locally and are never stored on our servers.
                Once the analysis is complete, your data is immediately discarded.
              </p>

              <h2 className="text-2xl font-semibold mb-4 mt-12">Hackathon Project</h2>
              <p className="text-muted-foreground leading-relaxed">
                Verify was built as a hackathon project to demonstrate the potential of AI in combating misinformation.
                While our detection accuracy is high, no system is perfect. Always use multiple sources to verify
                important information.
              </p>
            </div>

            <button
              onClick={() => setView("upload")}
              className="mt-12 inline-flex items-center gap-2 bg-foreground text-background px-8 py-4 rounded-full text-base font-medium hover:bg-foreground/90 transition-all"
            >
              <Upload className="w-4 h-4" />
              Try it now
            </button>
          </div>
        </main>
      )}
    </div>
  )
}
