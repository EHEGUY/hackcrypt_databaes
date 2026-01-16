"use client"

import { ArrowRight } from "lucide-react"

interface HeroProps {
  onStartScanning: () => void
  onLearnMore: () => void
}

export function Hero({ onStartScanning, onLearnMore }: HeroProps) {
  return (
    <section className="min-h-[90vh] flex flex-col items-center justify-center px-6 relative overflow-hidden">
      {/* Subtle grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.015)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.015)_1px,transparent_1px)] bg-[size:80px_80px]" />

      {/* Gradient orbs */}
      <div className="absolute top-1/4 -left-32 w-64 h-64 bg-success/5 rounded-full blur-3xl animate-float" />
      <div
        className="absolute bottom-1/4 -right-32 w-64 h-64 bg-destructive/5 rounded-full blur-3xl animate-float"
        style={{ animationDelay: "3s" }}
      />

      <div className="relative z-10 max-w-4xl mx-auto text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-card/50 text-sm text-muted-foreground mb-10">
          <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
          AI-Powered Detection
        </div>

        <h1 className="text-5xl md:text-7xl lg:text-8xl font-semibold tracking-tighter mb-8 text-balance">
          Detect deepfakes
          <br />
          <span className="text-muted-foreground">before they spread</span>
        </h1>

        <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-12 text-pretty leading-relaxed">
          Upload any image or video and our AI will analyze it for synthetic manipulation. Get instant results with
          confidence scores.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            onClick={onStartScanning}
            className="flex items-center gap-2 bg-foreground text-background px-8 py-4 rounded-full text-base font-medium hover:bg-foreground/90 transition-all"
          >
            Start Scanning
            <ArrowRight className="w-4 h-4" />
          </button>
          <button
            onClick={onLearnMore}
            className="flex items-center gap-2 px-8 py-4 rounded-full text-base font-medium border border-border hover:bg-card transition-colors"
          >
            Learn More
          </button>
        </div>
      </div>
    </section>
  )
}
