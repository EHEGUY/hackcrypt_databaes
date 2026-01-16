"use client"

import { CheckCircle2, XCircle } from "lucide-react"

interface ResultCardProps {
  status: string
  confidence: string
  isFake: boolean
  error?: string | null
}

export function ResultCard({ status, confidence, isFake, error }: ResultCardProps) {
  return (
    <div className="space-y-4">
      {error && (
        <div className="p-4 rounded-xl bg-muted/50 border border-border text-center">
          <p className="text-sm text-muted-foreground">{error}</p>
        </div>
      )}

      <div
        className={`p-10 rounded-2xl border text-center ${
          isFake ? "border-destructive/40 bg-destructive/10" : "border-success/40 bg-success/10"
        }`}
      >
        <div className="mb-6">
          {isFake ? (
            <XCircle className="w-20 h-20 mx-auto text-destructive" strokeWidth={1.5} />
          ) : (
            <CheckCircle2 className="w-20 h-20 mx-auto text-success" strokeWidth={1.5} />
          )}
        </div>

        <h2 className={`text-3xl font-semibold mb-3 ${isFake ? "text-destructive" : "text-success"}`}>
          {isFake ? "Likely Synthetic" : "Likely Authentic"}
        </h2>

        <p className="text-muted-foreground mb-8 text-lg">
          {isFake ? "This media shows signs of AI manipulation" : "No signs of synthetic manipulation detected"}
        </p>

        <div className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-card border border-border">
          <span className="text-muted-foreground">Confidence</span>
          <span className="text-3xl font-semibold">{confidence}%</span>
        </div>
      </div>
    </div>
  )
}
