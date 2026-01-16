"use client"

import { Eye, Lock, Zap } from "lucide-react"

const features = [
  {
    icon: Eye,
    title: "Deep Analysis",
    description: "Our AI examines pixel patterns, compression artifacts, and temporal inconsistencies in videos.",
  },
  {
    icon: Lock,
    title: "Privacy First",
    description: "All processing happens locally. Your media never leaves your device or touches our servers.",
  },
  {
    icon: Zap,
    title: "Instant Results",
    description: "Get detailed analysis in under 2 seconds with confidence scores and explanations.",
  },
]

export function Features() {
  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-semibold tracking-tight mb-4">Built for trust</h2>
          <p className="text-muted-foreground text-lg max-w-xl mx-auto">
            Powerful detection technology that respects your privacy
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {features.map((feature, i) => (
            <div key={i} className="p-8 rounded-2xl border border-border bg-card/30 hover:bg-card/50 transition-colors">
              <div className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center mb-6">
                <feature.icon className="w-7 h-7 text-foreground" strokeWidth={1.5} />
              </div>
              <h3 className="text-xl font-medium mb-3">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
