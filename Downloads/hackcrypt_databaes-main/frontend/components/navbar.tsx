"use client"

import { Shield } from "lucide-react"

interface NavbarProps {
  view: "landing" | "upload" | "about"
  onNavigate: (view: "landing" | "upload" | "about") => void
  onReset: () => void
}

export function Navbar({ view, onNavigate, onReset }: NavbarProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-border/40 bg-background/60 backdrop-blur-xl supports-[backdrop-filter]:bg-background/40">
      <nav className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <button
          onClick={() => {
            onNavigate("landing")
            onReset()
          }}
          className="flex items-center gap-2.5 text-foreground hover:opacity-70 transition-opacity"
        >
          <Shield className="w-5 h-5" strokeWidth={1.5} />
          <span className="font-medium tracking-tight text-lg">Verify</span>
        </button>

        <div className="flex items-center gap-8">
          <button
            onClick={() => onNavigate("about")}
            className={`text-sm transition-colors ${
              view === "about" ? "text-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            About
          </button>
          <button
            onClick={() => onNavigate("upload")}
            className="text-sm bg-foreground text-background px-5 py-2 rounded-full hover:bg-foreground/90 transition-colors font-medium"
          >
            Start Scanning
          </button>
        </div>
      </nav>
    </header>
  )
}
