'use client';

import { ChevronDown } from 'lucide-react';

export default function Hero() {
  const scrollToUpload = () => {
    document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="min-h-screen pt-24 pb-20 flex flex-col items-center justify-center px-4">
      <div className="max-w-4xl mx-auto text-center space-y-8">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-card border border-border">
          <div className="w-2 h-2 bg-accent rounded-full animate-pulse" />
          <span className="text-sm text-foreground/60">Advanced AI Detection</span>
        </div>

        {/* Main Heading */}
        <h1 className="text-5xl md:text-7xl font-bold text-balance leading-tight">
          Detect Deepfakes with
          <span className="bg-gradient-to-r from-accent via-blue-400 to-cyan-400 bg-clip-text text-transparent">
            {' '}
            Precision
          </span>
        </h1>

        {/* Subheading */}
        <p className="text-lg md:text-xl text-foreground/60 text-balance max-w-2xl mx-auto leading-relaxed">
          Powered by advanced machine learning models, HackCrypt identifies manipulated media with unprecedented accuracy. Trust the detection that leads the industry.
        </p>

        {/* CTA Button */}
        <div className="pt-4">
          <button
            onClick={scrollToUpload}
            className="px-8 py-4 bg-primary text-primary-foreground rounded-full font-semibold hover:shadow-lg hover:shadow-accent/20 transition-smooth hover:scale-105 inline-flex items-center gap-2"
          >
            Start Analyzing
            <ChevronDown className="w-5 h-5" />
          </button>
        </div>

        {/* Decorative Elements */}
        <div className="pt-16 relative h-64 md:h-80">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-72 h-72 bg-accent/5 rounded-full blur-3xl animate-pulse" />
            <div className="absolute w-48 h-48 bg-blue-500/5 rounded-full blur-2xl animate-pulse" style={{ animationDelay: '1s' }} />
          </div>
        </div>
      </div>
    </section>
  );
}
