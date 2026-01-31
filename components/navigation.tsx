'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Menu, X, Shield } from 'lucide-react';

interface NavigationProps {
  scrolled: boolean;
}

export default function Navigation({ scrolled }: NavigationProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav
      className={`fixed top-0 w-full z-50 transition-smooth ${
        scrolled
          ? 'bg-background/80 backdrop-blur-md border-b border-border'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16 md:h-20">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-semibold hidden sm:inline">HackCrypt</span>
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-8">
            <a href="#upload" className="text-sm text-foreground/70 hover:text-foreground transition-smooth">
              Analyze
            </a>
            <a href="#features" className="text-sm text-foreground/70 hover:text-foreground transition-smooth">
              Features
            </a>
            <a href="#contact" className="text-sm text-foreground/70 hover:text-foreground transition-smooth">
              Contact
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg hover:bg-card transition-smooth"
          >
            {mobileOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileOpen && (
          <div className="md:hidden pb-4 border-t border-border">
            <div className="flex flex-col gap-3 pt-4">
              <a
                href="#upload"
                className="px-4 py-2 text-sm text-foreground/70 hover:text-foreground hover:bg-card rounded-lg transition-smooth"
              >
                Analyze
              </a>
              <a
                href="#features"
                className="px-4 py-2 text-sm text-foreground/70 hover:text-foreground hover:bg-card rounded-lg transition-smooth"
              >
                Features
              </a>
              <a
                href="#contact"
                className="px-4 py-2 text-sm text-foreground/70 hover:text-foreground hover:bg-card rounded-lg transition-smooth"
              >
                Contact
              </a>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
