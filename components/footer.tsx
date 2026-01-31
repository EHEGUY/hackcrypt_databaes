import Link from 'next/link';
import { Shield } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-border py-12 px-4 bg-gradient-to-b from-transparent to-card/50">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-semibold">HackCrypt</span>
            </div>
            <p className="text-sm text-foreground/60">
              Advanced deepfake detection powered by AI
            </p>
          </div>

          {/* Product */}
          <div className="space-y-4">
            <h3 className="font-semibold">Product</h3>
            <ul className="space-y-2 text-sm text-foreground/60">
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Features
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Pricing
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Security
                </Link>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div className="space-y-4">
            <h3 className="font-semibold">Company</h3>
            <ul className="space-y-2 text-sm text-foreground/60">
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  About
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Blog
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Careers
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div className="space-y-4">
            <h3 className="font-semibold">Legal</h3>
            <ul className="space-y-2 text-sm text-foreground/60">
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Privacy
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Terms
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-foreground transition-smooth">
                  Contact
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="border-t border-border pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-sm text-foreground/50">
            © 2024 HackCrypt. All rights reserved.
          </p>
          <div className="flex gap-6 text-sm text-foreground/50">
            <Link href="#" className="hover:text-foreground transition-smooth">
              Twitter
            </Link>
            <Link href="#" className="hover:text-foreground transition-smooth">
              Discord
            </Link>
            <Link href="#" className="hover:text-foreground transition-smooth">
              GitHub
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
