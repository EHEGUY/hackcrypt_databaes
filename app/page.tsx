'use client';

import { useState } from 'react';
import Navigation from '@/components/navigation';
import Hero from '@/components/hero';
import UploadSection from '@/components/upload-section';
import Features from '@/components/features';
import Footer from '@/components/footer';

export default function Home() {
  const [scrolled, setScrolled] = useState(false);

  const handleScroll = () => {
    setScrolled(window.scrollY > 50);
  };

  if (typeof window !== 'undefined') {
    window.addEventListener('scroll', handleScroll, { passive: true });
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-background via-background to-card">
      <Navigation scrolled={scrolled} />
      <Hero />
      <UploadSection />
      <Features />
      <Footer />
    </main>
  );
}
