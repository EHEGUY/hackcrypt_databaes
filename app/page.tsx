'use client';

import { useState, useEffect } from 'react';
import Navigation from '@/components/navigation';
import Hero from '@/components/hero';
import UploadSection from '@/components/upload-section';
import Footer from '@/components/footer';

export default function Home() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-b from-background via-background to-card">
      <Navigation scrolled={scrolled} />
      <Hero />
      <UploadSection />
      <Footer />
    </main>
  );
}
