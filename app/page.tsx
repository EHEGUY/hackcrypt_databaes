'use client';

import { useState, useRef } from 'react';
import { Upload, CheckCircle2, AlertCircle, Loader, Shield, Menu, X } from 'lucide-react';

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [result, setResult] = useState<any>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleAnalyze(files[0]);
    }
  };

  const handleAnalyze = async (file: File) => {
    if (!file.type.startsWith('video/')) {
      setResult({ error: 'Please upload a video file' });
      return;
    }

    setIsLoading(true);
    setFileName(file.name);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/v1/analyze-video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        setResult({ error: errorData.error || 'Analysis failed' });
      } else {
        const data = await response.json();
        setResult(data);
      }
    } catch (error) {
      setResult({ error: 'Failed to analyze video. Ensure backend is running.' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      handleAnalyze(files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-black/40 backdrop-blur-md border-b border-blue-500/20 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-lg flex items-center justify-center">
              <Shield className="w-6 h-6 text-slate-950 font-bold" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">truverify</span>
          </div>

          <div className="hidden md:flex items-center gap-8">
            <a href="#" className="text-gray-300 hover:text-blue-400 transition">Features</a>
            <a href="#" className="text-gray-300 hover:text-blue-400 transition">About</a>
            <button className="px-6 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg hover:shadow-lg hover:shadow-blue-500/50 transition-all">
              Get Started
            </button>
          </div>

          <button
            className="md:hidden text-white"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-24 pb-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* Hero Section */}
          {!result && !isLoading && (
            <div className="text-center mb-12">
              <div className="inline-block mb-4 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full">
                <span className="text-sm font-medium text-blue-300">Advanced AI Detection</span>
              </div>
              <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 text-balance">
                Verify Video
                <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-300 bg-clip-text text-transparent">
                  {' '}Authenticity
                </span>
              </h1>
              <p className="text-xl text-gray-400 mb-8 text-balance max-w-2xl mx-auto">
                Detect deepfakes and manipulated videos with state-of-the-art AI. Get instant authenticity scores and detailed analysis.
              </p>
            </div>
          )}

          {/* Upload Section */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer ${
              isDragging
                ? 'border-blue-400 bg-blue-500/10'
                : 'border-blue-500/30 bg-blue-500/5 hover:border-blue-400 hover:bg-blue-500/10'
            }`}
            onClick={() => !isLoading && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileInput}
              className="hidden"
              disabled={isLoading}
            />

            {isLoading ? (
              <div className="space-y-4">
                <Loader className="w-12 h-12 text-blue-400 animate-spin mx-auto" />
                <p className="text-lg font-semibold text-white">Analyzing your video...</p>
                <p className="text-sm text-gray-400">{fileName}</p>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {result.error ? (
                  <div className="space-y-4">
                    <AlertCircle className="w-12 h-12 text-red-400 mx-auto" />
                    <div>
                      <p className="text-lg font-semibold text-red-300 mb-2">Analysis Failed</p>
                      <p className="text-sm text-gray-400">{result.error}</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className={`w-12 h-12 rounded-full mx-auto flex items-center justify-center ${
                      result.is_deepfake ? 'bg-red-500/20' : 'bg-green-500/20'
                    }`}>
                      {result.is_deepfake ? (
                        <AlertCircle className="w-6 h-6 text-red-400" />
                      ) : (
                        <CheckCircle2 className="w-6 h-6 text-green-400" />
                      )}
                    </div>
                    <div>
                      <p className="text-lg font-semibold text-white mb-2">
                        {result.is_deepfake ? 'Deepfake Detected' : 'Authentic Video'}
                      </p>
                      <p className="text-sm text-gray-400 mb-4">{fileName}</p>
                      
                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="bg-slate-800/50 rounded-lg p-3">
                          <p className="text-xs text-gray-500 mb-1">Confidence</p>
                          <p className="text-lg font-bold text-white">{(result.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-slate-800/50 rounded-lg p-3">
                          <p className="text-xs text-gray-500 mb-1">Authenticity</p>
                          <p className="text-lg font-bold text-green-400">{(result.authenticity_score * 100).toFixed(1)}%</p>
                        </div>
                        <div className="bg-slate-800/50 rounded-lg p-3">
                          <p className="text-xs text-gray-500 mb-1">Deepfake Prob.</p>
                          <p className="text-lg font-bold text-red-400">{(result.deepfake_probability * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <button
                  onClick={() => {
                    setResult(null);
                    setFileName('');
                  }}
                  className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-lg hover:shadow-lg hover:shadow-blue-500/50 hover:scale-105 transition-all font-semibold"
                >
                  Analyze Another Video
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="inline-flex p-4 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-full">
                  <Upload className="w-8 h-8 text-blue-400" />
                </div>
                <div>
                  <p className="text-xl font-semibold text-white mb-2">Drop your video here</p>
                  <p className="text-sm text-gray-400">or click to browse • MP4, WebM, MOV supported</p>
                </div>
              </div>
            )}
          </div>

          {/* Features Section */}
          <div className="grid md:grid-cols-3 gap-6 mt-16">
            {[
              {
                title: 'Advanced Detection',
                description: 'AI-powered algorithms detect subtle deepfake artifacts',
              },
              {
                title: 'Instant Results',
                description: 'Get analysis results in seconds with detailed metrics',
              },
              {
                title: 'Secure Analysis',
                description: 'Your videos are analyzed securely and never stored',
              },
            ].map((feature, i) => (
              <div
                key={i}
                className="p-6 rounded-lg border border-blue-500/20 bg-blue-500/5 hover:border-blue-500/50 hover:bg-blue-500/10 transition-all"
              >
                <p className="text-lg font-semibold text-white mb-2">{feature.title}</p>
                <p className="text-sm text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-blue-500/20 bg-black/20 mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            <div>
              <p className="font-semibold text-white mb-4">truverify</p>
              <p className="text-sm text-gray-400">Advanced deepfake detection powered by AI</p>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-300 mb-4">Product</p>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-blue-400 transition">Features</a></li>
                <li><a href="#" className="hover:text-blue-400 transition">Pricing</a></li>
              </ul>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-300 mb-4">Company</p>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-blue-400 transition">About</a></li>
                <li><a href="#" className="hover:text-blue-400 transition">Blog</a></li>
              </ul>
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-300 mb-4">Legal</p>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-blue-400 transition">Privacy</a></li>
                <li><a href="#" className="hover:text-blue-400 transition">Terms</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-blue-500/20 pt-8 text-center text-sm text-gray-500">
            <p>&copy; 2024 truverify. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
