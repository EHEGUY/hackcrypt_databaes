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

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      handleAnalyze(e.target.files[0]);
    }
  };

  const handleAnalyze = async (file: File) => {
    if (!file.type.startsWith('video/')) {
      setResult({ error: 'Please upload a video file (MP4, WebM, or MOV)' });
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
        setResult({ error: errorData.error || 'Analysis failed. Please try again.' });
      } else {
        const data = await response.json();
        setResult(data);
      }
    } catch (error) {
      console.log('[v0] Error during analysis:', error);
      setResult({ error: 'Failed to analyze video. Please ensure the backend is running.' });
    } finally {
      setIsLoading(false);
    }
  };

  const scrollToUpload = () => {
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
      uploadSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white overflow-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-slate-950/80 backdrop-blur border-b border-blue-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                truverify
              </span>
            </div>
            
            <div className="hidden md:flex items-center gap-8">
              <button onClick={scrollToUpload} className="text-blue-300 hover:text-cyan-300 transition">
                Upload
              </button>
              <a href="#" className="text-blue-300 hover:text-cyan-300 transition">
                About
              </a>
              <a href="#" className="text-blue-300 hover:text-cyan-300 transition">
                Contact
              </a>
            </div>

            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden text-blue-300 hover:text-cyan-300"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>

          {mobileMenuOpen && (
            <div className="md:hidden pb-4 border-t border-blue-900/30 mt-4 space-y-2">
              <button
                onClick={() => {
                  scrollToUpload();
                  setMobileMenuOpen(false);
                }}
                className="block w-full text-left px-4 py-2 text-blue-300 hover:text-cyan-300"
              >
                Upload
              </button>
              <a href="#" className="block px-4 py-2 text-blue-300 hover:text-cyan-300">
                About
              </a>
              <a href="#" className="block px-4 py-2 text-blue-300 hover:text-cyan-300">
                Contact
              </a>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <div className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-block mb-6 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full">
            <span className="text-sm font-medium text-blue-300">AI-Powered Verification</span>
          </div>

          {/* Heading */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6 leading-tight">
            Verify Video
            <span className="block bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-300 bg-clip-text text-transparent">
              Authenticity
            </span>
          </h1>

          {/* Subheading */}
          <p className="text-lg sm:text-xl text-slate-300 mb-12 max-w-2xl mx-auto leading-relaxed">
            Advanced deepfake detection powered by cutting-edge AI. Analyze video authenticity in seconds with industry-leading accuracy.
          </p>

          {/* CTA Button */}
          <button
            onClick={scrollToUpload}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-full font-semibold text-lg hover:shadow-2xl hover:shadow-blue-500/50 hover:scale-105 transition-all duration-300 inline-flex items-center gap-2"
          >
            Start Verifying
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </button>
        </div>

        {/* Floating cards */}
        <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <div className="p-6 bg-blue-500/5 border border-blue-500/20 rounded-2xl backdrop-blur-sm hover:border-blue-500/40 transition-all">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Advanced Detection</h3>
            <p className="text-slate-400 text-sm">AI-powered algorithms detect deepfakes with 99% accuracy</p>
          </div>

          <div className="p-6 bg-blue-500/5 border border-blue-500/20 rounded-2xl backdrop-blur-sm hover:border-blue-500/40 transition-all">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mb-4">
              <Loader className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Fast Analysis</h3>
            <p className="text-slate-400 text-sm">Get results in seconds, not minutes</p>
          </div>

          <div className="p-6 bg-blue-500/5 border border-blue-500/20 rounded-2xl backdrop-blur-sm hover:border-blue-500/40 transition-all">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mb-4">
              <CheckCircle2 className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Reliable Results</h3>
            <p className="text-slate-400 text-sm">Industry-leading detection technology</p>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div id="upload-section" className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-transparent to-blue-950/20">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl sm:text-5xl font-bold mb-4">Upload Your Video</h2>
            <p className="text-slate-400 text-lg">Drag and drop or click to select a video file</p>
          </div>

          {!result ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative p-12 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 ${
                isDragging
                  ? 'border-cyan-400 bg-cyan-500/10 scale-105'
                  : 'border-blue-500/40 bg-blue-500/5 hover:border-blue-400 hover:bg-blue-500/10'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/webm,video/quicktime"
                onChange={handleFileSelect}
                className="hidden"
              />

              <div className="flex flex-col items-center justify-center">
                <div className={`p-4 rounded-full mb-4 transition-all ${isDragging ? 'bg-cyan-500/20 scale-110' : 'bg-blue-500/10'}`}>
                  <Upload className={`w-8 h-8 ${isDragging ? 'text-cyan-400' : 'text-blue-400'}`} />
                </div>
                <p className="text-xl font-semibold mb-2">
                  {isLoading ? 'Analyzing your video...' : 'Drop your video here'}
                </p>
                <p className="text-slate-400">or click to browse • Supports MP4, WebM, and MOV</p>
                
                {isLoading && (
                  <div className="mt-4 flex items-center gap-2">
                    <Loader className="w-5 h-5 text-cyan-400 animate-spin" />
                    <span className="text-slate-300">Processing...</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Results */}
              {result.error ? (
                <div className="p-8 bg-red-500/10 border border-red-500/30 rounded-2xl">
                  <div className="flex items-start gap-4">
                    <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1" />
                    <div>
                      <h3 className="text-lg font-semibold text-red-300 mb-2">Analysis Failed</h3>
                      <p className="text-red-200">{result.error}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-8">
                    <div className="flex items-start justify-between mb-6">
                      <div>
                        <h3 className="text-2xl font-bold mb-2">{fileName}</h3>
                        <p className="text-slate-400">Analysis Complete</p>
                      </div>
                      <CheckCircle2 className="w-8 h-8 text-green-400" />
                    </div>

                    {/* Metrics Grid */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-xl">
                        <p className="text-slate-400 text-sm mb-1">Deepfake Probability</p>
                        <div className="flex items-end gap-2">
                          <span className="text-3xl font-bold">{result.deepfake_probability?.toFixed(1) || 'N/A'}%</span>
                        </div>
                      </div>

                      <div className="p-4 bg-cyan-500/5 border border-cyan-500/20 rounded-xl">
                        <p className="text-slate-400 text-sm mb-1">Authenticity Score</p>
                        <div className="flex items-end gap-2">
                          <span className="text-3xl font-bold">{result.authenticity_score?.toFixed(1) || 'N/A'}%</span>
                        </div>
                      </div>

                      {result.confidence !== undefined && (
                        <div className="p-4 bg-slate-500/5 border border-slate-500/20 rounded-xl">
                          <p className="text-slate-400 text-sm mb-1">Confidence</p>
                          <div className="flex items-end gap-2">
                            <span className="text-3xl font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                      )}

                      {result.face_manipulated !== undefined && (
                        <div className="p-4 bg-slate-500/5 border border-slate-500/20 rounded-xl">
                          <p className="text-slate-400 text-sm mb-1">Face Manipulation</p>
                          <div className="flex items-end gap-2">
                            <span className="text-3xl font-bold">{result.face_manipulated ? 'Detected' : 'Not Detected'}</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Verdict */}
                    <div className="mt-6 p-4 rounded-xl flex items-center gap-3" style={{
                      backgroundColor: (result.deepfake_probability || 0) > 50 ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)',
                      borderColor: (result.deepfake_probability || 0) > 50 ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 197, 94, 0.3)',
                      borderWidth: '1px'
                    }}>
                      {(result.deepfake_probability || 0) > 50 ? (
                        <>
                          <AlertCircle className="w-5 h-5 text-red-400" />
                          <span className="font-semibold text-red-300">Likely Deepfake Detected</span>
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="w-5 h-5 text-green-400" />
                          <span className="font-semibold text-green-300">Appears Authentic</span>
                        </>
                      )}
                    </div>
                  </div>
                </>
              )}

              {/* Action Button */}
              <button
                onClick={() => {
                  setResult(null);
                  setFileName('');
                }}
                className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl hover:shadow-2xl hover:shadow-blue-500/50 hover:scale-105 transition-all duration-300 font-semibold text-lg"
              >
                Analyze Another Video
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-blue-900/30 bg-slate-950/50 backdrop-blur py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                  <Shield className="w-4 h-4 text-white" />
                </div>
                <span className="font-bold text-lg bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  truverify
                </span>
              </div>
              <p className="text-slate-400 text-sm">Advanced deepfake detection technology</p>
            </div>

            <div>
              <h4 className="font-semibold mb-4 text-slate-200">Product</h4>
              <ul className="space-y-2">
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">Features</a></li>
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">Pricing</a></li>
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">API</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4 text-slate-200">Company</h4>
              <ul className="space-y-2">
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">About</a></li>
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">Blog</a></li>
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">Contact</a></li>
              </ul>
            </div>

            <div>
              <h4 className="font-semibold mb-4 text-slate-200">Legal</h4>
              <ul className="space-y-2">
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">Privacy</a></li>
                <li><a href="#" className="text-slate-400 hover:text-cyan-400 transition">Terms</a></li>
              </ul>
            </div>
          </div>

          <div className="pt-8 border-t border-blue-900/30 text-center text-slate-400 text-sm">
            <p>&copy; 2024 truverify. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
