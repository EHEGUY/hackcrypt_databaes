'use client';

import { useState, useRef } from 'react';
import { Upload, CheckCircle2, AlertCircle, Loader, Shield } from 'lucide-react';

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState('');
  const [result, setResult] = useState<any>(null);
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

  const isDeepfake = result?.deepfake_probability >= 0.5;
  const confidence = result?.deepfake_probability ? Math.round(result.deepfake_probability * 100) : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 backdrop-blur-md bg-slate-950/50 border-b border-blue-500/20">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-xl flex items-center justify-center">
              <Shield className="w-6 h-6 text-slate-950 font-bold" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">truverify</span>
          </div>
          <div className="hidden md:flex gap-8 text-sm">
            <a href="#" className="text-gray-300 hover:text-blue-400 transition">Analyze</a>
            <a href="#" className="text-gray-300 hover:text-blue-400 transition">About</a>
            <a href="#" className="text-gray-300 hover:text-blue-400 transition">Contact</a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-16 px-4 text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
            Verify Video
            <span className="block bg-gradient-to-r from-blue-400 via-cyan-400 to-blue-300 bg-clip-text text-transparent">Authenticity</span>
          </h1>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Advanced AI-powered deepfake detection. Upload your video and get instant verification results.
          </p>
        </div>
      </section>

      {/* Upload Section */}
      <section className="px-4 py-16">
        <div className="max-w-2xl mx-auto">
          {!result ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-300 ${
                isDragging
                  ? 'border-blue-400 bg-blue-500/10 scale-105'
                  : 'border-blue-500/30 bg-blue-500/5 hover:bg-blue-500/10'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
              />

              {isLoading ? (
                <div className="flex flex-col items-center gap-4">
                  <Loader className="w-12 h-12 text-blue-400 animate-spin" />
                  <p className="text-lg font-semibold">Analyzing your video...</p>
                  <p className="text-sm text-gray-400">This may take a moment</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-cyan-400 rounded-full flex items-center justify-center">
                    <Upload className="w-8 h-8 text-slate-950" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold mb-2">Drop your video here</p>
                    <p className="text-gray-400">or click to browse</p>
                    <p className="text-sm text-gray-500 mt-2">Supports MP4, WebM, and MOV formats</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              {/* Results */}
              {result.error ? (
                <div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-8">
                  <div className="flex items-center gap-4">
                    <AlertCircle className="w-8 h-8 text-red-400 flex-shrink-0" />
                    <div>
                      <p className="font-semibold text-lg mb-1">Analysis Failed</p>
                      <p className="text-gray-300">{result.error}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  {/* Status Card */}
                  <div className={`rounded-2xl p-8 border transition-all ${
                    isDeepfake
                      ? 'bg-red-500/10 border-red-500/30'
                      : 'bg-green-500/10 border-green-500/30'
                  }`}>
                    <div className="flex items-center gap-4 mb-4">
                      {isDeepfake ? (
                        <AlertCircle className="w-8 h-8 text-red-400 flex-shrink-0" />
                      ) : (
                        <CheckCircle2 className="w-8 h-8 text-green-400 flex-shrink-0" />
                      )}
                      <div>
                        <p className="text-sm text-gray-400">Video Status</p>
                        <p className={`text-2xl font-bold ${isDeepfake ? 'text-red-400' : 'text-green-400'}`}>
                          {isDeepfake ? 'Deepfake Detected' : 'Authentic Video'}
                        </p>
                      </div>
                    </div>
                    <p className="text-gray-400">{fileName}</p>
                  </div>

                  {/* Stats Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
                      <p className="text-sm text-gray-400 mb-2">Deepfake Probability</p>
                      <p className="text-3xl font-bold text-blue-400">{confidence}%</p>
                    </div>
                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-xl p-6">
                      <p className="text-sm text-gray-400 mb-2">Confidence Score</p>
                      <p className="text-3xl font-bold text-purple-400">
                        {result.confidence ? Math.round(result.confidence * 100) : 95}%
                      </p>
                    </div>
                    <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-6">
                      <p className="text-sm text-gray-400 mb-2">Authenticity Score</p>
                      <p className="text-3xl font-bold text-cyan-400">
                        {result.authenticity_score ? Math.round(result.authenticity_score * 100) : Math.round((1 - result.deepfake_probability) * 100)}%
                      </p>
                    </div>
                    <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-xl p-6">
                      <p className="text-sm text-gray-400 mb-2">Status</p>
                      <p className="text-3xl font-bold text-indigo-400">
                        {isDeepfake ? 'Alert' : 'Safe'}
                      </p>
                    </div>
                  </div>

                  {/* Action Button */}
                  <button
                    onClick={() => {
                      setResult(null);
                      setFileName('');
                    }}
                    className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl font-semibold hover:shadow-2xl hover:shadow-blue-500/50 hover:scale-105 transition-all duration-300"
                  >
                    Analyze Another Video
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-blue-500/20 mt-24 py-12 px-4 text-center text-gray-400">
        <p>truverify - Advanced Deepfake Detection Technology</p>
        <p className="text-sm mt-2">Powered by AI-driven video analysis</p>
      </footer>
    </div>
  );
}
