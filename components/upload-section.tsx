'use client';

import { useState, useRef } from 'react';
import { Upload, CheckCircle, AlertCircle, Loader } from 'lucide-react';

interface AnalysisResult {
  deepfake_probability?: number;
  authenticity_score?: number;
  confidence?: number;
  error?: string;
}

export default function UploadSection() {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
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
      const response = await fetch('http://localhost:8000/api/v1/analyze-video', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: 'Failed to connect to analysis server. Make sure backend is running.' });
    } finally {
      setIsLoading(false);
    }
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
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      handleAnalyze(files[0]);
    }
  };

  return (
    <section id="upload" className="py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="space-y-12">
          {/* Upload Area */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative px-8 py-16 rounded-2xl border-2 border-dashed cursor-pointer transition-smooth ${
              isDragging
                ? 'border-accent bg-accent/5'
                : 'border-border bg-card/30 hover:bg-card/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            <div className="flex flex-col items-center justify-center gap-4">
              <div className={`p-4 rounded-full transition-smooth ${isDragging ? 'bg-accent/20' : 'bg-secondary'}`}>
                <Upload className={`w-8 h-8 ${isDragging ? 'text-accent' : 'text-foreground/60'}`} />
              </div>
              <div className="text-center">
                <p className="text-lg font-semibold text-foreground">
                  {isLoading ? 'Analyzing...' : 'Drop your video here'}
                </p>
                <p className="text-sm text-foreground/50 mt-2">
                  or click to browse. Supports MP4, WebM, and MOV formats
                </p>
              </div>
            </div>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="flex flex-col items-center gap-4 py-8">
              <div className="relative w-16 h-16">
                <Loader className="w-16 h-16 text-accent animate-spin" />
              </div>
              <p className="text-foreground/60">Analyzing video with AI models...</p>
              <p className="text-sm text-foreground/40">This may take a minute</p>
            </div>
          )}

          {/* Results */}
          {result && !isLoading && (
            <div className="space-y-6 animate-fade-in">
              <div className="flex items-center gap-3">
                {result.error ? (
                  <AlertCircle className="w-6 h-6 text-red-500" />
                ) : (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                )}
                <div>
                  <p className="font-semibold text-foreground">Analysis Complete</p>
                  <p className="text-sm text-foreground/60">{fileName}</p>
                </div>
              </div>

              {result.error ? (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl">
                  <p className="text-red-400 text-sm">{result.error}</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {result.authenticity_score !== undefined && (
                    <ResultCard
                      label="Authenticity Score"
                      value={`${(result.authenticity_score * 100).toFixed(1)}%`}
                      color={result.authenticity_score > 0.7 ? 'green' : result.authenticity_score > 0.4 ? 'yellow' : 'red'}
                    />
                  )}
                  {result.deepfake_probability !== undefined && (
                    <ResultCard
                      label="Deepfake Probability"
                      value={`${(result.deepfake_probability * 100).toFixed(1)}%`}
                      color={result.deepfake_probability > 0.5 ? 'red' : 'green'}
                    />
                  )}
                  {result.confidence !== undefined && (
                    <ResultCard
                      label="Confidence"
                      value={`${(result.confidence * 100).toFixed(1)}%`}
                      color="blue"
                    />
                  )}
                </div>
              )}

              <button
                onClick={() => {
                  setResult(null);
                  setFileName('');
                }}
                className="w-full py-3 px-4 bg-secondary text-foreground rounded-xl hover:bg-secondary/80 transition-smooth font-medium"
              >
                Analyze Another Video
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

interface ResultCardProps {
  label: string;
  value: string;
  color: 'green' | 'red' | 'yellow' | 'blue';
}

function ResultCard({ label, value, color }: ResultCardProps) {
  const colorClasses = {
    green: 'border-green-500/30 bg-green-500/5',
    red: 'border-red-500/30 bg-red-500/5',
    yellow: 'border-yellow-500/30 bg-yellow-500/5',
    blue: 'border-blue-500/30 bg-blue-500/5',
  };

  const textClasses = {
    green: 'text-green-400',
    red: 'text-red-400',
    yellow: 'text-yellow-400',
    blue: 'text-blue-400',
  };

  return (
    <div className={`p-4 rounded-xl border ${colorClasses[color]}`}>
      <p className="text-sm text-foreground/60 mb-2">{label}</p>
      <p className={`text-2xl font-bold ${textClasses[color]}`}>{value}</p>
    </div>
  );
}
