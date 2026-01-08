'use client';

import { useEffect, useCallback, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAnalysisStore } from '@/store/analysis.store';
import { analyzeWebsite, checkMLServerHealth } from '@/lib/api/analysis.api';
import PerformanceGrade from '@/components/dashboard/PerformanceGrade';
import CoreWebVitals from '@/components/dashboard/CoreWebVitals';
import IssuesList from '@/components/dashboard/IssuesList';
import RecommendationsList from '@/components/dashboard/RecommendationsList';
import ExtraFeaturesPanel from '@/components/dashboard/ExtraFeaturesPanel';
import LoadingState from '@/components/dashboard/LoadingState';
import ErrorState from '@/components/dashboard/ErrorState';
import Header from '@/components/shared/Header';

export default function DashboardPage() {
  const router = useRouter();
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('Initializing...');
  const {
    currentUrl,
    analysisResult,
    isAnalyzing,
    error,
    setAnalysisResult,
    setAnalyzing,
    setError,
  } = useAnalysisStore();
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [newUrl, setNewUrl] = useState('');

  const performAnalysis = useCallback(async (urlOverride?: string) => {
    const target = urlOverride ?? currentUrl;
    if (!target) return;

    console.log('🚀 Starting analysis for:', currentUrl);
    setAnalyzing(true);
    setError(null);
    setProgress(0);
    setProgressMessage('Initializing...');

    try {
      // Check ML server health first
      console.log('📡 Checking ML server health...');
      setProgress(5);
      setProgressMessage('Checking ML server...');
      
      const isHealthy = await checkMLServerHealth();
      console.log('✅ ML server health:', isHealthy);
      
      if (!isHealthy) {
        throw new Error('ML server is offline. Please start: python src/api/ml_server_fast.py');
      }

      console.log('🔍 Starting website analysis...');
      const result = await analyzeWebsite(target, (prog, msg) => {
        console.log(`📊 Progress: ${prog}% - ${msg}`);
        setProgress(prog);
        setProgressMessage(msg);
      });
      
      console.log('✨ Analysis complete:', result);
      setAnalysisResult(result);
    } catch (err) {
      console.error('❌ Analysis error:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  }, [currentUrl, setAnalyzing, setError, setAnalysisResult]);

  useEffect(() => {
    console.log('Dashboard mount state:', { currentUrl, analysisResult, isAnalyzing, error });

    if (!currentUrl) {
      router.push('/');
      return;
    }

    if (!analysisResult && !isAnalyzing && !error) {
      performAnalysis();
    }
  }, [currentUrl, analysisResult, isAnalyzing, error, router, performAnalysis]);

  if (isAnalyzing) {
    return <LoadingState url={currentUrl!} progress={progress} message={progressMessage} />;
  }

  if (error) {
    return <ErrorState error={error} onRetry={performAnalysis} />;
  }

  if (!analysisResult) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Header />

      <main className="container mx-auto px-4 py-12">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <div className="flex items-center justify-between gap-4">
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Performance Analysis</h1>
                <p className="text-gray-400">{analysisResult.url}</p>
              </div>

              <div className="flex items-center gap-2">
                {showUrlInput ? (
                  <div className="flex gap-2">
                    <input
                      className="px-3 py-2 rounded bg-slate-800 text-white border border-slate-700"
                      placeholder="https://example.com"
                      value={newUrl}
                      onChange={(e) => setNewUrl(e.target.value)}
                    />
                    <button
                      className="py-2 px-3 bg-emerald-600 text-white rounded"
                      onClick={() => {
                        if (!newUrl) return;
                        setAnalysisResult(null as any);
                        useAnalysisStore.getState().setUrl(newUrl);
                        setShowUrlInput(false);
                        setNewUrl('');
                        performAnalysis(newUrl).catch(() => {});
                      }}
                    >
                      Analyze
                    </button>
                    <button className="py-2 px-3 bg-slate-700 text-white rounded" onClick={() => setShowUrlInput(false)}>Cancel</button>
                  </div>
                ) : (
                  <button
                    className="py-2 px-4 bg-slate-700 text-white rounded hover:bg-slate-600"
                    onClick={() => {
                      setNewUrl(analysisResult.url || '');
                      setShowUrlInput(true);
                    }}
                  >
                    Analyze another URL
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="grid gap-8">
            <PerformanceGrade
              grade={analysisResult.prediction.label}
              confidence={analysisResult.prediction.confidence}
              probabilities={analysisResult.prediction.probabilities}
            />

            <CoreWebVitals metrics={analysisResult.metrics} />

            <div className="grid lg:grid-cols-2 gap-8">
              <IssuesList issues={analysisResult.issues} />
              <RecommendationsList recommendations={analysisResult.recommendations} />
            </div>

            <div className="mt-8">
              <ExtraFeaturesPanel result={analysisResult} />
            </div>

            {/* Bottom action removed — moved to header */}
          </div>
        </div>
      </main>
    </div>
  );
}
