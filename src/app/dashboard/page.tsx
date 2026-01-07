'use client';

import { useEffect, useCallback, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAnalysisStore } from '@/store/analysis.store';
import { analyzeWebsite, checkMLServerHealth } from '@/lib/api/analysis.api';
import PerformanceGrade from '@/components/dashboard/PerformanceGrade';
import CoreWebVitals from '@/components/dashboard/CoreWebVitals';
import IssuesList from '@/components/dashboard/IssuesList';
import RecommendationsList from '@/components/dashboard/RecommendationsList';
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

  const performAnalysis = useCallback(async () => {
    if (!currentUrl) return;

    setAnalyzing(true);
    setError(null);
    setProgress(0);
    setProgressMessage('Initializing...');

    try {
      // Check ML server health first
      setProgress(5);
      setProgressMessage('Checking ML server...');
      const isHealthy = await checkMLServerHealth();
      
      if (!isHealthy) {
        throw new Error('ML server is offline. Please start the Python server: python src/api/ml_server.py');
      }

      const result = await analyzeWebsite(currentUrl, (prog, msg) => {
        setProgress(prog);
        setProgressMessage(msg);
      });
      
      setAnalysisResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  }, [currentUrl, setAnalyzing, setError, setAnalysisResult]);

  useEffect(() => {
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
            <h1 className="text-3xl font-bold text-white mb-2">Performance Analysis</h1>
            <p className="text-gray-400">{analysisResult.url}</p>
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
          </div>
        </div>
      </main>
    </div>
  );
}
