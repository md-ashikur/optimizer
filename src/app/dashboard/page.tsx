'use client';

import { useEffect, useCallback, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAnalysisStore } from '@/store/analysis.store';
import { analyzeWebsite, checkMLServerHealth } from '@/lib/api/analysis.api';
import { AdvancedAnalysisResult } from '@/types/performance';
import PerformanceGrade from '@/components/dashboard/PerformanceGrade';
import CoreWebVitals from '@/components/dashboard/CoreWebVitals';
import IssuesList from '@/components/dashboard/IssuesList';
import RecommendationsList from '@/components/dashboard/RecommendationsList';
import ExtraFeaturesPanel from '@/components/dashboard/ExtraFeaturesPanel';
import LoadingState from '@/components/dashboard/LoadingState';
import ErrorState from '@/components/dashboard/ErrorState';
import Header from '@/components/shared/Header';
import ShapExplanationPanel from '@/components/dashboard/ShapExplanationPanel';
import RegressionPredictionsPanel from '@/components/dashboard/RegressionPredictionsPanel';
import IntelligentRecommendations from '@/components/dashboard/IntelligentRecommendations';
import OptimizationStrategyPanel from '@/components/dashboard/OptimizationStrategyPanel';
import { HiSparkles as Sparkles, HiTrendingUp as TrendingUp } from 'react-icons/hi';
import { FiInfo as Info } from 'react-icons/fi';
import { FaChartBar as BarChart } from 'react-icons/fa';

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
  const [advancedResult, setAdvancedResult] = useState<AdvancedAnalysisResult | null>(null);
  const [loadingAdvanced, setLoadingAdvanced] = useState(false);
  const [showAdvancedFeatures, setShowAdvancedFeatures] = useState(false);

  const performAnalysis = useCallback(async (urlOverride?: string) => {
    const target = urlOverride ?? currentUrl;
    if (!target) return;

    console.log('Starting analysis for:', currentUrl);
    setAnalyzing(true);
    setError(null);
    setProgress(0);
    setProgressMessage('Initializing...');
    setAdvancedResult(null);

    try {
      // Check ML server health first
      console.log('Checking ML server health...');
      setProgress(5);
      setProgressMessage('Checking ML server...');
      
      const isHealthy = await checkMLServerHealth();
      console.log('ML server health:', isHealthy);
      
      if (!isHealthy) {
        throw new Error('ML server is offline. Start the advanced server (python src/api/ml_server_advanced.py) or the dev server (python src/api/ml_server_fast.py)');
      }

      console.log('Starting website analysis...');
      const result = await analyzeWebsite(target, (prog, msg) => {
        console.log(`Progress: ${prog}% - ${msg}`);
        setProgress(prog);
        setProgressMessage(msg);
      });
      
      console.log('Analysis complete:', result);
      setAnalysisResult(result);

      // Automatically load advanced features after basic analysis
      await loadAdvancedFeatures(result.metrics);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  }, [currentUrl, setAnalyzing, setError, setAnalysisResult]);

  const loadAdvancedFeatures = useCallback(async (metrics: any) => {
    setLoadingAdvanced(true);
    try {
      console.log('Loading advanced ML features...');
      const response = await fetch('/api/analyze-advanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ metrics })
      });

      if (!response.ok) {
        throw new Error('Failed to load advanced features');
      }

      const advanced = await response.json();
      console.log('Advanced features loaded:', advanced);
      setAdvancedResult(advanced);
      setShowAdvancedFeatures(true);
    } catch (err) {
      console.error('Advanced features error:', err);
      // Don't fail the whole analysis if advanced features fail
    } finally {
      setLoadingAdvanced(false);
    }
  }, []);

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
                {advancedResult && (
                  <button
                    onClick={() => router.push('/dashboard/analytics')}
                    className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-smooth flex items-center gap-2"
                  >
                    <BarChart className="w-4 h-4" />
                    View Analytics
                  </button>
                )}
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
            {/* Performance Grade */}
            <PerformanceGrade
              grade={analysisResult.prediction.label}
              confidence={analysisResult.prediction.confidence}
              probabilities={analysisResult.prediction.probabilities}
            />

            {/* Core Web Vitals */}
            <CoreWebVitals metrics={analysisResult.metrics} />

            {/* Advanced ML Features Toggle */}
            <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-xl p-6 border border-purple-500/30">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
                    <Sparkles className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white flex items-center gap-2">
                      Advanced ML Features
                      <span className="px-2 py-1 bg-purple-500/30 rounded-full text-xs font-semibold">NEW</span>
                    </h3>
                    <p className="text-gray-300 text-sm">
                      AI-powered insights with SHAP explanations, predictions & strategies
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    if (!advancedResult && !loadingAdvanced) {
                      loadAdvancedFeatures(analysisResult.metrics);
                    } else {
                      setShowAdvancedFeatures(!showAdvancedFeatures);
                    }
                  }}
                  disabled={loadingAdvanced}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-semibold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {loadingAdvanced ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Loading...
                    </>
                  ) : showAdvancedFeatures ? (
                    <>Hide Advanced Features</>
                  ) : (
                    <>
                      <TrendingUp className="w-5 h-5" />
                      Show Advanced Features
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Advanced ML Components */}
            {showAdvancedFeatures && advancedResult && (
              <div className="space-y-8 animate-fadeIn">
                {/* Info Banner */}
                <div className="bg-blue-900/30 border border-blue-500/30 rounded-xl p-4">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                    <div className="flex-1">
                      <h4 className="text-white font-semibold mb-1">Advanced Analysis Powered by 5 ML Features</h4>
                      <p className="text-gray-300 text-sm">
                        These insights are generated using ensemble classification (100% accuracy), SHAP explainability, 
                        regression models (R² up to 0.97), ML recommendations, and Pareto optimization strategies.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Regression Predictions */}
                <RegressionPredictionsPanel
                  predictions={advancedResult.regression_predictions}
                  currentMetrics={{
                    lcp: analysisResult.metrics.Largest_contentful_paint_LCP_ms,
                    fid: analysisResult.metrics.Max_Potential_FID_ms,
                    cls: analysisResult.metrics.Cumulative_Layout_Shift_CLS
                  }}
                />

                {/* SHAP Explanation */}
                <ShapExplanationPanel explanation={advancedResult.shap_explanation} />

                {/* Intelligent Recommendations */}
                <IntelligentRecommendations recommendations={advancedResult.recommendations} />

                {/* Optimization Strategy */}
                <OptimizationStrategyPanel 
                  currentStrategy={advancedResult.optimization_strategy}
                  onStrategyChange={(strategy) => console.log('Strategy changed to:', strategy)}
                />
              </div>
            )}

            {/* Original Features */}
            <div className="grid lg:grid-cols-2 gap-8">
              <IssuesList issues={analysisResult.issues} />
              {!showAdvancedFeatures && (
                <RecommendationsList recommendations={analysisResult.recommendations} />
              )}
            </div>

            {!showAdvancedFeatures && (
              <div className="mt-8">
                <ExtraFeaturesPanel result={analysisResult} />
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
