'use client';

import { RegressionPredictions } from '@/types/performance';
import { FaBullseye as Target, FaBolt as Zap } from 'react-icons/fa';
import { HiTrendingUp as TrendingUp } from 'react-icons/hi';
import { useEffect, useState } from 'react';

interface RegressionPredictionsPanelProps {
  predictions: RegressionPredictions;
  currentMetrics?: {
    lcp: number;
    fid: number;
    cls: number;
  };
}

export default function RegressionPredictionsPanel({ 
  predictions, 
  currentMetrics 
}: RegressionPredictionsPanelProps) {
  const [animatedValues, setAnimatedValues] = useState({
    lcp: 0,
    fid: 0,
    cls: 0
  });

  useEffect(() => {
    // Animate numbers on mount
    const duration = 1500;
    const steps = 60;
    const interval = duration / steps;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      const progress = step / steps;
      
      setAnimatedValues({
        lcp: predictions.predicted_lcp_ms * progress,
        fid: predictions.predicted_fid_ms * progress,
        cls: predictions.predicted_cls * progress
      });

      if (step >= steps) {
        clearInterval(timer);
        setAnimatedValues({
          lcp: predictions.predicted_lcp_ms,
          fid: predictions.predicted_fid_ms,
          cls: predictions.predicted_cls
        });
      }
    }, interval);

    return () => clearInterval(timer);
  }, [predictions]);

  const getStatus = (metric: 'lcp' | 'fid' | 'cls', value: number) => {
    const thresholds = {
      lcp: { good: 2500, needsImprovement: 4000 },
      fid: { good: 100, needsImprovement: 300 },
      cls: { good: 0.1, needsImprovement: 0.25 }
    };

    const threshold = thresholds[metric];
    if (value <= threshold.good) return 'good';
    if (value <= threshold.needsImprovement) return 'needs-improvement';
    return 'poor';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30 border-green-300 dark:border-green-700';
      case 'needs-improvement': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30 border-yellow-300 dark:border-yellow-700';
      case 'poor': return 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30 border-red-300 dark:border-red-700';
      default: return 'text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-700 border-gray-300 dark:border-gray-600';
    }
  };

  const calculateImprovement = (current: number | undefined, predicted: number) => {
    if (!current) return null;
    const improvement = ((current - predicted) / current) * 100;
    return improvement;
  };

  const metrics = [
    {
      key: 'lcp',
      name: 'Largest Contentful Paint',
      abbr: 'LCP',
      current: currentMetrics?.lcp,
      predicted: animatedValues.lcp,
      final: predictions.predicted_lcp_ms,
      unit: 'ms',
      icon: Target,
      description: 'Time to render largest content element',
      thresholds: { good: 2500, needsImprovement: 4000 }
    },
    {
      key: 'fid',
      name: 'First Input Delay',
      abbr: 'FID',
      current: currentMetrics?.fid,
      predicted: animatedValues.fid,
      final: predictions.predicted_fid_ms,
      unit: 'ms',
      icon: Zap,
      description: 'Time to first user interaction',
      thresholds: { good: 100, needsImprovement: 300 }
    },
    {
      key: 'cls',
      name: 'Cumulative Layout Shift',
      abbr: 'CLS',
      current: currentMetrics?.cls,
      predicted: animatedValues.cls,
      final: predictions.predicted_cls,
      unit: '',
      icon: TrendingUp,
      description: 'Visual stability score',
      thresholds: { good: 0.1, needsImprovement: 0.25 }
    }
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 animate-fadeIn card-hover">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
          <Target className="w-6 h-6 text-blue-600 dark:text-blue-400" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-gray-900 dark:text-white">
            Optimized Predictions
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Expected metrics after implementing recommendations
          </p>
        </div>
      </div>

      <div className="grid gap-6">
        {metrics.map((metric) => {
          const status = getStatus(metric.key as 'lcp' | 'fid' | 'cls', metric.final);
          const improvement = calculateImprovement(metric.current, metric.final);

          return (
            <div
              key={metric.key}
              className="p-5 bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-700/50 dark:to-gray-800/50 rounded-xl border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow duration-300"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start gap-3">
                  <div className="p-2 bg-white dark:bg-gray-700 rounded-lg shadow-sm">
                    <metric.icon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900 dark:text-white">
                      {metric.name} ({metric.abbr})
                    </h4>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {metric.description}
                    </p>
                  </div>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${getStatusColor(status)}`}>
                  {status === 'good' ? '✓ Good' : status === 'needs-improvement' ? '⚠ Needs Work' : '✗ Poor'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                {/* Current Value */}
                {metric.current && (
                  <div className="text-center p-3 bg-white dark:bg-gray-700 rounded-lg">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Current</div>
                    <div className="text-2xl font-bold text-gray-700 dark:text-gray-300">
                      {metric.key === 'cls' 
                        ? metric.current.toFixed(4)
                        : Math.round(metric.current)
                      }
                      <span className="text-sm ml-1">{metric.unit}</span>
                    </div>
                  </div>
                )}

                {/* Predicted Value */}
                <div className="text-center p-3 bg-white dark:bg-gray-700 rounded-lg">
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">After Optimization</div>
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {metric.key === 'cls' 
                      ? metric.predicted.toFixed(4)
                      : Math.round(metric.predicted)
                    }
                    <span className="text-sm ml-1">{metric.unit}</span>
                  </div>
                </div>
              </div>

              {/* Improvement Indicator */}
              {improvement !== null && (
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                  <span className="text-sm text-green-700 dark:text-green-300 font-medium">
                    Expected Improvement
                  </span>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-green-600 dark:text-green-400" />
                    <span className="text-lg font-bold text-green-600 dark:text-green-400">
                      {improvement.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Threshold Reference */}
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500 dark:text-gray-400">Thresholds:</span>
                  <div className="flex items-center gap-3">
                    <span className="text-green-600 dark:text-green-400">
                      Good: ≤ {metric.thresholds.good}{metric.unit}
                    </span>
                    <span className="text-yellow-600 dark:text-yellow-400">
                      Fair: ≤ {metric.thresholds.needsImprovement}{metric.unit}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          <strong>ML-Powered Predictions:</strong> These values are predicted using advanced regression models 
          (Gradient Boosting & Random Forest) trained on thousands of optimized websites. Accuracy: LCP (R²=0.81), 
          FID (R²=0.63), CLS (R²=0.97).
        </p>
      </div>
    </div>
  );
}
