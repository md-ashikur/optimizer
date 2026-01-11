'use client';

import { ShapExplanation } from '@/types/performance';
import { HiTrendingUp as TrendingUp, HiTrendingDown as TrendingDown } from 'react-icons/hi';
import { FiAlertCircle as AlertCircle } from 'react-icons/fi';

interface ShapExplanationPanelProps {
  explanation: ShapExplanation;
}

export default function ShapExplanationPanel({ explanation }: ShapExplanationPanelProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 animate-fadeIn card-hover">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
          <AlertCircle className="w-6 h-6 text-purple-600 dark:text-purple-400" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-gray-900 dark:text-white">
            AI Explanation
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Understanding why your site got this classification
          </p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Positive Factors */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" />
            <h4 className="font-semibold text-gray-900 dark:text-white">
              Positive Factors
            </h4>
          </div>
          {explanation.top_positive.length > 0 ? (
            <div className="space-y-3">
              {explanation.top_positive.map((impact, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-green-900 dark:text-green-100 capitalize">
                      {impact.feature.replace(/_/g, ' ')}
                    </span>
                    <span className="text-sm font-semibold text-green-700 dark:text-green-300">
                      +{impact.shap_value.toFixed(4)}
                    </span>
                  </div>
                  <div className="w-full bg-green-200 dark:bg-green-800 rounded-full h-2">
                    <div
                      className="bg-green-600 dark:bg-green-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(impact.impact * 100, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-green-700 dark:text-green-300 mt-2">
                    Value: {impact.value.toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400 italic">
              No significant positive factors detected
            </p>
          )}
        </div>

        {/* Negative Factors */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-5 h-5 text-red-600 dark:text-red-400" />
            <h4 className="font-semibold text-gray-900 dark:text-white">
              Negative Factors
            </h4>
          </div>
          {explanation.top_negative.length > 0 ? (
            <div className="space-y-3">
              {explanation.top_negative.map((impact, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-red-900 dark:text-red-100 capitalize">
                      {impact.feature.replace(/_/g, ' ')}
                    </span>
                    <span className="text-sm font-semibold text-red-700 dark:text-red-300">
                      {impact.shap_value.toFixed(4)}
                    </span>
                  </div>
                  <div className="w-full bg-red-200 dark:bg-red-800 rounded-full h-2">
                    <div
                      className="bg-red-600 dark:bg-red-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${Math.min(impact.impact * 100, 100)}%` }}
                    />
                  </div>
                  <p className="text-xs text-red-700 dark:text-red-300 mt-2">
                    Value: {impact.value.toFixed(2)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400 italic">
              No significant negative factors detected
            </p>
          )}
        </div>
      </div>

      {/* Most Important Features */}
      <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <h4 className="font-semibold text-gray-900 dark:text-white mb-4">
          Top Influencing Features
        </h4>
        <div className="space-y-2">
          {explanation.most_important.slice(0, 5).map((impact, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg"
            >
              <div className="flex items-center gap-3">
                <span className="flex items-center justify-center w-6 h-6 bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 rounded-full text-xs font-bold">
                  {idx + 1}
                </span>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                  {impact.feature.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <div className="text-xs text-gray-500 dark:text-gray-400">Impact</div>
                  <div className="text-sm font-bold text-gray-900 dark:text-white">
                    {(impact.impact * 100).toFixed(1)}%
                  </div>
                </div>
                {impact.shap_value > 0 ? (
                  <TrendingUp className="w-4 h-4 text-green-500" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-500" />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-800 dark:text-blue-200">
          <strong>What is SHAP?</strong> SHAP (SHapley Additive exPlanations) explains how each feature 
          contributes to the prediction. Positive values push toward better classification, negative values 
          push toward worse classification.
        </p>
      </div>
    </div>
  );
}
