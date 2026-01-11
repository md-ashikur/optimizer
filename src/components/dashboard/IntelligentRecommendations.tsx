'use client';

import { CategorizedRecommendations } from '@/types/performance';
import { FiAlertTriangle as AlertTriangle, FiCheckCircle as CheckCircle, FiInfo as Info } from 'react-icons/fi';
import { HiSparkles as Sparkles } from 'react-icons/hi';
import { useState } from 'react';

interface IntelligentRecommendationsProps {
  recommendations: CategorizedRecommendations;
}

export default function IntelligentRecommendations({ recommendations }: IntelligentRecommendationsProps) {
  const [expandedCategory, setExpandedCategory] = useState<string | null>('HIGH');

  const categories = [
    {
      key: 'HIGH',
      name: 'High Priority',
      icon: AlertTriangle,
      color: 'red',
      description: 'Critical issues that significantly impact performance',
      gradient: 'from-red-500 to-red-600'
    },
    {
      key: 'MEDIUM',
      name: 'Medium Priority',
      icon: Info,
      color: 'yellow',
      description: 'Important optimizations for better user experience',
      gradient: 'from-yellow-500 to-yellow-600'
    },
    {
      key: 'LOW',
      name: 'Low Priority',
      icon: CheckCircle,
      color: 'blue',
      description: 'Optional enhancements for marginal improvements',
      gradient: 'from-blue-500 to-blue-600'
    }
  ];

  const totalRecommendations = Object.values(recommendations).reduce((sum, arr) => sum + arr.length, 0);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 animate-fadeIn card-hover">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              AI-Powered Recommendations
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {totalRecommendations} personalized optimization suggestions
            </p>
          </div>
        </div>
        <div className="px-4 py-2 bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 rounded-full border border-purple-300 dark:border-purple-700">
          <span className="text-sm font-semibold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            ML-Generated
          </span>
        </div>
      </div>

      {totalRecommendations === 0 ? (
        <div className="text-center py-12">
          <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
            Excellent Performance!
          </h4>
          <p className="text-gray-500 dark:text-gray-400">
            Your website is already well-optimized. Keep up the great work!
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {categories.map((category) => {
            const items = recommendations[category.key as keyof CategorizedRecommendations];
            if (items.length === 0) return null;

            const isExpanded = expandedCategory === category.key;
            const Icon = category.icon;

            return (
              <div
                key={category.key}
                className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden"
              >
                {/* Category Header */}
                <button
                  onClick={() => setExpandedCategory(isExpanded ? null : category.key)}
                  className="w-full p-4 flex items-center justify-between bg-gradient-to-r hover:opacity-90 transition-opacity"
                  style={{
                    background: isExpanded 
                      ? `linear-gradient(to right, var(--tw-gradient-stops))`
                      : `linear-gradient(to right, rgba(var(--tw-${category.color}-50-rgb, 0.05), 0.05) 0%, rgba(var(--tw-${category.color}-100-rgb, 0.1), 0.1) 100%)`
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div className={`p-2 bg-${category.color}-100 dark:bg-${category.color}-900/30 rounded-lg`}>
                      <Icon className={`w-5 h-5 text-${category.color}-600 dark:text-${category.color}-400`} />
                    </div>
                    <div className="text-left">
                      <h4 className={`font-semibold text-${category.color}-900 dark:text-${category.color}-100`}>
                        {category.name}
                      </h4>
                      <p className={`text-xs text-${category.color}-700 dark:text-${category.color}-300`}>
                        {category.description}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 bg-${category.color}-100 dark:bg-${category.color}-900/30 text-${category.color}-700 dark:text-${category.color}-300 rounded-full text-sm font-semibold`}>
                      {items.length} {items.length === 1 ? 'item' : 'items'}
                    </span>
                    <svg
                      className={`w-5 h-5 text-${category.color}-600 dark:text-${category.color}-400 transition-transform ${
                        isExpanded ? 'rotate-180' : ''
                      }`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </button>

                {/* Category Content */}
                {isExpanded && (
                  <div className="p-4 bg-white dark:bg-gray-800 space-y-3">
                    {items.map((item, idx) => (
                      <div
                        key={idx}
                        className={`p-4 border-l-4 border-${category.color}-500 bg-${category.color}-50/50 dark:bg-${category.color}-900/10 rounded-r-lg hover:bg-${category.color}-50 dark:hover:bg-${category.color}-900/20 transition-colors`}
                      >
                        <div className="flex items-start gap-3">
                          <div className={`mt-1 flex-shrink-0 w-6 h-6 bg-${category.color}-100 dark:bg-${category.color}-900/30 rounded-full flex items-center justify-center`}>
                            <span className={`text-xs font-bold text-${category.color}-700 dark:text-${category.color}-300`}>
                              {idx + 1}
                            </span>
                          </div>
                          <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed flex-1">
                            {item}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Feature Info */}
      <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
        <div className="flex items-start gap-3">
          <Sparkles className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5" />
          <div className="flex-1">
            <h5 className="font-semibold text-purple-900 dark:text-purple-100 mb-1">
              Machine Learning Powered
            </h5>
            <p className="text-sm text-purple-800 dark:text-purple-200">
              These recommendations are generated by analyzing your site's performance metrics using a hybrid 
              ML model (Random Forest + Rule Engine) trained on thousands of optimized websites. The system 
              learns from successful optimization cases to provide personalized suggestions.
            </p>
          </div>
        </div>
      </div>

      {/* Implementation Guide */}
      {totalRecommendations > 0 && (
        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h5 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
            Implementation Priority
          </h5>
          <ol className="text-sm text-blue-800 dark:text-blue-200 space-y-1 list-decimal list-inside">
            <li>Start with HIGH priority items for maximum impact</li>
            <li>Implement MEDIUM priority items for additional improvements</li>
            <li>Consider LOW priority items for marginal gains</li>
            <li>Re-analyze your site after implementing each batch</li>
          </ol>
        </div>
      )}
    </div>
  );
}
