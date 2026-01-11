'use client';

import { OptimizationStrategy } from '@/types/performance';
import { FaBullseye, FaBolt, FaEye, FaChartBar } from 'react-icons/fa';
import { useState } from 'react';

interface OptimizationStrategyPanelProps {
  currentStrategy: string;
  onStrategyChange?: (strategy: string) => void;
}

export default function OptimizationStrategyPanel({ 
  currentStrategy,
  onStrategyChange 
}: OptimizationStrategyPanelProps) {
  const [selectedStrategy, setSelectedStrategy] = useState(currentStrategy);

  const strategies: Record<string, OptimizationStrategy> = {
    BALANCED: {
      name: 'Balanced Optimization',
      description: 'Equal weight to all metrics',
      weights: { LCP: 0.33, FID: 0.33, CLS: 0.33 },
      targets: { LCP: 2500, FID: 100, CLS: 0.1 },
      use_case: 'General-purpose websites, content sites, portfolios',
      expected_improvements: {
        LCP: 'Moderate improvement (30-50%)',
        FID: 'Moderate improvement (30-50%)',
        CLS: 'Moderate improvement (30-50%)'
      }
    },
    LCP_FOCUSED: {
      name: 'LCP-Focused Optimization',
      description: 'Prioritize loading performance',
      weights: { LCP: 0.6, FID: 0.2, CLS: 0.2 },
      targets: { LCP: 1500, FID: 150, CLS: 0.15 },
      use_case: 'Content-heavy sites, blogs, news, media sites',
      expected_improvements: {
        LCP: 'Significant improvement (60-80%)',
        FID: 'Moderate improvement (20-40%)',
        CLS: 'Moderate improvement (20-40%)'
      }
    },
    INTERACTIVITY_FOCUSED: {
      name: 'Interactivity-Focused Optimization',
      description: 'Prioritize user interaction responsiveness',
      weights: { LCP: 0.2, FID: 0.6, CLS: 0.2 },
      targets: { LCP: 3000, FID: 50, CLS: 0.15 },
      use_case: 'Web apps, SPAs, interactive platforms, dashboards',
      expected_improvements: {
        LCP: 'Moderate improvement (20-40%)',
        FID: 'Significant improvement (60-80%)',
        CLS: 'Moderate improvement (20-40%)'
      }
    },
    STABILITY_FOCUSED: {
      name: 'Stability-Focused Optimization',
      description: 'Prioritize visual stability',
      weights: { LCP: 0.2, FID: 0.2, CLS: 0.6 },
      targets: { LCP: 3000, FID: 150, CLS: 0.05 },
      use_case: 'E-commerce, checkout pages, forms, landing pages',
      expected_improvements: {
        LCP: 'Moderate improvement (20-40%)',
        FID: 'Moderate improvement (20-40%)',
        CLS: 'Significant improvement (60-80%)'
      }
    }
  };

  const strategyIcons = {
    BALANCED: FaChartBar,
    LCP_FOCUSED: FaBullseye,
    INTERACTIVITY_FOCUSED: FaBolt,
    STABILITY_FOCUSED: FaEye
  };

  const strategyColors = {
    BALANCED: 'blue',
    LCP_FOCUSED: 'green',
    INTERACTIVITY_FOCUSED: 'purple',
    STABILITY_FOCUSED: 'orange'
  };

  const handleStrategySelect = (strategyKey: string) => {
    setSelectedStrategy(strategyKey);
    onStrategyChange?.(strategyKey);
  };

  const selected = strategies[selectedStrategy] || strategies.BALANCED;
  const SelectedIcon = strategyIcons[selectedStrategy as keyof typeof strategyIcons] || FaChartBar;
  const selectedColor = strategyColors[selectedStrategy as keyof typeof strategyColors] || 'blue';

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700 animate-fadeIn card-hover">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg">
          <FaBullseye className="w-6 h-6 text-white" />
        </div>
        <div>
          <h3 className="text-xl font-bold text-gray-900 dark:text-white">
            Optimization Strategy
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Choose the best approach for your website type
          </p>
        </div>
      </div>

      {/* Strategy Selector */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
        {Object.entries(strategies).map(([key, strategy]) => {
          const Icon = strategyIcons[key as keyof typeof strategyIcons];
          const color = strategyColors[key as keyof typeof strategyColors];
          const isSelected = key === selectedStrategy;

          return (
            <button
              key={key}
              onClick={() => handleStrategySelect(key)}
              className={`p-4 rounded-xl border-2 transition-all duration-300 text-left ${
                isSelected
                  ? `border-${color}-500 bg-${color}-50 dark:bg-${color}-900/20 shadow-md`
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              <div className="flex items-start gap-3">
                <div className={`p-2 ${isSelected ? `bg-${color}-100 dark:bg-${color}-900/30` : 'bg-gray-100 dark:bg-gray-700'} rounded-lg`}>
                  <Icon className={`w-5 h-5 ${isSelected ? `text-${color}-600 dark:text-${color}-400` : 'text-gray-500'}`} />
                </div>
                <div className="flex-1">
                  <h4 className={`font-semibold mb-1 ${isSelected ? `text-${color}-900 dark:text-${color}-100` : 'text-gray-700 dark:text-gray-300'}`}>
                    {strategy.name.replace(' Optimization', '')}
                  </h4>
                  <p className={`text-xs ${isSelected ? `text-${color}-700 dark:text-${color}-300` : 'text-gray-500 dark:text-gray-400'}`}>
                    {strategy.description}
                  </p>
                </div>
                {isSelected && (
                  <div className={`flex-shrink-0 w-5 h-5 bg-${color}-500 rounded-full flex items-center justify-center`}>
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Strategy Details */}
      <div className={`p-5 bg-gradient-to-br from-${selectedColor}-50 to-${selectedColor}-100 dark:from-${selectedColor}-900/20 dark:to-${selectedColor}-900/10 rounded-xl border border-${selectedColor}-200 dark:border-${selectedColor}-800`}>
        <div className="flex items-start gap-3 mb-4">
          <SelectedIcon className={`w-6 h-6 text-${selectedColor}-600 dark:text-${selectedColor}-400 mt-1`} />
          <div>
            <h4 className={`text-lg font-bold text-${selectedColor}-900 dark:text-${selectedColor}-100 mb-1`}>
              {selected.name}
            </h4>
            <p className={`text-sm text-${selectedColor}-700 dark:text-${selectedColor}-300`}>
              {selected.description}
            </p>
          </div>
        </div>

        {/* Metric Weights */}
        <div className="mb-4">
          <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">Optimization Focus:</h5>
          <div className="space-y-2">
            {Object.entries(selected.weights).map(([metric, weight]) => (
              <div key={metric}>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="text-gray-700 dark:text-gray-300">{metric}</span>
                  <span className={`font-semibold text-${selectedColor}-700 dark:text-${selectedColor}-300`}>
                    {(weight * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`bg-gradient-to-r from-${selectedColor}-500 to-${selectedColor}-600 h-2 rounded-full transition-all duration-500`}
                    style={{ width: `${weight * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Target Metrics */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="text-center p-3 bg-white/50 dark:bg-gray-800/50 rounded-lg">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Target LCP</div>
            <div className={`text-lg font-bold text-${selectedColor}-600 dark:text-${selectedColor}-400`}>
              ≤ {selected.targets.LCP}ms
            </div>
          </div>
          <div className="text-center p-3 bg-white/50 dark:bg-gray-800/50 rounded-lg">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Target FID</div>
            <div className={`text-lg font-bold text-${selectedColor}-600 dark:text-${selectedColor}-400`}>
              ≤ {selected.targets.FID}ms
            </div>
          </div>
          <div className="text-center p-3 bg-white/50 dark:bg-gray-800/50 rounded-lg">
            <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Target CLS</div>
            <div className={`text-lg font-bold text-${selectedColor}-600 dark:text-${selectedColor}-400`}>
              ≤ {selected.targets.CLS}
            </div>
          </div>
        </div>

        {/* Use Case */}
        <div className="p-3 bg-white/70 dark:bg-gray-800/70 rounded-lg mb-4">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Best For:</div>
          <div className="text-sm font-medium text-gray-900 dark:text-white">
            {selected.use_case}
          </div>
        </div>

        {/* Expected Improvements */}
        <div>
          <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">Expected Improvements:</h5>
          <div className="space-y-1">
            {Object.entries(selected.expected_improvements).map(([metric, improvement]) => (
              <div key={metric} className="flex items-center justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">{metric}:</span>
                <span className={`font-medium text-${selectedColor}-700 dark:text-${selectedColor}-300`}>
                  {improvement}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Pareto Analysis Info */}
      <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 rounded-lg border border-purple-200 dark:border-purple-800">
        <h5 className="font-semibold text-purple-900 dark:text-purple-100 mb-2 flex items-center gap-2">
          <FaChartBar className="w-4 h-4" />
          Pareto Optimization Analysis
        </h5>
        <p className="text-sm text-purple-800 dark:text-purple-200">
          These strategies are based on Pareto frontier analysis of 1,167 websites. Only 5 sites (0.4%) achieve 
          optimal balance. Average improvement potential: <strong>91%</strong> (LCP: 99.99%, FID: 73.04%, CLS: 99.87%).
        </p>
      </div>
    </div>
  );
}
