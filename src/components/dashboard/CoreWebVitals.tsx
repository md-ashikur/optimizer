import { PerformanceMetrics } from '@/types/performance';
import MetricCard from './MetricCard';

interface CoreWebVitalsProps {
  metrics: PerformanceMetrics;
}

const vitalMetrics = [
  {
    key: 'Largest_contentful_paint_LCP_ms' as keyof PerformanceMetrics,
    label: 'LCP',
    description: 'Largest Contentful Paint',
    optimal: 2500,
    threshold: 4000,
    unit: 'ms',
  },
  {
    key: 'First_Contentful_Paint_FCP_ms' as keyof PerformanceMetrics,
    label: 'FCP',
    description: 'First Contentful Paint',
    optimal: 1800,
    threshold: 3000,
    unit: 'ms',
  },
  {
    key: 'Cumulative_Layout_Shift_CLS' as keyof PerformanceMetrics,
    label: 'CLS',
    description: 'Cumulative Layout Shift',
    optimal: 0.1,
    threshold: 0.25,
    unit: '',
  },
  {
    key: 'Time_to_interactive_TTI_ms' as keyof PerformanceMetrics,
    label: 'TTI',
    description: 'Time to Interactive',
    optimal: 3800,
    threshold: 7300,
    unit: 'ms',
  },
  {
    key: 'Total_Blocking_Time_TBT_ms' as keyof PerformanceMetrics,
    label: 'TBT',
    description: 'Total Blocking Time',
    optimal: 200,
    threshold: 600,
    unit: 'ms',
  },
  {
    key: 'Speed_Index_ms' as keyof PerformanceMetrics,
    label: 'Speed Index',
    description: 'Speed Index',
    optimal: 3400,
    threshold: 5800,
    unit: 'ms',
  },
];

export default function CoreWebVitals({ metrics }: CoreWebVitalsProps) {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8">
      <h2 className="text-2xl font-bold text-white mb-6">Core Web Vitals</h2>
      
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {vitalMetrics.map((metric) => (
          <MetricCard
            key={metric.key}
            label={metric.label}
            description={metric.description}
            value={metrics[metric.key] as number}
            optimal={metric.optimal}
            threshold={metric.threshold}
            unit={metric.unit}
          />
        ))}
      </div>
    </div>
  );
}
