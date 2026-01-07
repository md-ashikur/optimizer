import { getMetricStatus } from '@/lib/utils/metrics.utils';

interface MetricCardProps {
  label: string;
  description: string;
  value: number;
  optimal: number;
  threshold: number;
  unit: string;
}

const statusConfig = {
  good: {
    bg: 'bg-green-500/10',
    border: 'border-green-500/30',
    text: 'text-green-400',
    label: 'Good',
  },
  average: {
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    text: 'text-yellow-400',
    label: 'Needs Improvement',
  },
  poor: {
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    text: 'text-red-400',
    label: 'Poor',
  },
};

export default function MetricCard({
  label,
  description,
  value,
  optimal,
  threshold,
  unit,
}: MetricCardProps) {
  const status = getMetricStatus(value, optimal, threshold);
  const config = statusConfig[status];

  return (
    <div className={`p-4 ${config.bg} rounded-xl border ${config.border}`}>
      <div className="flex justify-between items-start mb-2">
        <div>
          <h3 className="font-bold text-white">{label}</h3>
          <p className="text-xs text-gray-400">{description}</p>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${config.bg} ${config.text}`}>
          {config.label}
        </span>
      </div>
      <div className={`text-2xl font-bold ${config.text}`}>
        {value.toFixed(value < 1 ? 3 : 0)}
        <span className="text-sm ml-1">{unit}</span>
      </div>
    </div>
  );
}
