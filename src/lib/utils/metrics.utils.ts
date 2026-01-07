export function getMetricStatus(
  value: number,
  optimal: number,
  threshold: number
): 'good' | 'average' | 'poor' {
  if (value <= optimal) return 'good';
  if (value <= threshold) return 'average';
  return 'poor';
}

export function formatMetricValue(value: number, decimals: number = 0): string {
  return value.toFixed(decimals);
}

export function formatTime(ms: number): string {
  return `${(ms / 1000).toFixed(2)}s`;
}
