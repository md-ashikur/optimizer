import { PerformanceGrade, IssueSeverity } from '@/types/performance';

export const gradeConfig = {
  Good: {
    color: 'text-green-400',
    bg: 'bg-green-500/20',
    border: 'border-green-500/30',
    progressBg: 'bg-green-500',
  },
  Average: {
    color: 'text-yellow-400',
    bg: 'bg-yellow-500/20',
    border: 'border-yellow-500/30',
    progressBg: 'bg-yellow-500',
  },
  Weak: {
    color: 'text-red-400',
    bg: 'bg-red-500/20',
    border: 'border-red-500/30',
    progressBg: 'bg-red-500',
  },
};

export const severityConfig = {
  high: {
    badge: 'bg-red-500 text-white',
    border: 'border-red-500/50',
    bg: 'bg-red-500/10',
  },
  medium: {
    badge: 'bg-yellow-500 text-black',
    border: 'border-yellow-500/50',
    bg: 'bg-yellow-500/10',
  },
  low: {
    badge: 'bg-blue-500 text-white',
    border: 'border-blue-500/50',
    bg: 'bg-blue-500/10',
  },
};

export function getGradeConfig(grade: PerformanceGrade) {
  return gradeConfig[grade];
}

export function getSeverityConfig(severity: IssueSeverity) {
  return severityConfig[severity];
}
