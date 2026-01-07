import { PerformanceIssue } from '@/types/performance';
import { getSeverityConfig } from '@/lib/utils/theme.utils';
import { HiExclamationCircle, HiInformationCircle } from 'react-icons/hi';

interface IssuesListProps {
  issues: PerformanceIssue[];
}

export default function IssuesList({ issues }: IssuesListProps) {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8">
      <h2 className="text-2xl font-bold text-white mb-6">Issues Detected</h2>
      
      <div className="space-y-4">
        {issues.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <HiInformationCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>No critical issues detected</p>
          </div>
        ) : (
          issues.map((issue, index) => {
            const config = getSeverityConfig(issue.severity);
            return (
              <div
                key={index}
                className={`p-4 ${config.bg} rounded-xl border ${config.border}`}
              >
                <div className="flex items-start gap-3">
                  <HiExclamationCircle className={`w-5 h-5 ${config.badge.split(' ')[0].replace('bg-', 'text-')} flex-shrink-0 mt-0.5`} />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`text-xs px-2 py-1 rounded-full ${config.badge} uppercase font-semibold`}>
                        {issue.severity}
                      </span>
                      <span className="text-white font-semibold">{issue.metric}</span>
                    </div>
                    <p className="text-gray-300 text-sm mb-2">{issue.message}</p>
                    <p className="text-gray-400 text-xs italic">{issue.suggestion}</p>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
