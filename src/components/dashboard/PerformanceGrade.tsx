import { PerformanceGrade as GradeType } from '@/types/performance';
import { getGradeConfig } from '@/lib/utils/theme.utils';

interface PerformanceGradeProps {
  grade: GradeType;
  confidence: number;
  probabilities?: {
    Good: number;
    Average: number;
    Weak: number;
  };
}

export default function PerformanceGrade({
  grade,
  confidence,
  probabilities,
}: PerformanceGradeProps) {
  const config = getGradeConfig(grade);

  return (
    <div className={`p-8 bg-white/5 backdrop-blur-sm rounded-2xl border ${config.border}`}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Performance Grade</h2>
          <p className="text-gray-400">Based on ML analysis with {(confidence * 100).toFixed(1)}% confidence</p>
        </div>
        <div className={`px-6 py-3 ${config.bg} rounded-xl border ${config.border}`}>
          <span className={`text-3xl font-bold ${config.color}`}>{grade}</span>
        </div>
      </div>

      {probabilities && (
        <div className="space-y-3">
          {Object.entries(probabilities).map(([label, prob]) => {
            const labelConfig = getGradeConfig(label as GradeType);
            return (
              <div key={label}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-300">{label}</span>
                  <span className={labelConfig.color}>{(prob * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${labelConfig.progressBg} transition-all duration-500`}
                    style={{ width: `${prob * 100}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
