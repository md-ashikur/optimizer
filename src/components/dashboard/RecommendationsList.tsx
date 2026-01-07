import { HiCheckCircle } from 'react-icons/hi';

interface RecommendationsListProps {
  recommendations: string[];
}

export default function RecommendationsList({ recommendations }: RecommendationsListProps) {
  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 p-8">
      <h2 className="text-2xl font-bold text-white mb-6">Recommendations</h2>
      
      <div className="space-y-3">
        {recommendations.map((recommendation, index) => (
          <div
            key={index}
            className="flex items-start gap-3 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20"
          >
            <HiCheckCircle className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <p className="text-gray-300 text-sm">{recommendation}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
