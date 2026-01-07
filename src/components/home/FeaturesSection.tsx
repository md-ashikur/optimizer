import { HiTrendingUp, HiLightningBolt, HiShieldCheck } from 'react-icons/hi';

const features = [
  {
    icon: HiTrendingUp,
    title: 'Real-Time Analysis',
    description: 'Instant performance metrics including LCP, FCP, TTI, Speed Index, and CLS using advanced algorithms.',
    color: 'purple',
  },
  {
    icon: HiLightningBolt,
    title: 'AI Recommendations',
    description: 'Get personalized optimization suggestions based on our ML model trained on 1000+ websites.',
    color: 'pink',
  },
  {
    icon: HiShieldCheck,
    title: 'Performance Grade',
    description: 'Receive a comprehensive performance grade (Good, Average, Weak) with detailed breakdown.',
    color: 'blue',
  },
];

export default function FeaturesSection() {
  return (
    <div id="features" className="mt-32 max-w-6xl mx-auto">
      <h2 className="text-4xl font-bold text-white text-center mb-16">
        Powered by Advanced Machine Learning
      </h2>

      <div className="grid md:grid-cols-3 gap-8">
        {features.map((feature) => {
          const Icon = feature.icon;
          return (
            <div
              key={feature.title}
              className={`p-8 bg-white/5 backdrop-blur-sm rounded-2xl border border-white/10 hover:border-${feature.color}-500/50 transition`}
            >
              <div className={`w-12 h-12 bg-${feature.color}-500/20 rounded-lg flex items-center justify-center mb-4`}>
                <Icon className={`w-6 h-6 text-${feature.color}-400`} />
              </div>
              <h3 className="text-xl font-semibold text-white mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
