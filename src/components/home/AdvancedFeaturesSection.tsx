'use client';

import { FaBrain, FaChartLine, FaLightbulb, FaLayerGroup, FaCog } from 'react-icons/fa';
import { HiSparkles, HiTrendingUp } from 'react-icons/hi';
import { RiRobotFill } from 'react-icons/ri';

export default function AdvancedFeaturesSection() {
  const features = [
    {
      icon: FaLayerGroup,
      title: 'Ensemble Classification',
      description: '100% accuracy performance grading using 5 advanced ML models',
      gradient: 'from-green-500 to-emerald-600',
      stats: '5 Models',
      badge: '100% Accurate'
    },
    {
      icon: FaBrain,
      title: 'SHAP Explainability',
      description: 'Understand exactly why your site performs the way it does',
      gradient: 'from-purple-500 to-violet-600',
      stats: '22 Features',
      badge: 'AI-Powered'
    },
    {
      icon: FaChartLine,
      title: 'Regression Predictions',
      description: 'See exact predicted metrics after optimization implementation',
      gradient: 'from-blue-500 to-cyan-600',
      stats: 'R² 0.81+',
      badge: 'Precise'
    },
    {
      icon: FaLightbulb,
      title: 'Smart Recommendations',
      description: 'ML-generated prioritized optimization strategies tailored to your site',
      gradient: 'from-pink-500 to-rose-600',
      stats: '25+ Tips',
      badge: 'Intelligent'
    },
    {
      icon: FaCog,
      title: 'Optimization Strategies',
      description: 'Choose from 4 data-driven approaches based on 1,167 real websites',
      gradient: 'from-indigo-500 to-purple-600',
      stats: '4 Strategies',
      badge: 'Pareto-Optimal'
    }
  ];

  return (
    <section className="py-20">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16 animate-fadeIn">
          <div className="flex items-center justify-center gap-3 mb-4">
            <RiRobotFill className="w-12 h-12 text-purple-400" />
            <h2 className="text-4xl md:text-5xl font-bold text-white">
              Advanced ML Features
            </h2>
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Powered by cutting-edge machine learning models trained on real-world performance data
          </p>
          <div className="flex items-center justify-center gap-2 mt-4">
            <HiSparkles className="w-5 h-5 text-yellow-400" />
            <span className="text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              5 Advanced AI Features Included
            </span>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="group relative bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all duration-300 card-hover animate-scaleIn"
              style={{ animationDelay: `${idx * 0.1}s` }}
            >
              {/* Badge */}
              <div className="absolute top-4 right-4">
                <span className="px-3 py-1 bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-400/30 rounded-full text-xs font-semibold text-purple-300">
                  {feature.badge}
                </span>
              </div>

              {/* Icon */}
              <div className={`w-14 h-14 bg-gradient-to-br ${feature.gradient} rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="w-7 h-7 text-white" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-bold text-white mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-400 text-sm mb-4">
                {feature.description}
              </p>

              {/* Stats */}
              <div className="flex items-center gap-2">
                <HiTrendingUp className="w-4 h-4 text-emerald-400" />
                <span className="text-sm font-semibold text-emerald-400">
                  {feature.stats}
                </span>
              </div>
            </div>
          ))}

          {/* CTA Card */}
          <div className="group relative bg-gradient-to-br from-purple-600/20 to-pink-600/20 border-2 border-purple-400/30 rounded-2xl p-6 flex flex-col items-center justify-center text-center animate-scaleIn hover:from-purple-600/30 hover:to-pink-600/30 transition-all duration-300" style={{ animationDelay: '0.5s' }}>
            <HiSparkles className="w-12 h-12 text-purple-400 mb-4 group-hover:scale-110 transition-transform" />
            <h3 className="text-xl font-bold text-white mb-2">
              Try All Features Now
            </h3>
            <p className="text-gray-300 text-sm mb-4">
              Get instant access to advanced ML-powered performance analysis
            </p>
            <button className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-smooth">
              Start Analysis
            </button>
          </div>
        </div>

        {/* Performance Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 p-8 bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl animate-fadeIn">
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-emerald-500 mb-2">
              100%
            </div>
            <div className="text-sm text-gray-400">Classification Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-500 mb-2">
              0.81+
            </div>
            <div className="text-sm text-gray-400">Avg R² Score</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-violet-500 mb-2">
              1,167
            </div>
            <div className="text-sm text-gray-400">Websites Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-rose-500 mb-2">
              25+
            </div>
            <div className="text-sm text-gray-400">ML Recommendations</div>
          </div>
        </div>

        {/* Pareto Analysis Banner */}
        <div className="mt-8 p-6 bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border border-indigo-400/30 rounded-xl animate-fadeIn">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-indigo-500/20 rounded-lg">
              <FaBrain className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <h4 className="text-lg font-bold text-white mb-2">
                Powered by Pareto Optimization Analysis
              </h4>
              <p className="text-gray-300 text-sm">
                Our ML models are trained on comprehensive analysis of 1,167 real-world websites. 
                We identified that only 0.4% achieve optimal performance balance, while 91% have 
                significant improvement potential. Our strategies are designed to move your site 
                toward the Pareto frontier with an average improvement of 45-70% across Core Web Vitals.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
