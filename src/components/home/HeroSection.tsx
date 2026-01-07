import { HiShieldCheck } from 'react-icons/hi';

export default function HeroSection() {
  return (
    <div className="max-w-4xl mx-auto text-center">
      <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/20 rounded-full border border-purple-500/30 mb-6">
        <HiShieldCheck className="w-4 h-4 text-purple-400" />
        <span className="text-sm text-purple-300">AI-Powered Performance Analysis</span>
      </div>

      <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
        Optimize Your Website
        <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
          {' '}Performance
        </span>
      </h1>

      <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
        Get instant, AI-powered insights into your website's performance.
        Discover bottlenecks, receive actionable recommendations, and improve your site's speed.
      </p>
    </div>
  );
}
