import { HiLightningBolt } from 'react-icons/hi';

export default function Header() {
  return (
    <header className="container mx-auto px-4 py-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <HiLightningBolt className="w-8 h-8 text-purple-400" />
          <span className="text-2xl font-bold text-white">WebOptimizer AI</span>
        </div>
        <nav className="hidden md:flex gap-6">
          <a href="#features" className="text-gray-300 hover:text-white transition">
            Features
          </a>
          <a href="#how-it-works" className="text-gray-300 hover:text-white transition">
            How It Works
          </a>
          <a href="#stats" className="text-gray-300 hover:text-white transition">
            Statistics
          </a>
        </nav>
      </div>
    </header>
  );
}
