'use client';

import { useState } from 'react';
import { HiArrowRight } from 'react-icons/hi2';
import { useRouter } from 'next/navigation';
import { useAnalysisStore } from '@/store/analysis.store';

export default function HeroInput() {
  const [url, setUrl] = useState('');
  const router = useRouter();
  const { setUrl: setStoreUrl } = useAnalysisStore();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!url) return;

    setStoreUrl(url);
    // Do not set analyzing here — dashboard should start analysis on mount
    router.push('/dashboard');
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-2xl mx-auto mb-16">
      <div className="flex flex-col sm:flex-row gap-4 p-2 bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20">
        <input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter your website URL (e.g., https://example.com)"
          className="flex-1 px-6 py-4 bg-transparent text-white placeholder-gray-400 outline-none"
          required
        />
        <button
          type="submit"
          className="px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold rounded-xl transition-all transform hover:scale-105 flex items-center justify-center gap-2 whitespace-nowrap"
        >
          Analyze Now
          <HiArrowRight className="w-5 h-5" />
        </button>
      </div>
    </form>
  );
}
