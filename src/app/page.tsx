'use client';

import Header from '@/components/shared/Header';
import Footer from '@/components/shared/Footer';
import HeroSection from '@/components/home/HeroSection';
import HeroInput from '@/components/home/HeroInput';
import StatsSection from '@/components/home/StatsSection';
import FeaturesSection from '@/components/home/FeaturesSection';
import HowItWorksSection from '@/components/home/HowItWorksSection';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Header />

      <main className="container mx-auto px-4 py-20">
        <HeroSection />
        <HeroInput />
        <StatsSection />
        <FeaturesSection />
        <HowItWorksSection />
      </main>

      <Footer />
    </div>
  );
}

          
