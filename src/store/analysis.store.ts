import { create } from 'zustand';
import { AnalysisResult } from '@/types/performance';

interface AnalysisStore {
  currentUrl: string | null;
  analysisResult: AnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;
  
  setUrl: (url: string) => void;
  setAnalysisResult: (result: AnalysisResult) => void;
  setAnalyzing: (isAnalyzing: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  currentUrl: null,
  analysisResult: null,
  isAnalyzing: false,
  error: null,
  
  setUrl: (url) => set({ currentUrl: url }),
  setAnalysisResult: (result) => set({ analysisResult: result, error: null }),
  setAnalyzing: (isAnalyzing) => set({ isAnalyzing }),
  setError: (error) => set({ error, isAnalyzing: false }),
  reset: () => set({ 
    currentUrl: null, 
    analysisResult: null, 
    isAnalyzing: false, 
    error: null 
  }),
}));
