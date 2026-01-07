import { HiOutlineRefresh } from 'react-icons/hi';
import ProgressBar from './ProgressBar';

interface LoadingStateProps {
  url: string;
  progress?: number;
  message?: string;
}

export default function LoadingState({ url, progress = 0, message = 'Initializing...' }: LoadingStateProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
      <div className="text-center max-w-2xl px-4">
        <div className="relative inline-flex mb-6">
          <div className="w-20 h-20 border-4 border-purple-500/30 rounded-full"></div>
          <div className="absolute top-0 left-0 w-20 h-20 border-4 border-t-purple-500 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"></div>
          <HiOutlineRefresh className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 text-purple-400" />
        </div>
        
        <h2 className="text-2xl font-bold text-white mb-2">Analyzing Your Website</h2>
        <p className="text-gray-400 mb-8 break-all">{url}</p>
        
        <ProgressBar progress={progress} message={message} />
        
        <p className="text-xs text-gray-500 mt-6">Please wait while we analyze performance metrics...</p>
      </div>
    </div>
  );
}
