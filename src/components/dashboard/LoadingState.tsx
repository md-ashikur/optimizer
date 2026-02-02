import { HiOutlineRefresh } from 'react-icons/hi';
import ProgressBar from './ProgressBar';

interface LoadingStateProps {
  url: string;
  progress?: number;
  message?: string;
}

export default function LoadingState({ url, progress = 0, message = 'Initializing...' }: LoadingStateProps) {
  // Debug: log when progress updates
  console.log('LoadingState render:', { progress, message });
  
  return (
    <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center z-50">
      <div className="text-center max-w-2xl w-full px-4">
        {/* Animated background effect */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute w-96 h-96 -top-48 -left-48 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute w-96 h-96 -bottom-48 -right-48 bg-pink-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        </div>
        
        {/* Content */}
        <div className="relative z-10">
          <div className="relative inline-flex mb-6">
            <div className="w-24 h-24 border-4 border-purple-500/30 rounded-full"></div>
            <div className="absolute top-0 left-0 w-24 h-24 border-4 border-t-purple-500 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"></div>
            <HiOutlineRefresh className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-10 h-10 text-purple-400 animate-pulse" />
          </div>
          
          <h2 className="text-3xl font-bold text-white mb-3">Analyzing Your Website</h2>
          <p className="text-gray-300 mb-8 break-all max-w-lg mx-auto">{url}</p>
          
          <ProgressBar progress={progress} message={message} />
          
          <div className="mt-8 space-y-3">
            <p className="text-sm text-gray-400">Please wait while we analyze performance metrics...</p>
            <div className="flex items-center justify-center gap-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-100"></div>
              <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-200"></div>
            </div>
            <p className="text-sm text-purple-300 font-semibold">{progress}% Complete</p>
          </div>
        </div>
      </div>
    </div>
  );
}
