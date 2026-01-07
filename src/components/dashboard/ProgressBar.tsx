interface ProgressBarProps {
  progress: number;
  message: string;
}

export default function ProgressBar({ progress, message }: ProgressBarProps) {
  return (
    <div className="w-full max-w-md">
      <div className="mb-3">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-300">{message}</span>
          <span className="text-sm font-bold text-purple-400">{progress}%</span>
        </div>
        <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500 ease-out relative overflow-hidden"
            style={{ width: `${progress}%` }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
          </div>
        </div>
      </div>
      
      {/* Progress steps */}
      <div className="flex justify-between text-xs text-gray-400 mt-3">
        <span className={progress >= 20 ? 'text-purple-400' : ''}>Connect</span>
        <span className={progress >= 40 ? 'text-purple-400' : ''}>Audit</span>
        <span className={progress >= 90 ? 'text-purple-400' : ''}>Process</span>
        <span className={progress === 100 ? 'text-green-400' : ''}>Done</span>
      </div>
    </div>
  );
}
