import axios from 'axios';
import { AnalysisResult } from '@/types/performance';

export async function analyzeWebsite(
  url: string,
  onProgress?: (progress: number, message: string) => void
): Promise<AnalysisResult> {
  try {
    // Progress: Starting
    onProgress?.(10, 'Starting analysis...');
    
    // Check ML server health first
    onProgress?.(15, 'Connecting to ML server...');
    
    let isHealthy = false;
    try {
      const healthCheck = await axios.get('http://localhost:8000/health', { timeout: 5000 });
      isHealthy = healthCheck.data.status === 'healthy';
      console.log('ML Server Health:', healthCheck.data);
    } catch (err) {
      console.error('ML Server health check failed:', err);
      throw new Error('ML server is not running. Please start it using: start_ml_server.bat');
    }

    if (!isHealthy) {
      throw new Error('ML server is unhealthy. Please restart the Python server.');
    }

    onProgress?.(20, 'ML server connected');

    // Try using Server-Sent Events for real-time progress
    try {
      console.log('Attempting streaming analysis with real-time progress...');
      const result = await analyzeWithStreaming(url, onProgress);
      return result;
    } catch (streamError) {
      console.log('Streaming failed, falling back to regular analysis:', streamError);
      // Fallback to regular analysis with simulated progress
      return await analyzeWithFallback(url, onProgress);
    }
  } catch (error) {
    console.error('Analysis error:', error);
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED') {
        throw new Error('Analysis timed out. Please try again.');
      }
      if (error.response) {
        const resp = error.response;
        const body: unknown = resp.data;

        let message = '';
        if (typeof body === 'string') {
          message = body;
        } else if (body && typeof body === 'object') {
          const b = body as Record<string, unknown>;
          const maybeError = b['error'] ?? b['detail'];
          if (typeof maybeError === 'string') {
            message = maybeError;
          } else {
            try {
              message = JSON.stringify(b);
            } catch (_) {
              message = String(maybeError ?? resp.statusText ?? resp.status);
            }
          }
        } else {
          message = String(body);
        }

        throw new Error(`${message} (status ${resp.status})`);
      }
      if (error.request) {
        throw new Error('No response from server. Is the ML server running?');
      }
    }
    throw error;
  }
}

async function analyzeWithStreaming(
  url: string,
  onProgress?: (progress: number, message: string) => void
): Promise<AnalysisResult> {
  // Streaming is currently disabled - use fallback instead
  // This ensures recommendations and issues are generated via /api/analyze route
  console.log('Streaming not available, using standard analysis');
  throw new Error('Streaming disabled, using fallback');
  
  /*
  return new Promise((resolve, reject) => {
    let resolved = false;
    const eventSource = new EventSource(
      `http://localhost:8000/predict-stream?url=${encodeURIComponent(url)}`
    );

    let lastResult: AnalysisResult | null = null;

    const cleanup = () => {
      if (eventSource.readyState !== EventSource.CLOSED) {
        eventSource.close();
      }
    };

    eventSource.onmessage = (event) => {
      if (resolved) return;
      
      try {
        const data = JSON.parse(event.data);
        
        if (data.error) {
          cleanup();
          resolved = true;
          reject(new Error(data.error));
          return;
        }

        if (data.progress !== undefined && data.message) {
          console.log(`Progress: ${data.progress}% - ${data.message}`);
          onProgress?.(data.progress, data.message);
        }

        if (data.result) {
          lastResult = data.result as AnalysisResult;
        }

        if (data.progress === 100 && lastResult) {
          cleanup();
          resolved = true;
          resolve(lastResult);
        }
      } catch (err) {
        console.error('Error parsing SSE data:', err);
      }
    };

    eventSource.onerror = (error) => {
      if (resolved) return;
      
      console.error('EventSource error:', error);
      cleanup();
      resolved = true;
      
      if (lastResult) {
        resolve(lastResult);
      } else {
        reject(new Error('Streaming connection failed'));
      }
    };

    // Timeout after 10 minutes
    const timeoutId = setTimeout(() => {
      if (resolved) return;
      
      cleanup();
      resolved = true;
      
      if (lastResult) {
        resolve(lastResult);
      } else {
        reject(new Error('Analysis timed out'));
      }
    }, 600000);

    // Cleanup timeout on early resolution
    const originalResolve = resolve;
    const originalReject = reject;
    resolve = (value: AnalysisResult) => {
      clearTimeout(timeoutId);
      cleanup();
      originalResolve(value);
    };
    reject = (reason?: unknown) => {
      clearTimeout(timeoutId);
      cleanup();
      originalReject(reason);
    };
  });
  */
}

async function analyzeWithFallback(
  url: string,
  onProgress?: (progress: number, message: string) => void
): Promise<AnalysisResult> {
  const progressMessages = [
    { progress: 25, message: 'Launching headless browser...' },
    { progress: 30, message: 'Loading website...' },
    { progress: 35, message: 'Collecting navigation timings...' },
    { progress: 40, message: 'Running Lighthouse audit...' },
    { progress: 50, message: 'Analyzing page performance...' },
    { progress: 55, message: 'Measuring Core Web Vitals...' },
    { progress: 60, message: 'Calculating LCP, FCP, CLS...' },
    { progress: 65, message: 'Checking resource loading...' },
    { progress: 70, message: 'Scanning for broken links...' },
    { progress: 75, message: 'Analyzing JavaScript execution...' },
    { progress: 80, message: 'Processing metrics data...' },
    { progress: 85, message: 'Running ML prediction...' },
  ];

  // Start the actual request to Next.js API route (which calls Python and adds recommendations/issues)
  console.log('Sending request to /api/analyze with URL:', url);
  const requestPromise = axios.post<AnalysisResult>(
    '/api/analyze',
    { url },
    { 
      timeout: 600000, // 10 minutes
      headers: { 'Content-Type': 'application/json' }
    }
  );

  // Simulate progress while waiting for response
  let progressIndex = 0;
  const progressInterval = setInterval(() => {
    if (progressIndex < progressMessages.length) {
      const { progress, message } = progressMessages[progressIndex];
      onProgress?.(progress, message);
      progressIndex++;
    }
  }, 8000);

  try {
    const response = await requestPromise;
    clearInterval(progressInterval);
    
    console.log('Analysis response:', response.data);

    onProgress?.(90, 'Processing results...');
    await new Promise(resolve => setTimeout(resolve, 500));
    onProgress?.(100, 'Analysis complete!');
    
    return response.data;
  } catch (error) {
    clearInterval(progressInterval);
    throw error;
  }
}

export async function checkMLServerHealth(): Promise<boolean> {
  try {
    const response = await axios.get('http://localhost:8000/health', { timeout: 3000 });
    return response.data.status === 'healthy' && response.data.model_loaded;
  } catch {
    return false;
  }
}
