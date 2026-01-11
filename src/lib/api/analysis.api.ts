import axios from 'axios';
import { AnalysisResult } from '@/types/performance';

export async function analyzeWebsite(
  url: string,
  onProgress?: (progress: number, message: string) => void
): Promise<AnalysisResult> {
  try {
    // Progress: Starting
    onProgress?.(10, 'Starting analysis...');
    
    // Add delay to see progress
    await new Promise(resolve => setTimeout(resolve, 300));

    // Check ML server health first
    onProgress?.(20, 'Connecting to ML server...');
    
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

    // Make the analysis request
    onProgress?.(40, 'Running performance audit...');
    console.log('Sending request to /api/analyze with URL:', url);

    // Guard: ensure we send a plain string URL (avoid Window or other circular objects)
    let payloadUrl: string;
    if (typeof url === 'string') {
      payloadUrl = url;
    } else if (url && typeof url === 'object') {
      if (typeof (url as any).href === 'string') payloadUrl = (url as any).href;
      else if (typeof (url as any).url === 'string') payloadUrl = (url as any).url;
      else {
        throw new Error('Invalid URL passed to analyzeWebsite — expected string or object with `href`/`url`.');
      }
    } else {
      throw new Error('Invalid URL passed to analyzeWebsite — expected string.');
    }

    const response = await axios.post<AnalysisResult>(
      '/api/analyze',
      { url: payloadUrl },
      {
        timeout: 90000,
        headers: { 'Content-Type': 'application/json' }
      }
    );

    console.log('Analysis response:', response.data);

    onProgress?.(90, 'Processing results...');
    
    // Simulate final processing
    await new Promise(resolve => setTimeout(resolve, 500));
    
    onProgress?.(100, 'Analysis complete!');
    
    return response.data;
  } catch (error) {
    console.error('Analysis error:', error);
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED') {
        throw new Error('Analysis timed out. Please try again.');
      }
      if (error.response) {
        const resp = error.response;
        const body = resp.data as any;
        const message = body?.error ?? body?.detail ?? (typeof body === 'string' ? body : JSON.stringify(body || {}));
        throw new Error(`${message} (status ${resp.status})`);
      }
      if (error.request) {
        throw new Error('No response from server. Is the ML server running?');
      }
    }
    throw error;
  }
}

export async function checkMLServerHealth(): Promise<boolean> {
  const candidates: string[] = [];
  const advancedUrl = process.env.NEXT_PUBLIC_ADVANCED_ML_URL;
  if (advancedUrl) candidates.push(advancedUrl);
  candidates.push('http://localhost:8000');
  // try localhost and 127.0.0.1
  candidates.push('http://127.0.0.1:8000');

  for (const base of candidates) {
    try {
      // Try /health first
      const health = await axios.get(`${base.replace(/\/$/, '')}/health`, { timeout: 2500 });
      if (health?.data && (health.data.status === 'healthy' || health.data.models_loaded || health.data.model_loaded)) {
        return true;
      }
      // Fallback to models info
      const info = await axios.get(`${base.replace(/\/$/, '')}/api/models/info`, { timeout: 2500 }).catch(() => null);
      if (info && info.data) return true;
    } catch (err) {
      // continue trying other candidates
      continue;
    }
  }

  return false;
}
