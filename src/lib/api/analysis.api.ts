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
    
    const response = await axios.post<AnalysisResult>(
      '/api/analyze',
      { url },
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
        throw new Error(error.response.data?.error || error.message || 'Analysis failed');
      }
      if (error.request) {
        throw new Error('No response from server. Is the ML server running?');
      }
    }
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
