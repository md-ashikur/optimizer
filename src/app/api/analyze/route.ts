import { NextRequest, NextResponse } from 'next/server';
import { PerformanceMetrics, PerformanceIssue } from '@/types/performance';

type Prediction = {
  label: string;
  confidence?: number;
  [key: string]: unknown;
};

type PythonRaw = {
  metrics?: PerformanceMetrics;
  prediction?: Prediction;
  [key: string]: unknown;
};

function isPythonResponse(obj: unknown): obj is PythonRaw {
  return typeof obj === 'object' && obj !== null && 'metrics' in (obj as object) && 'prediction' in (obj as object);
}

export async function POST(request: NextRequest) {
  try {
    console.log('Received analysis request');
    let url: string | undefined;
    // Read the raw body once and parse manually to avoid "body already read" errors
    const rawBody = await request.text();
    try {
      const parsed = rawBody ? JSON.parse(rawBody) : {};
      url = parsed?.url;
      console.log('URL to analyze:', url);
      if (!url) {
        return NextResponse.json({ error: 'URL is required' }, { status: 400 });
      }
    } catch (err) {
      console.error('Failed to parse JSON body:', err, 'raw body:', rawBody);
      return NextResponse.json({ error: 'Invalid JSON in request body', raw: rawBody }, { status: 400 });
    }

    // Call Python ML prediction service with fallbacks (localhost -> 127.0.0.1)
    const pythonApiUrl = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://localhost:8000';
    console.log('Calling Python API, primary URL:', pythonApiUrl);

    const candidates: string[] = [pythonApiUrl];
    if (pythonApiUrl.includes('localhost')) {
      candidates.push(pythonApiUrl.replace('localhost', '127.0.0.1'));
    }

    let data: unknown = null;
    let lastError: string | null = null;
    for (const candidate of candidates) {
      try {
        console.log('Attempting Python service at:', candidate);

        // Try /predict first (ml_server.py basic), then /api/predict (ml_server_advanced.py)
        const endpoints = [
          `${candidate.replace(/\/$/, '')}/predict`
        ];

        let respText: string | null = null;
        let parsed: unknown = null;
        let success = false;

        for (const ep of endpoints) {
          try {
            console.log('Trying endpoint:', ep);
            const r = await fetch(ep, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ url }),
            });

            const text = await r.text();
            console.log('Response from', ep, 'status:', r.status, 'length:', text?.length ?? 0);

            if (!r.ok) {
              console.error(`Endpoint ${ep} returned ${r.status}: ${text}`);
              continue; // try next endpoint
            }

            // parse JSON
            try {
              parsed = text ? JSON.parse(text) : {};
              respText = text;
              success = true;
              break;
            } catch (err) {
              console.error(`Failed to parse JSON from ${ep}:`, err);
              continue; // try next endpoint
            }
          } catch (e) {
            console.error('Request to', ep, 'failed:', e);
            continue; // try next endpoint
          }
        }

        if (!success) {
          lastError = `All endpoints for ${candidate} failed or returned non-JSON responses`;
          console.error(lastError);
          continue; // try next candidate
        }

        data = parsed;
        const keys = typeof data === 'object' && data !== null ? Object.keys(data as Record<string, unknown>) : [];
        console.log('Received prediction data keys from', candidate, ':', keys);
        lastError = null;
        break; // success
      } catch (err) {
        lastError = `Error calling ${candidate}: ${String(err)}`;
        console.error(lastError);
        continue;
      }
    }

    if (!data) {
      return NextResponse.json({ error: 'Failed to call Python service', detail: lastError }, { status: 502 });
    }

    // Generate recommendations based on metrics and prediction
    if (!isPythonResponse(data)) {
      console.error('Python response missing expected fields or has unexpected shape:', data);
      return NextResponse.json({ error: 'Python response missing metrics or prediction', data }, { status: 502 });
    }

    const metrics = data.metrics as PerformanceMetrics;
    const prediction = data.prediction as Prediction;

    let recommendations: string[] = [];
    let issues = [];
    let score = 0;
    try {
      recommendations = generateRecommendations(metrics, prediction.label);
      issues = identifyIssues(metrics);
      score = calculateScore(metrics);
    } catch (err) {
      console.error('Error while processing prediction data:', err);
      return NextResponse.json({ error: 'Failed to process prediction data', detail: String(err) }, { status: 500 });
    }

    const result = {
      url,
      metrics: data.metrics,
      prediction: data.prediction,
      recommendations,
      issues,
      score,
    };

    console.log('Sending final result');
    return NextResponse.json(result);
  } catch (error) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to analyze website' },
      { status: 500 }
    );
  }
}

function generateRecommendations(metrics: PerformanceMetrics, label: string): string[] {
  const recommendations: string[] = [];

  if (metrics.Largest_contentful_paint_LCP_ms > 2500) {
    recommendations.push(
      'Optimize your Largest Contentful Paint (LCP) by reducing server response time, using a CDN, and optimizing images.'
    );
  }

  if (metrics.First_Contentful_Paint_FCP_ms > 1800) {
    recommendations.push(
      'Improve First Contentful Paint (FCP) by eliminating render-blocking resources and minifying CSS.'
    );
  }

  if (metrics.Time_to_interactive_TTI_ms > 3800) {
    recommendations.push(
      'Reduce Time to Interactive (TTI) by minimizing JavaScript execution time and removing unused code.'
    );
  }

  if (metrics.Cumulative_Layout_Shift_CLS > 0.1) {
    recommendations.push(
      'Fix Cumulative Layout Shift (CLS) by setting size attributes on images and videos, and avoiding inserting content above existing content.'
    );
  }

  if (metrics.Total_Blocking_Time_TBT_ms > 300) {
    recommendations.push(
      'Decrease Total Blocking Time (TBT) by breaking up long tasks and optimizing your JavaScript.'
    );
  }

  if (metrics.Speed_Index_ms > 3400) {
    recommendations.push(
      'Improve Speed Index by optimizing content visibility and reducing main-thread work.'
    );
  }

  if (label === 'Weak') {
    recommendations.push(
      'Consider implementing lazy loading for images and using modern image formats like WebP or AVIF.'
    );
    recommendations.push(
      'Enable compression (gzip or brotli) for text-based resources.'
    );
  }

  if (label === 'Average') {
    recommendations.push(
      'Implement resource hints (preload, prefetch) for critical assets.'
    );
  }

  if (recommendations.length === 0) {
    recommendations.push(
      'Your website performance is excellent! Continue monitoring and maintaining these metrics.'
    );
  }

  return recommendations;
}

function identifyIssues(metrics: PerformanceMetrics): PerformanceIssue[] {
  const issues: PerformanceIssue[] = [];

  // High severity issues
  if (metrics.Largest_contentful_paint_LCP_ms > 4000) {
    issues.push({
      severity: 'high',
      metric: 'Largest Contentful Paint',
      message: `Your LCP is ${(metrics.Largest_contentful_paint_LCP_ms / 1000).toFixed(2)}s, which is significantly slower than the recommended 2.5s threshold.`,
      suggestion: 'Optimize server response times, implement CDN, compress images, and use lazy loading for offscreen content.'
    });
  } else if (metrics.Largest_contentful_paint_LCP_ms > 2500) {
    issues.push({
      severity: 'medium',
      metric: 'Largest Contentful Paint',
      message: `Your LCP is ${(metrics.Largest_contentful_paint_LCP_ms / 1000).toFixed(2)}s, exceeding the 2.5s threshold.`,
      suggestion: 'Consider optimizing your largest image or text block, and ensure critical resources are loaded quickly.'
    });
  }

  if (metrics.Time_to_interactive_TTI_ms > 5000) {
    issues.push({
      severity: 'high',
      metric: 'Time to Interactive',
      message: `Your TTI is ${(metrics.Time_to_interactive_TTI_ms / 1000).toFixed(2)}s, which may frustrate users.`,
      suggestion: 'Reduce JavaScript execution time, split code bundles, and defer non-critical scripts.'
    });
  }

  if (metrics.Cumulative_Layout_Shift_CLS > 0.25) {
    issues.push({
      severity: 'high',
      metric: 'Cumulative Layout Shift',
      message: `Your CLS score of ${metrics.Cumulative_Layout_Shift_CLS.toFixed(3)} indicates significant visual instability.`,
      suggestion: 'Set explicit dimensions for images and embeds, avoid inserting content above existing content, and use transform animations.'
    });
  } else if (metrics.Cumulative_Layout_Shift_CLS > 0.1) {
    issues.push({
      severity: 'medium',
      metric: 'Cumulative Layout Shift',
      message: `Your CLS score of ${metrics.Cumulative_Layout_Shift_CLS.toFixed(3)} needs improvement.`,
      suggestion: 'Reserve space for ads and embeds, and ensure web fonts load without causing layout shifts.'
    });
  }

  if (metrics.Total_Blocking_Time_TBT_ms > 600) {
    issues.push({
      severity: 'medium',
      metric: 'Total Blocking Time',
      message: `Your TBT of ${metrics.Total_Blocking_Time_TBT_ms}ms indicates significant main thread blocking.`,
      suggestion: 'Break up long tasks, optimize third-party scripts, and consider code splitting.'
    });
  }

  if (metrics.First_Contentful_Paint_FCP_ms > 3000) {
    issues.push({
      severity: 'medium',
      metric: 'First Contentful Paint',
      message: `Your FCP is ${(metrics.First_Contentful_Paint_FCP_ms / 1000).toFixed(2)}s, users see blank page for too long.`,
      suggestion: 'Eliminate render-blocking resources, inline critical CSS, and optimize font loading.'
    });
  }

  // Low severity issues
  if (metrics.Speed_Index_ms > 3400 && metrics.Speed_Index_ms <= 5800) {
    issues.push({
      severity: 'low',
      metric: 'Speed Index',
      message: `Your Speed Index of ${(metrics.Speed_Index_ms / 1000).toFixed(2)}s could be improved.`,
      suggestion: 'Optimize content visibility progression and reduce main thread work.'
    });
  }

  if (issues.length === 0) {
    issues.push({
      severity: 'low',
      metric: 'Overall Performance',
      message: 'Your website is performing well across all core metrics.',
      suggestion: 'Continue monitoring performance and consider implementing advanced optimizations like HTTP/3 or edge caching.'
    });
  }

  return issues;
}

function calculateScore(metrics: PerformanceMetrics): number {
  // Weighted scoring based on Core Web Vitals importance
  let score = 100;

  // LCP (30% weight)
  if (metrics.Largest_contentful_paint_LCP_ms > 4000) score -= 30;
  else if (metrics.Largest_contentful_paint_LCP_ms > 2500) score -= 15;
  else if (metrics.Largest_contentful_paint_LCP_ms > 1200) score -= 5;

  // FCP (15% weight)
  if (metrics.First_Contentful_Paint_FCP_ms > 3000) score -= 15;
  else if (metrics.First_Contentful_Paint_FCP_ms > 1800) score -= 8;
  else if (metrics.First_Contentful_Paint_FCP_ms > 1000) score -= 3;

  // TTI (25% weight)
  if (metrics.Time_to_interactive_TTI_ms > 7300) score -= 25;
  else if (metrics.Time_to_interactive_TTI_ms > 3800) score -= 12;
  else if (metrics.Time_to_interactive_TTI_ms > 2000) score -= 5;

  // CLS (20% weight)
  if (metrics.Cumulative_Layout_Shift_CLS > 0.25) score -= 20;
  else if (metrics.Cumulative_Layout_Shift_CLS > 0.1) score -= 10;
  else if (metrics.Cumulative_Layout_Shift_CLS > 0.05) score -= 3;

  // TBT (10% weight)
  if (metrics.Total_Blocking_Time_TBT_ms > 600) score -= 10;
  else if (metrics.Total_Blocking_Time_TBT_ms > 300) score -= 5;

  return Math.max(0, Math.min(100, score));
}
