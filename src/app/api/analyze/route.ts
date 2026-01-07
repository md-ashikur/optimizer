import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { url } = await request.json();

    if (!url) {
      return NextResponse.json(
        { error: 'URL is required' },
        { status: 400 }
      );
    }

    // Call Python ML prediction service
    const pythonApiUrl = process.env.NEXT_PUBLIC_PYTHON_API_URL || 'http://localhost:8000';
    
    const response = await fetch(`${pythonApiUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      throw new Error('ML prediction failed');
    }

    const data = await response.json();

    // Generate recommendations based on metrics and prediction
    const recommendations = generateRecommendations(data.metrics, data.prediction.label);
    const issues = identifyIssues(data.metrics);
    const score = calculateScore(data.metrics);

    return NextResponse.json({
      url,
      metrics: data.metrics,
      prediction: data.prediction,
      recommendations,
      issues,
      score,
    });
  } catch (error) {
    console.error('Analysis error:', error);
    return NextResponse.json(
      { error: 'Failed to analyze website' },
      { status: 500 }
    );
  }
}

function generateRecommendations(metrics: any, label: string): string[] {
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

function identifyIssues(metrics: any) {
  const issues: any[] = [];

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

function calculateScore(metrics: any): number {
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
