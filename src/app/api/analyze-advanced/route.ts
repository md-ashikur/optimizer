import { NextRequest, NextResponse } from 'next/server';
import { PerformanceMetrics, AdvancedAnalysisResult } from '@/types/performance';

interface AdvancedMLRequest {
  lcp: number;
  fid: number;
  cls: number;
  fcp?: number;
  tti?: number;
  tbt?: number;
  speed_index?: number;
  ttfb?: number;
  page_size?: number;
  num_requests?: number;
  dom_load_time?: number;
  load_time?: number;
  response_time?: number;
  total_links?: number;
  byte_size?: number;
  composite_score?: number;
}

function metricsToAdvancedMLRequest(metrics: PerformanceMetrics): AdvancedMLRequest {
  return {
    lcp: metrics.Largest_contentful_paint_LCP_ms,
    fid: metrics.Max_Potential_FID_ms,
    cls: metrics.Cumulative_Layout_Shift_CLS,
    fcp: metrics.First_Contentful_Paint_FCP_ms || 0,
    tti: metrics.Time_to_interactive_TTI_ms || 0,
    tbt: metrics.Total_Blocking_Time_TBT_ms || 0,
    speed_index: metrics.Speed_Index_ms || 0,
    ttfb: metrics.Server_Response_Time_ms || 0,
    page_size: (metrics.Total_Page_Size_KB || 0) / 1024, // Convert to MB
    num_requests: metrics.Number_of_Requests || 0,
    dom_load_time: metrics.DOM_Content_Loaded_ms || 0,
    load_time: metrics.Fully_Loaded_Time_ms || 0,
    response_time: metrics.Server_Response_Time_ms || 0,
    total_links: 0, // Not available in current metrics
    byte_size: (metrics.Total_Page_Size_KB || 0) * 1024,
    composite_score: calculateCompositeScore(metrics)
  };
}

function calculateCompositeScore(metrics: PerformanceMetrics): number {
  // Normalize and combine metrics (0-100 scale)
  const lcpScore = Math.max(0, 100 - (metrics.Largest_contentful_paint_LCP_ms / 40));
  const fidScore = Math.max(0, 100 - (metrics.Max_Potential_FID_ms / 3));
  const clsScore = Math.max(0, 100 - (metrics.Cumulative_Layout_Shift_CLS * 400));
  
  return (lcpScore * 0.4 + fidScore * 0.3 + clsScore * 0.3);
}

export async function POST(request: NextRequest) {
  try {
    console.log('Advanced ML analysis request received');
    
    const body = await request.json();
    const { metrics } = body;
    
    if (!metrics) {
      return NextResponse.json({ error: 'Metrics are required' }, { status: 400 });
    }

    // Convert metrics to format expected by advanced ML server
    const mlRequest = metricsToAdvancedMLRequest(metrics as PerformanceMetrics);
    
    // Call advanced ML server (defaults to port 8001 for advanced, 8000 for basic)
    const advancedMLUrl = process.env.NEXT_PUBLIC_ADVANCED_ML_URL || 'http://localhost:8001';
    console.log('Calling Advanced ML API:', advancedMLUrl);

    const candidates: string[] = [advancedMLUrl];
    if (advancedMLUrl.includes('localhost')) {
      candidates.push(advancedMLUrl.replace('localhost', '127.0.0.1'));
    }

    let advancedResult: AdvancedAnalysisResult | null = null;
    let lastError: string | null = null;

    for (const candidate of candidates) {
      try {
        console.log('Attempting Advanced ML service at:', candidate);
        const resp = await fetch(`${candidate}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(mlRequest),
        });

        const respText = await resp.text();
        console.log('Response status:', resp.status, 'length:', respText?.length ?? 0);

        if (!resp.ok) {
          let parsedErr: unknown = null;
          try {
            parsedErr = respText ? JSON.parse(respText) : null;
          } catch (e) {
            // ignore
          }

          const detail = (typeof parsedErr === 'object' && parsedErr !== null && 'detail' in parsedErr)
            ? (parsedErr as Record<string, unknown>)['detail']
            : respText;
          lastError = `Endpoint ${candidate} returned ${resp.status}: ${detail}`;
          console.error(lastError);
          continue;
        }

        try {
          advancedResult = respText ? JSON.parse(respText) as AdvancedAnalysisResult : null;
        } catch (err) {
          lastError = `Failed to parse JSON from ${candidate}: ${String(err)}`;
          console.error(lastError);
          continue;
        }

        console.log('Advanced ML analysis successful');
        lastError = null;
        break;
      } catch (err) {
        lastError = `Error calling ${candidate}: ${String(err)}`;
        console.error(lastError);
        continue;
      }
    }

    if (!advancedResult) {
      console.error('Failed to get advanced ML predictions');
      return NextResponse.json({ 
        error: 'Failed to call Advanced ML service', 
        detail: lastError 
      }, { status: 502 });
    }

    console.log('Sending advanced analysis result');
    return NextResponse.json(advancedResult);
    
  } catch (error) {
    console.error('Advanced analysis error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to perform advanced analysis' },
      { status: 500 }
    );
  }
}
