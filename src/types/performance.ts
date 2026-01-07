export interface PerformanceMetrics {
  Largest_contentful_paint_LCP_ms: number;
  First_Contentful_Paint_FCP_ms: number;
  Time_to_interactive_TTI_ms: number;
  Speed_Index_ms: number;
  Total_Blocking_Time_TBT_ms: number;
  Cumulative_Layout_Shift_CLS: number;
  Max_Potential_FID_ms: number;
  Server_Response_Time_ms: number;
  DOM_Content_Loaded_ms?: number;
  First_Meaningful_Paint_ms?: number;
  Fully_Loaded_Time_ms?: number;
  Total_Page_Size_KB?: number;
  Number_of_Requests?: number;
  JavaScript_Size_KB?: number;
  CSS_Size_KB?: number;
  Image_Size_KB?: number;
  Font_Size_KB?: number;
  HTML_Size_KB?: number;
  Main_Thread_Work_ms?: number;
  Bootup_Time_ms?: number;
  Offscreen_Images_KB?: number;
}

export type PerformanceGrade = 'Good' | 'Average' | 'Weak';

export type IssueSeverity = 'high' | 'medium' | 'low';

export interface PerformanceIssue {
  severity: IssueSeverity;
  metric: string;
  message: string;
  suggestion: string;
}

export interface PredictionResult {
  label: PerformanceGrade;
  confidence: number;
  probabilities?: {
    Good: number;
    Average: number;
    Weak: number;
  };
}

export interface AnalysisResult {
  url: string;
  metrics: PerformanceMetrics;
  prediction: PredictionResult;
  recommendations: string[];
  issues: PerformanceIssue[];
  score: number;
}
