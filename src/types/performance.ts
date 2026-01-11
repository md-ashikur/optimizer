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

// SHAP Explanation Types
export interface ShapFeatureImpact {
  feature: string;
  value: number;
  shap_value: number;
  impact: number;
}

export interface ShapExplanation {
  top_positive: ShapFeatureImpact[];
  top_negative: ShapFeatureImpact[];
  most_important: ShapFeatureImpact[];
}

// Regression Predictions Types
export interface RegressionPredictions {
  predicted_lcp_ms: number;
  predicted_fid_ms: number;
  predicted_cls: number;
  confidence?: string;
}

// Recommendations Types
export interface CategorizedRecommendations {
  HIGH: string[];
  MEDIUM: string[];
  LOW: string[];
}

// Optimization Strategy Types
export interface OptimizationStrategy {
  name: string;
  description: string;
  weights: {
    LCP: number;
    FID: number;
    CLS: number;
  };
  targets: {
    LCP: number;
    FID: number;
    CLS: number;
  };
  use_case: string;
  expected_improvements: {
    LCP: string;
    FID: string;
    CLS: string;
  };
}

// Advanced Analysis Result (includes all 5 ML features)
export interface AdvancedAnalysisResult {
  // Ensemble Classification
  ensemble_prediction: string;
  ensemble_confidence: number;
  voting_prediction: string;
  stacking_prediction: string;
  
  // Regression Predictions
  regression_predictions: RegressionPredictions;
  
  // Intelligent Recommendations
  recommendations: CategorizedRecommendations;
  
  // Optimization Strategy
  optimization_strategy: string;
  
  // SHAP Explanations
  shap_explanation: ShapExplanation;
}

// Complete Analysis Result (combines basic + advanced)
export interface AnalysisResult {
  url: string;
  metrics: PerformanceMetrics;
  prediction: PredictionResult;
  recommendations: string[];
  issues: PerformanceIssue[];
  score: number;
  
  // Advanced ML Features (optional for backward compatibility)
  advanced?: AdvancedAnalysisResult;
}
