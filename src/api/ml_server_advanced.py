"""
Advanced ML API Server - Integration of all new features
Provides endpoints for:
- Ensemble predictions
- SHAP explanations  
- Regression predictions
- Intelligent recommendations
- Multi-metric optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import numpy as np
import json
from pathlib import Path
import sys

# Fix tensorflow import
try:
    from tensorflow import keras
except ImportError:
    import tensorflow.keras as keras

app = FastAPI(title="Advanced Web Performance ML API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
base_path = Path("src/ML-data")
models_path = base_path / "8_Advanced_Models"
results_path = base_path / "9_Advanced_Results"

# ============================================================================
# LOAD ALL MODELS
# ============================================================================

print("Loading models...")

# Ensemble models
voting_model = joblib.load(models_path / "ensemble/models/voting_classifier.pkl")
stacking_model = joblib.load(models_path / "ensemble/models/stacking_classifier.pkl")
ensemble_scaler = joblib.load(models_path / "ensemble/models/ensemble_scaler.pkl")
label_encoder = joblib.load(models_path / "ensemble/models/label_encoder.pkl")

# SHAP explainer
shap_explainer = joblib.load(models_path / "explainable_ai/shap_analysis/shap_explainer.pkl")

# Regression models (best ones)
lcp_regressor = joblib.load(models_path / "regression/models/LCP_gradient_boosting.pkl")
fid_regressor = joblib.load(models_path / "regression/models/FID_INP_random_forest.pkl")
cls_regressor = joblib.load(models_path / "regression/models/CLS_random_forest.pkl")
lcp_scaler = joblib.load(models_path / "regression/models/LCP_scaler.pkl")
fid_scaler = joblib.load(models_path / "regression/models/FID_INP_scaler.pkl")
cls_scaler = joblib.load(models_path / "regression/models/CLS_scaler.pkl")

# Recommendation system
rec_model = joblib.load(models_path / "recommendation/models/recommendation_scorer.pkl")
rec_scaler = joblib.load(models_path / "recommendation/models/recommendation_scaler.pkl")

with open(models_path / "recommendation/models/recommendation_rules.json", 'r') as f:
    recommendation_rules = json.load(f)

# Optimization strategies
with open(models_path / "multi_metric_optimizer/models/optimization_strategies.json", 'r') as f:
    optimization_strategies = json.load(f)

print("All models loaded successfully!")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PerformanceMetrics(BaseModel):
    lcp: float
    fid: float
    cls: float
    fcp: Optional[float] = 0
    tti: Optional[float] = 0
    tbt: Optional[float] = 0
    speed_index: Optional[float] = 0
    ttfb: Optional[float] = 0
    page_size: Optional[float] = 0
    num_requests: Optional[int] = 0
    dom_load_time: Optional[float] = 0
    load_time: Optional[float] = 0
    response_time: Optional[float] = 0
    total_links: Optional[int] = 0
    byte_size: Optional[float] = 0
    composite_score: Optional[float] = 0
    # Add remaining features as needed

class PredictionResponse(BaseModel):
    ensemble_prediction: str
    ensemble_confidence: float
    voting_prediction: str
    stacking_prediction: str
    regression_predictions: Dict[str, float]
    recommendations: Dict[str, List[str]]
    optimization_strategy: str
    shap_explanation: Dict[str, List[Dict]]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_feature_vector(metrics: PerformanceMetrics) -> np.ndarray:
    """Convert metrics to feature vector (22 features)"""
    features = [
        metrics.composite_score,
        metrics.response_time,
        metrics.dom_load_time,
        metrics.ttfb,
        metrics.total_links,
        metrics.load_time,
        metrics.num_requests,
        metrics.byte_size,
        metrics.lcp,
        metrics.page_size,
        metrics.fcp,
        metrics.tti,
        metrics.speed_index,
        metrics.cls,
        metrics.fid,
        metrics.tbt,
        0,  # placeholder features
        0,
        0,
        0,
        0,
        0
    ]
    return np.array([features])

def generate_recommendations(metrics_dict: Dict) -> Dict:
    """Generate rule-based recommendations"""
    recommendations = {
        'HIGH': [],
        'MEDIUM': [],
        'LOW': []
    }
    
    full_metrics = {
        'Largest_contentful_paint_LCP_ms': metrics_dict.get('lcp', 0),
        'Interaction_to_Next_Paint_INP_ms': metrics_dict.get('fid', 0),
        'Cumulative_Layout_Shift_CLS': metrics_dict.get('cls', 0),
        'Page_size_MB': metrics_dict.get('page_size', 0),
        'No_of_requests': metrics_dict.get('num_requests', 0),
        'Response_time_ms': metrics_dict.get('response_time', 0)
    }
    
    for category, rule in recommendation_rules.items():
        triggered = False
        for metric, condition in rule['triggers'].items():
            if metric in full_metrics:
                value = full_metrics[metric]
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '>' and value > threshold:
                    triggered = True
                    break
        
        if triggered:
            priority = rule['priority']
            recommendations[priority].extend(rule['recommendations'])
    
    # Remove duplicates
    for priority in recommendations:
        recommendations[priority] = list(set(recommendations[priority]))
    
    return recommendations

def get_optimization_strategy(site_type: str = 'general') -> str:
    """Get recommended optimization strategy"""
    strategy_map = {
        'content': 'LCP_FOCUSED',
        'blog': 'LCP_FOCUSED',
        'news': 'LCP_FOCUSED',
        'app': 'INTERACTIVITY_FOCUSED',
        'spa': 'INTERACTIVITY_FOCUSED',
        'ecommerce': 'STABILITY_FOCUSED',
        'shop': 'STABILITY_FOCUSED',
        'general': 'BALANCED'
    }
    return strategy_map.get(site_type.lower(), 'BALANCED')

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Advanced Web Performance ML API",
        "version": "2.0",
        "features": [
            "Ensemble Classification",
            "SHAP Explanations",
            "Regression Predictions",
            "Intelligent Recommendations",
            "Multi-Metric Optimization"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "ensemble": True,
            "shap": True,
            "regression": True,
            "recommendations": True,
            "optimization": True
        }
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_performance(metrics: PerformanceMetrics):
    """
    Comprehensive prediction with all advanced features
    """
    try:
        # Prepare features
        features = prepare_feature_vector(metrics)
        features_scaled = ensemble_scaler.transform(features)
        
        # 1. ENSEMBLE PREDICTIONS
        voting_pred_encoded = voting_model.predict(features_scaled)[0]
        voting_pred = label_encoder.inverse_transform([voting_pred_encoded])[0]
        voting_proba = voting_model.predict_proba(features_scaled)[0]
        
        stacking_pred_encoded = stacking_model.predict(features_scaled)[0]
        stacking_pred = label_encoder.inverse_transform([stacking_pred_encoded])[0]
        
        ensemble_confidence = float(max(voting_proba))
        
        # 2. REGRESSION PREDICTIONS
        reg_features = features[:, :21]  # First 21 features
        
        lcp_scaled = lcp_scaler.transform(reg_features)
        lcp_prediction = float(lcp_regressor.predict(lcp_scaled)[0])
        
        fid_scaled = fid_scaler.transform(reg_features)
        fid_prediction = float(fid_regressor.predict(fid_scaled)[0])
        
        cls_scaled = cls_scaler.transform(reg_features)
        cls_prediction = float(cls_regressor.predict(cls_scaled)[0])
        
        regression_predictions = {
            "predicted_lcp_ms": round(lcp_prediction, 2),
            "predicted_fid_ms": round(fid_prediction, 2),
            "predicted_cls": round(cls_prediction, 4)
        }
        
        # 3. RECOMMENDATIONS
        metrics_dict = metrics.dict()
        recommendations = generate_recommendations(metrics_dict)
        
        # 4. OPTIMIZATION STRATEGY
        strategy = get_optimization_strategy('general')
        
        # 5. SHAP EXPLANATION (top features)
        import pandas as pd
        feature_names = [
            'composite_score', 'response_time', 'dom_load_time', 'ttfb', 
            'total_links', 'load_time', 'num_requests', 'byte_size',
            'lcp', 'page_size', 'fcp', 'tti', 'speed_index', 'cls', 
            'fid', 'tbt', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21'
        ]
        
        features_df = pd.DataFrame(features_scaled, columns=feature_names)
        shap_values = shap_explainer(features_df)
        
        # Get top 5 most impactful features
        shap_impacts = []
        for i, (fname, fval, shap_val) in enumerate(zip(feature_names, features[0], shap_values.values[0])):
            shap_impacts.append({
                'feature': fname,
                'value': float(fval),
                'shap_value': float(shap_val),
                'impact': abs(float(shap_val))
            })
        
        shap_impacts_sorted = sorted(shap_impacts, key=lambda x: x['impact'], reverse=True)[:5]
        
        shap_explanation = {
            'top_positive': [s for s in shap_impacts_sorted if s['shap_value'] > 0][:3],
            'top_negative': [s for s in shap_impacts_sorted if s['shap_value'] < 0][:3],
            'most_important': shap_impacts_sorted[:5]
        }
        
        return PredictionResponse(
            ensemble_prediction=voting_pred,
            ensemble_confidence=ensemble_confidence,
            voting_prediction=voting_pred,
            stacking_prediction=stacking_pred,
            regression_predictions=regression_predictions,
            recommendations=recommendations,
            optimization_strategy=strategy,
            shap_explanation=shap_explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations")
async def get_recommendations_only(metrics: PerformanceMetrics):
    """Get only recommendations"""
    try:
        metrics_dict = metrics.dict()
        recommendations = generate_recommendations(metrics_dict)
        
        return {
            "recommendations": recommendations,
            "total": sum(len(v) for v in recommendations.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/regression")
async def regression_predictions(metrics: PerformanceMetrics):
    """Get exact metric predictions"""
    try:
        features = prepare_feature_vector(metrics)
        reg_features = features[:, :21]
        
        lcp_scaled = lcp_scaler.transform(reg_features)
        lcp_prediction = float(lcp_regressor.predict(lcp_scaled)[0])
        
        fid_scaled = fid_scaler.transform(reg_features)
        fid_prediction = float(fid_regressor.predict(fid_scaled)[0])
        
        cls_scaled = cls_scaler.transform(reg_features)
        cls_prediction = float(cls_regressor.predict(cls_scaled)[0])
        
        return {
            "predicted_lcp_ms": round(lcp_prediction, 2),
            "predicted_fid_ms": round(fid_prediction, 2),
            "predicted_cls": round(cls_prediction, 4),
            "confidence": "Based on Gradient Boosting (LCP), Random Forest (FID, CLS)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/strategies")
async def get_optimization_strategies():
    """Get all optimization strategies"""
    return optimization_strategies

@app.get("/api/models/info")
async def models_info():
    """Get information about loaded models"""
    return {
        "ensemble": {
            "voting_classifier": "Soft voting (RF + LightGBM)",
            "stacking_classifier": "Stacking with LightGBM meta-learner",
            "accuracy": "100%"
        },
        "regression": {
            "lcp": "Gradient Boosting Regressor (R²=0.81)",
            "fid": "Random Forest Regressor (R²=0.63)",
            "cls": "Random Forest Regressor (R²=0.97)"
        },
        "recommendations": {
            "categories": 5,
            "total_recommendations": 25,
            "ml_scorer": "Random Forest Classifier"
        },
        "explainability": {
            "method": "SHAP (SHapley Additive exPlanations)",
            "features_analyzed": 22
        },
        "optimization": {
            "strategies": 4,
            "pareto_points": 5
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("ADVANCED WEB PERFORMANCE ML API")
    print("="*80)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\n" + "="*80)
    uvicorn.run(app, host="0.0.0.0", port=8000)
