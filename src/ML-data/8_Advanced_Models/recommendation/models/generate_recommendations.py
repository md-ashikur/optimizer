
"""
Recommendation Generator
Use this to generate personalized recommendations for any website
"""

import joblib
import json
import numpy as np
from pathlib import Path

# Load models and rules
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
model_path = base_path / "8_Advanced_Models" / "recommendation" / "models"

rec_model = joblib.load(model_path / "recommendation_scorer.pkl")
scaler = joblib.load(model_path / "recommendation_scaler.pkl")

with open(model_path / "recommendation_rules.json", 'r') as f:
    rules = json.load(f)

def get_recommendations(metrics):
    """
    Generate personalized recommendations
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with keys: 'lcp', 'fid', 'cls', 'page_size', 
        'num_requests', 'response_time'
        
    Returns:
    --------
    dict : Prioritized recommendations
    """
    
    # Prepare features for ML model
    features = np.array([[
        metrics.get('lcp', 0),
        metrics.get('fid', 0),
        metrics.get('cls', 0),
        metrics.get('page_size', 0),
        metrics.get('num_requests', 0),
        metrics.get('response_time', 0)
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict improvement potential
    prediction = rec_model.predict(features_scaled)[0]
    probability = rec_model.predict_proba(features_scaled)[0]
    
    # Map prediction to category
    categories = ['Good - Minor tweaks', 'Average - Moderate changes', 'Weak - Major overhaul']
    improvement_potential = categories[prediction]
    confidence = max(probability) * 100
    
    # Generate rule-based recommendations
    recommendations = {
        'HIGH': [],
        'MEDIUM': [],
        'LOW': []
    }
    
    # Full metrics dict for rule matching
    full_metrics = {
        'Largest_contentful_paint_LCP_ms': metrics.get('lcp', 0),
        'Interaction_to_Next_Paint_INP_ms': metrics.get('fid', 0),
        'Cumulative_Layout_Shift_CLS': metrics.get('cls', 0),
        'Page_size_MB': metrics.get('page_size', 0),
        'No_of_requests': metrics.get('num_requests', 0),
        'Response_time_ms': metrics.get('response_time', 0)
    }
    
    # Check each rule
    for category, rule in rules.items():
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
    
    return {
        'improvement_potential': improvement_potential,
        'confidence': f'{confidence:.1f}%',
        'recommendations': recommendations,
        'total_recommendations': sum(len(v) for v in recommendations.values())
    }

# Example usage:
# recommendations = get_recommendations({
#     'lcp': 3500,
#     'fid': 250,
#     'cls': 0.15,
#     'page_size': 4.5,
#     'num_requests': 120,
#     'response_time': 800
# })
# print(recommendations)
