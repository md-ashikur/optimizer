
"""
SHAP Explanation Utility
Use this to explain predictions for new data points
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def explain_prediction(features, feature_names):
    """
    Explain a prediction using SHAP values
    
    Parameters:
    -----------
    features : array-like
        Feature values for the prediction (already scaled)
    feature_names : list
        Names of the features
        
    Returns:
    --------
    dict : Dictionary containing explanation details
    """
    # Load SHAP explainer
    base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
    explainer_path = base_path / "8_Advanced_Models" / "explainable_ai" / "shap_analysis"
    
    explainer = joblib.load(explainer_path / "shap_explainer.pkl")
    
    # Calculate SHAP values
    features_df = pd.DataFrame([features], columns=feature_names)
    shap_values = explainer(features_df)
    
    # Get feature contributions
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': features,
        'SHAP_Value': shap_values.values[0],
        'Abs_SHAP': np.abs(shap_values.values[0])
    }).sort_values('Abs_SHAP', ascending=False)
    
    return {
        'top_positive': contributions.nlargest(5, 'SHAP_Value'),
        'top_negative': contributions.nsmallest(5, 'SHAP_Value'),
        'most_important': contributions.nlargest(5, 'Abs_SHAP'),
        'base_value': shap_values.base_values,
        'prediction_impact': shap_values.values[0].sum()
    }

# Example usage:
# explanation = explain_prediction(feature_values, feature_names)
# print("Most important features:", explanation['most_important'])
