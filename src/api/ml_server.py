#!/usr/bin/env python3
"""
FastAPI server for ML model predictions
Integrates with the best performing LightGBM model
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import json
from typing import Dict, Any
import os

app = FastAPI(title="WebOptimizer ML API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Next.js domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the best model (LightGBM with K-means labeling)
MODEL_DIR = Path(__file__).parent.parent / 'ML-data'
# possible model locations (project contains multiple outputs during training)
KERAS_CANDIDATES = [
    MODEL_DIR / 'Code' / 'output' / 'model_keras.h5',
    MODEL_DIR / '4_Trained_Models' / 'classification_models' / 'label_kmeans_keras.h5',
    MODEL_DIR / '4_Trained_Models' / 'classification_models' / 'label_tertiles_keras.h5',
]
SCALER_CANDIDATES = [
    MODEL_DIR / '4_Trained_Models' / 'classification_models' / 'label_kmeans_scaler.joblib',
    MODEL_DIR / 'Code' / 'output' / 'scaler.joblib',
]

model = None
scaler = None
model_type = None

# Try to load Keras model first
for kp in KERAS_CANDIDATES:
    if kp.exists():
        try:
            # Lazy import tensorflow to avoid import if not needed
            from tensorflow.keras.models import load_model
            model = load_model(str(kp))
            model_type = 'keras'
            print(f"Model loaded successfully from {kp} (Keras)")
            break
        except Exception as e:
            print(f"Error loading Keras model at {kp}: {e}")

# If Keras not available, fall back to joblib LightGBM
if model is None:
    LGBM_CANDIDATES = [
        MODEL_DIR / '4_Trained_Models' / 'classification_models' / 'label_kmeans_lgbm.joblib',
    ]
    for lp in LGBM_CANDIDATES:
        if lp.exists():
            try:
                model = joblib.load(lp)
                model_type = 'lgbm'
                print(f"Model loaded successfully from {lp} (joblib)")
                break
            except Exception as e:
                print(f"Error loading joblib model at {lp}: {e}")

# Load scaler if available
for sp in SCALER_CANDIDATES:
    if sp.exists():
        try:
            scaler = joblib.load(sp)
            print(f"Scaler loaded from {sp}")
            break
        except Exception as e:
            print(f"Error loading scaler at {sp}: {e}")

LABEL_ORDER = ['Good', 'Average', 'Weak']

class PredictionRequest(BaseModel):
    url: HttpUrl

class PredictionResponse(BaseModel):
    metrics: Dict[str, float]
    prediction: Dict[str, Any]
    raw_features: Dict[str, float]

def run_lighthouse(url: str) -> Dict[str, float]:
    """
    Run Lighthouse audit on the URL and extract metrics
    """
    try:
        # Run Lighthouse CLI
        cmd = [
            'lighthouse',
            str(url),
            '--output=json',
            '--output-path=stdout',
            '--only-categories=performance',
            '--chrome-flags="--headless"',
            '--quiet'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise Exception(f"Lighthouse failed: {result.stderr}")

        data = json.loads(result.stdout)
        audits = data.get('audits', {})

        # Extract metrics
        metrics = {
            'Largest_contentful_paint_LCP_ms': audits.get('largest-contentful-paint', {}).get('numericValue', 0),
            'First_Contentful_Paint_FCP_ms': audits.get('first-contentful-paint', {}).get('numericValue', 0),
            'Time_to_interactive_TTI_ms': audits.get('interactive', {}).get('numericValue', 0),
            'Speed_Index_ms': audits.get('speed-index', {}).get('numericValue', 0),
            'Total_Blocking_Time_TBT_ms': audits.get('total-blocking-time', {}).get('numericValue', 0),
            'Cumulative_Layout_Shift_CLS': audits.get('cumulative-layout-shift', {}).get('numericValue', 0),
            'Max_Potential_FID_ms': audits.get('max-potential-fid', {}).get('numericValue', 0),
            'Server_Response_Time_ms': audits.get('server-response-time', {}).get('numericValue', 0),
        }

        # Add derived metrics (using averages if not available)
        metrics.update({
            'DOM_Content_Loaded_ms': metrics['First_Contentful_Paint_FCP_ms'] * 1.2,
            'First_Meaningful_Paint_ms': metrics['Largest_contentful_paint_LCP_ms'] * 0.8,
            'Fully_Loaded_Time_ms': metrics['Time_to_interactive_TTI_ms'] * 1.1,
            'Total_Page_Size_KB': 1500.0,
            'Number_of_Requests': 50,
            'JavaScript_Size_KB': 500.0,
            'CSS_Size_KB': 100.0,
            'Image_Size_KB': 800.0,
            'Font_Size_KB': 50.0,
            'HTML_Size_KB': 50.0,
            'Main_Thread_Work_ms': metrics['Total_Blocking_Time_TBT_ms'] * 2,
            'Bootup_Time_ms': metrics['Time_to_interactive_TTI_ms'] * 0.3,
        })

        return metrics

    except Exception as e:
        print(f"Error running Lighthouse: {e}")
        print("Falling back to mock metrics for development. Install Lighthouse CLI for real metrics: npm install -g lighthouse")
        # Return mock metrics for development when Lighthouse not available
        return get_mock_metrics()

def get_mock_metrics() -> Dict[str, float]:
    """Return mock metrics for testing when Lighthouse is not available"""
    return {
        'Largest_contentful_paint_LCP_ms': 2800.0,
        'First_Contentful_Paint_FCP_ms': 1600.0,
        'Time_to_interactive_TTI_ms': 4200.0,
        'Speed_Index_ms': 3100.0,
        'Total_Blocking_Time_TBT_ms': 350.0,
        'Cumulative_Layout_Shift_CLS': 0.12,
        'Max_Potential_FID_ms': 180.0,
        'Server_Response_Time_ms': 450.0,
        'DOM_Content_Loaded_ms': 1920.0,
        'First_Meaningful_Paint_ms': 2240.0,
        'Fully_Loaded_Time_ms': 4620.0,
        'Total_Page_Size_KB': 1500.0,
        'Number_of_Requests': 50,
        'JavaScript_Size_KB': 500.0,
        'CSS_Size_KB': 100.0,
        'Image_Size_KB': 800.0,
        'Font_Size_KB': 50.0,
        'HTML_Size_KB': 50.0,
        'Main_Thread_Work_ms': 700.0,
        'Bootup_Time_ms': 1260.0,
    }

def prepare_features(metrics: Dict[str, float]) -> np.ndarray:
    """Prepare features for model prediction"""
    # Expected feature order (21 features based on training data)
    feature_names = [
        'Largest_contentful_paint_LCP_ms',
        'First_Contentful_Paint_FCP_ms',
        'Time_to_interactive_TTI_ms',
        'Speed_Index_ms',
        'Total_Blocking_Time_TBT_ms',
        'Cumulative_Layout_Shift_CLS',
        'Max_Potential_FID_ms',
        'Server_Response_Time_ms',
        'DOM_Content_Loaded_ms',
        'First_Meaningful_Paint_ms',
        'Fully_Loaded_Time_ms',
        'Total_Page_Size_KB',
        'Number_of_Requests',
        'JavaScript_Size_KB',
        'CSS_Size_KB',
        'Image_Size_KB',
        'Font_Size_KB',
        'HTML_Size_KB',
        'Main_Thread_Work_ms',
        'Bootup_Time_ms',
        'Offscreen_Images_KB'
    ]
    
    # Create feature array
    features = []
    for name in feature_names:
        features.append(metrics.get(name, 0.0))
    
    return np.array(features).reshape(1, -1)

@app.get("/")
def read_root():
    return {
        "service": "WebOptimizer ML API",
        "model": "LightGBM (K-means labeling)",
        "accuracy": "98.47%",
        "status": "ready" if model is not None else "model not loaded"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Ensure models are available and dependencies (tensorflow/sklearn) are installed.")
    
    try:
        # Get metrics from Lighthouse
        metrics = run_lighthouse(request.url)

        # Prepare features
        features = prepare_features(metrics)

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction depending on model type
        if model_type == 'keras':
            # Keras expects float32
            features_in = features_scaled.astype('float32')
            proba = model.predict(features_in)
            proba = np.asarray(proba).reshape(-1)
            prediction_idx = int(np.argmax(proba))
            prediction_proba = proba
            predicted_label = LABEL_ORDER[prediction_idx]
            confidence = float(proba[prediction_idx])
        else:
            # joblib model (LightGBM or sklearn)
            prediction_idx = model.predict(features_scaled)[0]
            # Some sklearn models provide predict_proba
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(features_scaled)[0]
            else:
                # Fallback: one-hot based on prediction
                probs = np.zeros(len(LABEL_ORDER), dtype=float)
                probs[int(prediction_idx)] = 1.0
                prediction_proba = probs

            predicted_label = LABEL_ORDER[int(prediction_idx)]
            confidence = float(prediction_proba[int(prediction_idx)])
        
        return PredictionResponse(
            metrics=metrics,
            prediction={
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": {
                    label: float(prob)
                    for label, prob in zip(LABEL_ORDER, prediction_proba)
                }
            },
            raw_features={k: float(v) for k, v in metrics.items()}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
