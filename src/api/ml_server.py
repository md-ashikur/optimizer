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
import shutil

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
    # Determine how to invoke Lighthouse: prefer direct CLI, fallback to npx if available
    base_cmd = None
    if shutil.which('lighthouse'):
        base_cmd = ['lighthouse']
    elif shutil.which('npx'):
        base_cmd = ['npx', 'lighthouse']
    else:
        raise Exception(
            "Lighthouse CLI not found. Install it with `yarn global add lighthouse` or `npm install -g lighthouse`, and ensure the global bin is on PATH."
        )

    cmd = base_cmd + [
        str(url),
        '--output=json',
        '--output-path=stdout',
        '--only-categories=performance',
        '--chrome-flags=--headless',
        '--quiet'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )
    except FileNotFoundError as fe:
        raise Exception(
            "Failed to run Lighthouse: executable not found. Ensure Lighthouse (or npx) is installed and on PATH."
        ) from fe
    except Exception as ex:
        raise Exception(f"Failed to run Lighthouse: {ex}") from ex

    if result.returncode != 0:
        # Provide stderr/stdout for debugging
        detail = result.stderr or result.stdout or '<no output>'
        raise Exception(f"Lighthouse exited with code {result.returncode}: {detail}")

    data = json.loads(result.stdout)
    audits = data.get('audits', {})

    # Extract primary metrics (use 0 if missing)
    metrics = {
        'Largest_contentful_paint_LCP_ms': audits.get('largest-contentful-paint', {}).get('numericValue', 0.0),
        'First_Contentful_Paint_FCP_ms': audits.get('first-contentful-paint', {}).get('numericValue', 0.0),
        'Time_to_interactive_TTI_ms': audits.get('interactive', {}).get('numericValue', 0.0),
        'Speed_Index_ms': audits.get('speed-index', {}).get('numericValue', 0.0),
        'Total_Blocking_Time_TBT_ms': audits.get('total-blocking-time', {}).get('numericValue', 0.0),
        'Cumulative_Layout_Shift_CLS': audits.get('cumulative-layout-shift', {}).get('numericValue', 0.0),
        'Max_Potential_FID_ms': audits.get('max-potential-fid', {}).get('numericValue', 0.0),
        'Server_Response_Time_ms': audits.get('server-response-time', {}).get('numericValue', 0.0),
    }

    # Derived/simple computed metrics
    metrics.update({
        'DOM_Content_Loaded_ms': metrics['First_Contentful_Paint_FCP_ms'] * 1.2,
        'First_Meaningful_Paint_ms': metrics['Largest_contentful_paint_LCP_ms'] * 0.8,
        'Fully_Loaded_Time_ms': metrics['Time_to_interactive_TTI_ms'] * 1.1,
        'Main_Thread_Work_ms': metrics['Total_Blocking_Time_TBT_ms'] * 2,
        'Bootup_Time_ms': metrics['Time_to_interactive_TTI_ms'] * 0.3,
    })

    # Parse resource-summary for sizes and request counts if available
    resource_summary = audits.get('resource-summary', {}).get('details', {})
    total_size_kb = None
    number_of_requests = None
    js_kb = css_kb = img_kb = font_kb = html_kb = offscreen_images_kb = 0.0

    # resource-summary may contain "overallSavingsMs" or "items" depending on LH version
    items = resource_summary.get('items') if isinstance(resource_summary, dict) else None
    if items and isinstance(items, list):
        # items contains dicts with label and size in bytes or KB depending on LH version
        for it in items:
            label = it.get('label', '').lower()
            size = it.get('size', 0)
            # If size seems like bytes (>100000), convert to KB
            if size > 100000:
                size_kb = float(size) / 1024.0
            else:
                size_kb = float(size)

            if 'script' in label:
                js_kb += size_kb
            elif 'image' in label:
                img_kb += size_kb
            elif 'stylesheet' in label or 'css' in label:
                css_kb += size_kb
            elif 'font' in label:
                font_kb += size_kb
            elif 'document' in label or 'html' in label:
                html_kb += size_kb

    # Some LH versions store totals in resource-summary.summary
    totals = resource_summary.get('summary') if isinstance(resource_summary, dict) else None
    if totals and isinstance(totals, dict):
        total_size_kb = totals.get('totalBytes')
        number_of_requests = totals.get('requests')
        if total_size_kb and total_size_kb > 100000:
            total_size_kb = float(total_size_kb) / 1024.0

    # Fallbacks if not parsed
    if total_size_kb is None:
        total_size_kb = js_kb + css_kb + img_kb + font_kb + html_kb
    if number_of_requests is None:
        # Try to get from network-requests audit details
        details = audits.get('network-requests', {}).get('details', {})
        if isinstance(details, dict):
            reqs = details.get('items')
            if isinstance(reqs, list):
                number_of_requests = len(reqs)
    if number_of_requests is None:
        number_of_requests = 0

    # Try to estimate offscreen images size from image entries marked as offscreen in network-requests
    try:
        net_items = audits.get('network-requests', {}).get('details', {}).get('items', [])
        for it in net_items:
            if it.get('resourceType') == 'Image' and it.get('isOffscreen'):
                size_b = it.get('transferSize') or 0
                offscreen_images_kb += float(size_b) / 1024.0
    except Exception:
        pass

    metrics.update({
        'Total_Page_Size_KB': float(total_size_kb),
        'Number_of_Requests': int(number_of_requests),
        'JavaScript_Size_KB': float(js_kb),
        'CSS_Size_KB': float(css_kb),
        'Image_Size_KB': float(img_kb),
        'Font_Size_KB': float(font_kb),
        'HTML_Size_KB': float(html_kb),
        'Offscreen_Images_KB': float(offscreen_images_kb),
    })

    return metrics

def prepare_features(metrics: Dict[str, float]) -> pd.DataFrame:
    """Prepare features for model prediction and return a single-row DataFrame with named columns.

    Returning a DataFrame preserves feature names so the scaler (fitted with feature names)
    receives consistent columns and order, avoiding warnings and incorrect transforms.
    """
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

    row = {name: float(metrics.get(name, 0.0)) for name in feature_names}
    df = pd.DataFrame([row], columns=feature_names)
    return df

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

        # Prepare features as a named DataFrame so scaler gets correct feature names
        features_df = prepare_features(metrics)

        # If scaler was fitted with feature names, ensure DataFrame has the same columns/order
        if hasattr(scaler, 'feature_names_in_'):
            expected = list(scaler.feature_names_in_)
            for col in expected:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            features_df = features_df[expected]

        # Scale features
        features_scaled = scaler.transform(features_df)

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
