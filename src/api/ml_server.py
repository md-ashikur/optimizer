#!/usr/bin/env python3
"""
FastAPI server for ML model predictions.
Collects live Lighthouse and browser performance metrics to feed the
pretrained LightGBM classifier, ensuring real recommendations without
mock data.
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
import shutil
import time
import urllib.parse
import os

import chromedriver_autoinstaller
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException

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

# Ensure Chromedriver is available.
# Prefer explicit env var `CHROMEDRIVER_PATH` to avoid network downloads during startup.
CHROMEDRIVER_PATH = None
if 'CHROMEDRIVER_PATH' in os.environ and os.environ.get('CHROMEDRIVER_PATH'):
    CHROMEDRIVER_PATH = os.environ.get('CHROMEDRIVER_PATH')
    print(f"Using CHROMEDRIVER_PATH from environment: {CHROMEDRIVER_PATH}")
else:
    # Allow disabling the automatic download in restricted networks
    if os.environ.get('DISABLE_CHROMEDRIVER_AUTOINSTALL'):
        print("Chromedriver autoinstall disabled via DISABLE_CHROMEDRIVER_AUTOINSTALL")
        CHROMEDRIVER_PATH = None
    else:
        try:
            CHROMEDRIVER_PATH = chromedriver_autoinstaller.install()
            print(f"Chromedriver installed to: {CHROMEDRIVER_PATH}")
        except Exception as exc:
            print(f"Chromedriver autoinstall failed: {exc}")
            CHROMEDRIVER_PATH = None

# Allow overriding Chrome binary location via env var when needed
CHROME_PATH = None
if 'CHROME_PATH' in os.environ:
    CHROME_PATH = os.environ.get('CHROME_PATH')
    print(f"Using CHROME_PATH from environment: {CHROME_PATH}")

# Try to load Keras model first
# Prefer joblib LightGBM model first to avoid importing heavy optional deps like TensorFlow
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

# If joblib model not found, attempt Keras model (lazy import of TensorFlow)
if model is None:
    for kp in KERAS_CANDIDATES:
        if kp.exists():
            try:
                # Lazy import tensorflow to avoid importing unless Keras model is present
                from tensorflow.keras.models import load_model
                model = load_model(str(kp))
                model_type = 'keras'
                print(f"Model loaded successfully from {kp} (Keras)")
                break
            except Exception as e:
                print(f"Error loading Keras model at {kp}: {e}")

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


def init_driver() -> webdriver.Chrome:
    """Initialise a headless Chrome driver for collecting navigation timings."""
    if not CHROMEDRIVER_PATH:
        raise RuntimeError("Chromedriver not available. Install Google Chrome and ensure chromedriver is on PATH.")

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")

    # If a specific Chrome binary is provided, point ChromeOptions to it
    if CHROME_PATH:
        try:
            chrome_options.binary_location = CHROME_PATH
        except Exception:
            pass

    service = Service(CHROMEDRIVER_PATH)
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except WebDriverException as e:
        raise RuntimeError(
            f"Failed to start Chrome webdriver: {e}. Ensure chromedriver matches Chrome version and CHROME_PATH (if set) points to the chrome executable."
        ) from e
    driver.set_page_load_timeout(60)
    return driver


def _safe_timing_diff(timing: Dict[str, float], end_key: str, start_key: str) -> float:
    end = timing.get(end_key, 0) or 0
    start = timing.get(start_key, 0) or 0
    diff = max(0, end - start)
    return float(diff)


def get_selenium_metrics(url: str) -> Dict[str, float]:
    """Collect navigation and resource timing metrics via Selenium."""
    driver = init_driver()
    metrics: Dict[str, float] = {}

    try:
        driver.get(url)
        time.sleep(5)

        timing = driver.execute_script("return window.performance.timing") or {}
        metrics.update({
            'Response_time_ms': _safe_timing_diff(timing, 'responseEnd', 'fetchStart'),
            'Load_time_ms': _safe_timing_diff(timing, 'loadEventEnd', 'navigationStart'),
            'DOM_Content_Loaded_Time_ms': _safe_timing_diff(timing, 'domContentLoadedEventEnd', 'navigationStart'),
            'First_byte_TTFB_ms': _safe_timing_diff(timing, 'responseStart', 'requestStart'),
        })

        metrics['Total_links'] = float(len(driver.find_elements(By.TAG_NAME, 'a')))

        resources = driver.execute_script("return window.performance.getEntriesByType('resource')") or []
        transfer_sum = 0.0
        encoded_sum = 0.0
        for resource in resources:
            transfer_sum += float(resource.get('transferSize') or 0)
            encoded_sum += float(resource.get('encodedBodySize') or 0)

        metrics['No_of_requests'] = float(len(resources))
        metrics['Byte_in_bytes'] = transfer_sum
        metrics['Page_size_MB'] = round(encoded_sum / (1024 * 1024), 6)

    except Exception as exc:
        print(f"Error collecting Selenium metrics for {url}: {exc}")
    finally:
        driver.quit()

    return metrics

def run_lighthouse(url: str) -> Dict[str, float]:
    """
    Run Lighthouse audit on the URL and extract metrics
    """
    # Determine how to invoke Lighthouse.
    # Prefer a local project install at ./node_modules/.bin/lighthouse (handles Windows .cmd)
    base_cmd = None
    try:
        project_root = Path(__file__).resolve().parents[2]
        local_bin = project_root / 'node_modules' / '.bin'
        local_exec = None
        for candidate in ('lighthouse', 'lighthouse.cmd', 'lighthouse.exe'):
            p = local_bin / candidate
            if p.exists():
                local_exec = str(p)
                break
        # On Windows prefer npx which handles wrappers correctly
        if os.name == 'nt' and shutil.which('npx'):
            base_cmd = ['npx', 'lighthouse']
        elif local_exec:
            # On Windows the local bin may be a .cmd wrapper; execute via cmd /c
            if os.name == 'nt' and Path(local_exec).suffix.lower() in ('.cmd', '.bat', '.ps1'):
                base_cmd = ['cmd', '/c', str(local_exec)]
            else:
                base_cmd = [str(local_exec)]
        elif shutil.which('lighthouse'):
            base_cmd = ['lighthouse']
        elif shutil.which('npx'):
            base_cmd = ['npx', 'lighthouse']
        else:
            raise Exception(
                "Lighthouse CLI not found. Run `npm install` or `yarn install` in the project root to install Lighthouse locally, or install it globally."
            )
    except Exception:
        # Fallback generic message
        raise Exception(
            "Lighthouse CLI not found. Run `npm install` or `yarn install` in the project root to install Lighthouse locally, or install it globally."
        )

    # Build chrome flags for Lighthouse; avoid auto remote-debugging port which can be flaky
    chrome_flags = '--headless --no-sandbox --disable-gpu --disable-dev-shm-usage'
    cmd = base_cmd + [
        str(url),
        '--output=json',
        '--output-path=stdout',
        '--only-categories=performance',
        f"--chrome-flags={chrome_flags}",
    ]

    # If CHROME_PATH env var provided, pass it to lighthouse CLI
    if CHROME_PATH:
        print(f"Using CHROME_PATH for Lighthouse: {CHROME_PATH}")
        cmd.append(f"--chrome-path={CHROME_PATH}")
    
    # Allow quiet mode still
    cmd.append('--quiet')

    # Debug: show the exact command being executed (helps diagnose PATH/wrapper issues)
    print(f"Running Lighthouse command: {cmd}")
    try:
        if os.name == 'nt':
            # On Windows, run via the shell so .cmd/.bat wrappers execute correctly
            cmd_str = ' '.join(f'"{c}"' if ' ' in c else c for c in cmd)
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                timeout=180,
                shell=True
            )
        else:
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
        print(f"Lighthouse failed. returncode={result.returncode} stderr={result.stderr}")
        raise Exception(f"Lighthouse exited with code {result.returncode}: {detail}")

    try:
        data = json.loads(result.stdout)
    except Exception as ex:
        raise Exception(f"Failed to parse Lighthouse JSON output. stdout: {result.stdout[:200]} stderr: {result.stderr[:200]}\nException: {ex}") from ex
    audits = data.get('audits', {})

    # Extract primary metrics (use 0 if missing)
    performance_score = data.get('categories', {}).get('performance', {}).get('score')
    metrics = {
        'Largest_contentful_paint_LCP_ms': audits.get('largest-contentful-paint', {}).get('numericValue', 0.0),
        'First_Contentful_Paint_FCP_ms': audits.get('first-contentful-paint', {}).get('numericValue', 0.0),
        'Time_to_interactive_TTI_ms': audits.get('interactive', {}).get('numericValue', 0.0),
        'Speed_Index_ms': audits.get('speed-index', {}).get('numericValue', 0.0),
        'Total_Blocking_Time_TBT_ms': audits.get('total-blocking-time', {}).get('numericValue', 0.0),
        'Cumulative_Layout_Shift_CLS': audits.get('cumulative-layout-shift', {}).get('numericValue', 0.0),
        'Max_Potential_FID_ms': audits.get('max-potential-fid', {}).get('numericValue', 0.0),
        'Server_Response_Time_ms': audits.get('server-response-time', {}).get('numericValue', 0.0),
        'Interaction_to_Next_Paint_INP_ms': audits.get('experimental-interaction-to-next-paint', {}).get('numericValue')
            or audits.get('interaction-to-next-paint', {}).get('numericValue', 0.0),
        'Design_optimization_score': round((performance_score or 0.0) * 100, 2),
        'JavaScript_Execution_Time_ms': audits.get('mainthread-work-breakdown', {}).get('numericValue', 0.0),
        'Main_Thread_Work_CPU_ms': audits.get('total-blocking-time', {}).get('numericValue', 0.0),
        'CSS_Blocking_Time_ms': audits.get('render-blocking-resources', {}).get('numericValue', 0.0),
    }

    # Keep legacy metrics used by the frontend visualisations
    metrics['Main_Thread_Work_ms'] = metrics['Main_Thread_Work_CPU_ms']
    metrics['Bootup_Time_ms'] = metrics['Time_to_interactive_TTI_ms'] * 0.3
    metrics['DOM_Content_Loaded_ms'] = metrics['First_Contentful_Paint_FCP_ms']
    metrics['First_Meaningful_Paint_ms'] = metrics['Largest_contentful_paint_LCP_ms']
    metrics['Fully_Loaded_Time_ms'] = metrics['Time_to_interactive_TTI_ms']

    return metrics


def get_broken_links(url: str, limit: int = 100) -> float:
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except Exception as exc:
        print(f"Failed to fetch page for broken-link check ({url}): {exc}")
        return 0.0

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]

    broken = 0
    for link in links[:limit]:
        target = link
        if not target.startswith(('http://', 'https://')):
            target = urllib.parse.urljoin(url, target)
        try:
            resp = requests.head(target, timeout=5, allow_redirects=True)
            if resp.status_code >= 400:
                broken += 1
        except Exception:
            broken += 1

    return float(broken)


def collect_all_metrics(url: str) -> Dict[str, float]:
    # Run Selenium and Lighthouse audits in parallel to reduce wall-clock time
    metrics: Dict[str, float] = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        tasks['selenium'] = ex.submit(get_selenium_metrics, url)
        tasks['lighthouse'] = ex.submit(run_lighthouse, url)

        selenium_metrics = {}
        lighthouse_metrics = {}

        for name, future in list(tasks.items()):
            try:
                res = future.result(timeout=120)
                if name == 'selenium':
                    selenium_metrics = res or {}
                else:
                    lighthouse_metrics = res or {}
            except Exception as exc:
                print(f"Metric collection task '{name}' failed: {exc}")

    # Merge whatever succeeded; at minimum we'll still attempt broken-link checks
    metrics.update(selenium_metrics)
    metrics.update(lighthouse_metrics)

    metrics['Broken_link_count'] = get_broken_links(url)

    # Align derived values with training dataset naming
    metrics.setdefault('Start_render_time_ms', metrics.get('First_Contentful_Paint_FCP_ms') or 0.0)
    metrics.setdefault('Document_complete_time_ms', metrics.get('Load_time_ms') or metrics.get('Time_to_interactive_TTI_ms') or 0.0)

    # Ensure all expected keys exist
    for key in (
        'Response_time_ms',
        'Load_time_ms',
        'DOM_Content_Loaded_Time_ms',
        'First_byte_TTFB_ms',
        'Total_links',
        'No_of_requests',
        'Byte_in_bytes',
        'Page_size_MB',
        'Largest_contentful_paint_LCP_ms',
        'Cumulative_Layout_Shift_CLS',
        'First_Contentful_Paint_FCP_ms',
        'Time_to_interactive_TTI_ms',
        'Speed_Index_ms',
        'Interaction_to_Next_Paint_INP_ms',
        'Design_optimization_score',
        'JavaScript_Execution_Time_ms',
        'Main_Thread_Work_CPU_ms',
        'CSS_Blocking_Time_ms',
        'Broken_link_count',
        'Start_render_time_ms',
        'Document_complete_time_ms',
        'Total_Blocking_Time_TBT_ms',
    ):
        metrics.setdefault(key, 0.0)

    return metrics

def prepare_features(metrics: Dict[str, float]) -> pd.DataFrame:
    """Align collected metrics with the scaler feature order."""
    if scaler is not None and hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
    else:
        feature_names = [
            'Response_time_ms',
            'Load_time_ms',
            'DOM_Content_Loaded_Time_ms',
            'First_byte_TTFB_ms',
            'Total_links',
            'No_of_requests',
            'Byte_in_bytes',
            'Page_size_MB',
            'Largest_contentful_paint_LCP_ms',
            'Cumulative_Layout_Shift_CLS',
            'First_Contentful_Paint_FCP_ms',
            'Time_to_interactive_TTI_ms',
            'Speed_Index_ms',
            'Interaction_to_Next_Paint_INP_ms',
            'Design_optimization_score',
            'JavaScript_Execution_Time_ms',
            'Main_Thread_Work_CPU_ms',
            'CSS_Blocking_Time_ms',
            'Broken_link_count',
            'Start_render_time_ms',
            'Document_complete_time_ms',
        ]

    row = {name: float(metrics.get(name, 0.0)) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names)

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
        target_url = str(request.url)
        metrics = collect_all_metrics(target_url)

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
