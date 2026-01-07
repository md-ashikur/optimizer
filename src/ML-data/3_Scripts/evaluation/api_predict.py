from pathlib import Path
import joblib
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'Code' / 'output'
MODEL_RF = OUTDIR / 'model_rf.joblib'
MODEL_LGBM = OUTDIR / 'model_lgbm.joblib'
SCALER = OUTDIR / 'scaler.joblib'
FEATURE_IMPORTANCE = OUTDIR / 'feature_importances.csv'

app = FastAPI(title='Web Performance Predictor')

class PredictRequest(BaseModel):
    features: Dict[str, Any]


def load_models():
    models = {}
    if MODEL_LGBM.exists():
        models['lgbm'] = joblib.load(MODEL_LGBM)
    if MODEL_RF.exists():
        models['rf'] = joblib.load(MODEL_RF)
    if SCALER.exists():
        models['scaler'] = joblib.load(SCALER)
    # Keras model optional
    keras_path = OUTDIR / 'model_keras.h5'
    if keras_path.exists():
        try:
            models['keras'] = keras.models.load_model(keras_path)
        except Exception as e:
            print('Warning: could not load Keras model:', e)
    return models

MODELS = load_models()


def prepare_input_row(row: Dict[str, Any], feature_columns: List[str]):
    # Build a DataFrame with one row containing all features in feature_columns
    data = {c: row.get(c, 0) for c in feature_columns}
    df = pd.DataFrame([data])
    # ensure numeric types
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df


@app.on_event('startup')
def startup_event():
    # load feature columns from training by inferring from feature_importances.csv
    global FEATURE_COLS
    # Prefer the original CSV column order (if available) to preserve feature ordering
    sample = ROOT / 'All thesis data - set4.cleaned.imputed.csv'
    if sample.exists():
        df = pd.read_csv(sample)
        targets = [
            'Largest_contentful_paint_LCP_ms',
            'Cumulative_Layout_Shift_CLS',
            'First_Contentful_Paint_FCP_ms',
            'Time_to_interactive_TTI_ms'
        ]
        FEATURE_COLS = [c for c in df.columns if c not in targets + ['url']]
    elif FEATURE_IMPORTANCE.exists():
        fi = pd.read_csv(FEATURE_IMPORTANCE)
        FEATURE_COLS = list(fi['feature'].unique())
    else:
        FEATURE_COLS = []


@app.post('/predict')
def predict(req: PredictRequest):
    if not MODELS:
        raise HTTPException(status_code=500, detail='Models not loaded')
    if not FEATURE_COLS:
        raise HTTPException(status_code=500, detail='Feature columns unknown')
    X = prepare_input_row(req.features, FEATURE_COLS)
    results = {}
    # RF
    if 'rf' in MODELS:
        preds = MODELS['rf'].predict(X)
        results['rf'] = preds.tolist()[0]
    # LGBM
    if 'lgbm' in MODELS:
        preds = MODELS['lgbm'].predict(X)
        results['lgbm'] = preds.tolist()[0]
    # Keras (requires scaler)
    if 'keras' in MODELS:
        if 'scaler' not in MODELS:
            raise HTTPException(status_code=500, detail='Scaler required for Keras model')
        Xs = MODELS['scaler'].transform(X)
        preds = MODELS['keras'].predict(Xs)
        results['keras'] = preds.tolist()[0]
    return {
        'features_used': FEATURE_COLS,
        'predictions': results
    }


@app.post('/explain')
def explain(req: PredictRequest):
    """Return SHAP values for LGBM model predictions per target.

    Response structure:
    {
      "model": "lgbm",
      "predicted": {target_idx: value, ...},
      "base_values": {target_idx: base_value, ...},
      "shap_values": {target_idx: {feature: shap_val, ...}, ...}
    }
    """
    if 'lgbm' not in MODELS:
        raise HTTPException(status_code=404, detail='LightGBM model not loaded')
    model = MODELS['lgbm']
    if not FEATURE_COLS:
        raise HTTPException(status_code=500, detail='Feature columns unknown')
    X = prepare_input_row(req.features, FEATURE_COLS)

    # Handle MultiOutputRegressor wrapper
    estimators = None
    try:
        if hasattr(model, 'estimators_'):
            estimators = model.estimators_
        elif hasattr(model, 'estimators'):
            estimators = model.estimators
    except Exception:
        estimators = None

    import shap
    explainer_results = {'model': 'lgbm', 'predicted': {}, 'base_values': {}, 'shap_values': {}}

    # If wrapped in MultiOutputRegressor, explain each sub-estimator
    if estimators is not None:
        for i, est in enumerate(estimators):
            try:
                expl = shap.TreeExplainer(est)
                shap_vals = expl.shap_values(X)
                # shap_vals: array of shape (n_samples, n_features)
                base = expl.expected_value
                pred = est.predict(X)[0]
                explainer_results['predicted'][str(i)] = float(pred)
                explainer_results['base_values'][str(i)] = float(base if np.isscalar(base) else base[0])
                # Map feature names to shap values
                feats = {f: float(v) for f, v in zip(FEATURE_COLS, shap_vals[0])}
                explainer_results['shap_values'][str(i)] = feats
            except Exception as e:
                explainer_results['shap_values'][str(i)] = {'error': str(e)}
    else:
        # single estimator
        est = model
        try:
            expl = shap.TreeExplainer(est)
            shap_vals = expl.shap_values(X)
            base = expl.expected_value
            pred = est.predict(X)[0]
            explainer_results['predicted']['0'] = float(pred)
            explainer_results['base_values']['0'] = float(base if np.isscalar(base) else base[0])
            feats = {f: float(v) for f, v in zip(FEATURE_COLS, shap_vals[0])}
            explainer_results['shap_values']['0'] = feats
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'SHAP explanation error: {e}')

    return explainer_results


@app.get('/feature_importances')
def feature_importances():
    if FEATURE_IMPORTANCE.exists():
        fi = pd.read_csv(FEATURE_IMPORTANCE)
        # aggregate by feature across targets
        agg = fi.groupby('feature')['importance'].mean().reset_index().sort_values('importance', ascending=False)
        return agg.to_dict(orient='records')
    raise HTTPException(status_code=404, detail='feature_importances.csv not found')


if __name__ == '__main__':
    # Quick local test: predict for the first row in the imputed CSV
    # initialize feature columns same as FastAPI startup
    try:
        startup_event()
    except Exception:
        pass
    sample = ROOT / 'All thesis data - set4.cleaned.imputed.csv'
    if sample.exists():
        df = pd.read_csv(sample)
        targets = [
            'Largest_contentful_paint_LCP_ms',
            'Cumulative_Layout_Shift_CLS',
            'First_Contentful_Paint_FCP_ms',
            'Time_to_interactive_TTI_ms'
        ]
        feat_cols = [c for c in df.columns if c not in targets + ['url']]
        row = df.iloc[0][feat_cols].to_dict()
        print('Using features:', feat_cols)
        print('Sample input (first row):')
        print(row)
        res = predict(PredictRequest(features=row))
        print('Prediction result:')
        print(json.dumps(res, indent=2))
    else:
        print('Sample CSV not found at', sample)
