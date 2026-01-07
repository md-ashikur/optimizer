#!/usr/bin/env python3
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_features(df, target_cols, drop_cols=['url']):
    df = df.copy()
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])
    X = df.drop(columns=target_cols)
    y = df[target_cols]
    # Ensure numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')
    # Fill remaining NaNs (should be none after imputation)
    X = X.fillna(0)
    y = y.fillna(0)
    return X, y


def train_tree_models(X_train, y_train, X_val, y_val, outdir):
    results = {}
    # RandomForest
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_val)
    results['rf'] = evaluate_preds(y_val, preds_rf)
    joblib.dump(rf, outdir / 'model_rf.joblib')

    # LightGBM per-target via MultiOutputRegressor
    lgbm = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100, random_state=42))
    lgbm.fit(X_train, y_train)
    preds_lgb = lgbm.predict(X_val)
    results['lgbm'] = evaluate_preds(y_val, preds_lgb)
    joblib.dump(lgbm, outdir / 'model_lgbm.joblib')

    return results, rf, lgbm


def evaluate_preds(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # compute RMSE compatibly across sklearn versions
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    per_target = {}
    # Support both DataFrame and numpy array inputs for y_true
    if hasattr(y_true, 'columns'):
        cols = list(y_true.columns)
        y_true_arr = y_true.values
    else:
        # fallback column names
        cols = [f'target_{i}' for i in range(y_pred.shape[1])]
        y_true_arr = np.asarray(y_true)
    for i, col in enumerate(cols):
        mse_t = mean_squared_error(y_true_arr[:, i], y_pred[:, i])
        per_target[col] = {
            'mae': float(mean_absolute_error(y_true_arr[:, i], y_pred[:, i])),
            'rmse': float(np.sqrt(mse_t))
        }
    return {'mae': float(mae), 'rmse': float(rmse), 'per_target': per_target}


def train_keras(X_train, y_train, X_val, y_val, outdir, epochs=50, batch_size=32):
    tf.random.set_seed(42)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X_val)
    results = evaluate_preds(y_val, preds)
    model.save(outdir / 'model_keras.h5')
    # Save training history
    with open(outdir / 'keras_history.json', 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
    return results, model


def make_shap(lgbm_model, X_train, outdir):
    # Use TreeExplainer for LightGBM
    try:
        # If wrapped in MultiOutputRegressor, take the first estimator for explanation mapping per-target
        estimator = lgbm_model
        if hasattr(lgbm_model, 'estimators_') or hasattr(lgbm_model, 'estimators'):
            estimator = lgbm_model
        # shap: need a single estimator; explain each target separately
        expl_dir = outdir / 'shap'
        expl_dir.mkdir(parents=True, exist_ok=True)
        feature_names = X_train.columns.tolist()
        summary_frames = []
        # For each sub-estimator in MultiOutputRegressor
        if hasattr(lgbm_model, 'estimators_'):
            estimators = lgbm_model.estimators_
        elif hasattr(lgbm_model, 'estimators'):
            estimators = lgbm_model.estimators
        else:
            estimators = [lgbm_model]
        for i, est in enumerate(estimators):
            try:
                expl = shap.TreeExplainer(est)
                shap_values = expl.shap_values(X_train)
                # summary plot
                plt.figure(figsize=(8,6))
                shap.summary_plot(shap_values, X_train, show=False)
                plt.tight_layout()
                plt.savefig(expl_dir / f'shap_summary_target_{i}.png')
                plt.close()
            except Exception as e:
                print('SHAP error for estimator', i, e)
        return True
    except Exception as e:
        print('SHAP overall error', e)
        return False


def save_feature_importance(lgbm_model, feature_names, outdir):
    fi = []
    # If MultiOutputRegressor, aggregate importances
    if hasattr(lgbm_model, 'estimators_'):
        estimators = lgbm_model.estimators_
    elif hasattr(lgbm_model, 'estimators'):
        estimators = lgbm_model.estimators
    else:
        estimators = [lgbm_model]
    for i, est in enumerate(estimators):
        try:
            imp = getattr(est, 'feature_importances_', None)
            if imp is None and hasattr(est, 'booster_'):
                imp = est.booster_.feature_importance()
            if imp is not None:
                for fname, val in zip(feature_names, imp):
                    fi.append({'target_idx': i, 'feature': fname, 'importance': float(val)})
        except Exception:
            continue
    pd.DataFrame(fi).to_csv(outdir / 'feature_importances.csv', index=False)


def main(csv_path, outdir_path):
    outdir = Path(outdir_path)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(csv_path)
    # Define target columns
    targets = [
        'Largest_contentful_paint_LCP_ms',
        'Cumulative_Layout_Shift_CLS',
        'First_Contentful_Paint_FCP_ms',
        'Time_to_interactive_TTI_ms'
    ]
    for t in targets:
        if t not in df.columns:
            print('Target column missing:', t)
            sys.exit(1)
    X, y = prepare_features(df, targets, drop_cols=['url'])
    feature_names = X.columns.tolist()

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features for Keras
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    joblib.dump(scaler, outdir / 'scaler.joblib')

    # Train tree models
    tree_results, rf_model, lgbm_model = train_tree_models(X_train, y_train, X_val, y_val, outdir)

    # Train keras multi-output
    keras_results, keras_model = train_keras(X_train_scaled, y_train.values, X_val_scaled, y_val.values, outdir, epochs=30)

    # SHAP for LightGBM
    shap_ok = make_shap(lgbm_model, X_train, outdir)

    # Feature importances
    save_feature_importance(lgbm_model, feature_names, outdir)

    summary = {
        'tree_results': tree_results,
        'keras_results': keras_results,
        'shap_generated': bool(shap_ok)
    }
    with open(outdir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Training complete. Summary saved to', outdir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python train_models.py path/to/data.csv [output_dir]')
        sys.exit(1)
    csv_path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) >= 3 else 'Code/output'
    main(csv_path, outdir)
