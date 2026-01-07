#!/usr/bin/env python3
"""Evaluate saved models (if present) and train quick baselines.

Outputs a CSV with accuracy/precision/recall/f1 (macro) per strategy and model.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
LABELED_CSV = ROOT / 'All thesis data - labeled.csv'
IMPUTED_CSV = ROOT / 'All thesis data - set4.cleaned.imputed.csv'
OUTDIR = ROOT / 'Code' / 'output' / 'classifiers'
OUTDIR.mkdir(parents=True, exist_ok=True)


LABEL_ORDER = ['Good', 'Average', 'Weak']


def load_data():
    if LABELED_CSV.exists():
        return pd.read_csv(LABELED_CSV)
    return pd.read_csv(IMPUTED_CSV)


def features_and_base_metrics(df):
    exclude = ['url', 'label', 'composite_score']
    num_df = df.select_dtypes(include=[np.number]).copy()
    for c in exclude:
        if c in num_df.columns:
            num_df = num_df.drop(columns=[c])
    return num_df


def label_by_tertiles(df, cols):
    if 'composite_score' not in df.columns:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        comp = scaler.fit_transform(df[cols])
        df['composite_score'] = comp.mean(axis=1)
    q1 = df['composite_score'].quantile(1/3)
    q2 = df['composite_score'].quantile(2/3)
    def f(s):
        if s <= q1:
            return 'Good'
        if s <= q2:
            return 'Average'
        return 'Weak'
    return df['composite_score'].apply(f)


def label_by_weighted(df, cols, weights=None):
    default_weights = {
        'Largest_contentful_paint_LCP_ms': 0.3,
        'First_Contentful_Paint_FCP_ms': 0.15,
        'Time_to_interactive_TTI_ms': 0.3,
        'Speed_Index_ms': 0.2,
        'Cumulative_Layout_Shift_CLS': 0.05
    }
    w = default_weights if weights is None else weights
    use_cols = [c for c in cols if c in w]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    comp = scaler.fit_transform(df[use_cols])
    weights_arr = np.array([w[c] for c in use_cols])
    weights_arr = weights_arr / weights_arr.sum()
    df['composite_weighted'] = comp.dot(weights_arr)
    q1 = df['composite_weighted'].quantile(1/3)
    q2 = df['composite_weighted'].quantile(2/3)
    def f(s):
        if s <= q1:
            return 'Good'
        if s <= q2:
            return 'Average'
        return 'Weak'
    return df['composite_weighted'].apply(f)


def label_by_kmeans(df, cols):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    if 'Largest_contentful_paint_LCP_ms' in df.columns:
        comp = df['Largest_contentful_paint_LCP_ms']
    else:
        comp = df[cols].mean(axis=1)
    cluster_means = {}
    for c in np.unique(clusters):
        cluster_means[c] = comp[clusters == c].mean()
    sorted_clusters = sorted(cluster_means.keys(), key=lambda k: cluster_means[k])
    mapping = {sorted_clusters[0]: 'Good', sorted_clusters[1]: 'Average', sorted_clusters[2]: 'Weak'}
    return pd.Series([mapping[c] for c in clusters])


def compute_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }


def evaluate_strategy(df, X, y, prefix):
    results = []
    mapping = {l:i for i,l in enumerate(LABEL_ORDER)}
    y_enc = y.map(mapping).values
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # Scale data for Keras
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try to load saved models if present
    candidates = []
    rf_path = OUTDIR / f'{prefix}_rf.joblib'
    lgb_path = OUTDIR / f'{prefix}_lgbm.joblib'
    keras_path = OUTDIR / f'{prefix}_keras.h5'
    
    if rf_path.exists():
        candidates.append(('saved_rf', joblib.load(rf_path), False))
    if lgb_path.exists():
        candidates.append(('saved_lgbm', joblib.load(lgb_path), False))
    if keras_path.exists():
        try:
            from tensorflow import keras as keras_lib
            keras_model = keras_lib.models.load_model(keras_path)
            candidates.append(('saved_keras', keras_model, True))
        except Exception as e:
            print(f'Could not load Keras model: {e}')

    # Baseline quick training
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    candidates.append(('rf_trained', rf, False))

    lgbc = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    lgbc.fit(X_train, y_train)
    candidates.append(('lgbm_trained', lgbc, False))

    for item in candidates:
        if len(item) == 3:
            name, model, is_keras = item
        else:
            name, model = item
            is_keras = False
            
        try:
            if is_keras:
                # Keras models need scaled input and argmax for predictions
                preds_proba = model.predict(X_test_scaled, verbose=0)
                preds = preds_proba.argmax(axis=1)
            else:
                preds = model.predict(X_test)
        except Exception as e:
            # maybe model expects scaled input
            scaler_path = OUTDIR / f'{prefix}_scaler.joblib'
            if scaler_path.exists():
                scaler_loaded = joblib.load(scaler_path)
                X_test_s = scaler_loaded.transform(X_test)
                if is_keras:
                    preds_proba = model.predict(X_test_s, verbose=0)
                    preds = preds_proba.argmax(axis=1)
                else:
                    preds = model.predict(X_test_s)
            else:
                # try scaling on the fly
                model.fit(X_train_scaled if is_keras else X_train, y_train)
                if is_keras:
                    preds_proba = model.predict(X_test_scaled, verbose=0)
                    preds = preds_proba.argmax(axis=1)
                else:
                    preds = model.predict(X_test)

        metrics = compute_metrics(y_test, preds)
        metrics.update({'strategy': prefix, 'model': name})
        results.append(metrics)

    return results


def main():
    df = load_data()
    X_all = features_and_base_metrics(df)
    composite_cols = [
        'Largest_contentful_paint_LCP_ms',
        'First_Contentful_Paint_FCP_ms',
        'Time_to_interactive_TTI_ms',
        'Speed_Index_ms',
        'Cumulative_Layout_Shift_CLS'
    ]

    df['label_tertiles'] = label_by_tertiles(df, composite_cols)
    df['label_weighted'] = label_by_weighted(df, composite_cols)
    df['label_kmeans'] = label_by_kmeans(df, composite_cols)

    all_results = []
    for col in ['label_tertiles', 'label_weighted', 'label_kmeans']:
        y = df[col]
        res = evaluate_strategy(df, X_all, y, col)
        all_results.extend(res)

    out_df = pd.DataFrame(all_results)
    out_csv = OUTDIR / 'evaluation_summary.csv'
    out_df.to_csv(out_csv, index=False)
    print('Wrote', out_csv)
    print(out_df.sort_values(['strategy','model']))

    # Create metric-specific folders and save sorted CSVs
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    for m in metrics:
        folder = OUTDIR / m
        folder.mkdir(parents=True, exist_ok=True)
        sorted_df = out_df.sort_values([m], ascending=False)
        sorted_path = folder / f'evaluation_sorted_by_{m}.csv'
        sorted_df.to_csv(sorted_path, index=False)
        print('Wrote', sorted_path)


if __name__ == '__main__':
    main()
