#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.cluster import KMeans

sns.set(style='whitegrid')
ROOT = Path(__file__).resolve().parents[1]
LABELED_CSV = ROOT / 'All thesis data - set4.labeled.csv'
IMPUTED_CSV = ROOT / 'All thesis data - set4.cleaned.imputed.csv'
OUTDIR = ROOT / 'Code' / 'output' / 'classifiers'
OUTDIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

LABEL_ORDER = ['Good', 'Average', 'Weak']

def load_data():
    if LABELED_CSV.exists():
        df = pd.read_csv(LABELED_CSV)
    else:
        df = pd.read_csv(IMPUTED_CSV)
    return df


def features_and_base_metrics(df):
    # Use numeric columns except label/composite_score/url
    exclude = ['url', 'label', 'composite_score']
    num_df = df.select_dtypes(include=[np.number]).copy()
    # include only numeric features that are not composite or targets
    for c in exclude:
        if c in num_df.columns:
            num_df = num_df.drop(columns=[c])
    return num_df


def label_by_tertiles(df, cols):
    # Use existing composite_score if present or compute via MinMax
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
    # weights dict mapping column->weight; default heavier to LCP/TTI/SpeedIndex
    default_weights = {
        'Largest_contentful_paint_LCP_ms': 0.3,
        'First_Contentful_Paint_FCP_ms': 0.15,
        'Time_to_interactive_TTI_ms': 0.3,
        'Speed_Index_ms': 0.2,
        'Cumulative_Layout_Shift_CLS': 0.05
    }
    w = default_weights if weights is None else weights
    use_cols = [c for c in cols if c in w]
    # normalize values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    comp = scaler.fit_transform(df[use_cols])
    # weighted average
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
    # scale then cluster
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    # map clusters to labels by mean of LCP composite (lower composite -> Good)
    if 'Largest_contentful_paint_LCP_ms' in df.columns:
        comp = df['Largest_contentful_paint_LCP_ms']
    else:
        comp = df[cols].mean(axis=1)
    cluster_means = {}
    for c in np.unique(clusters):
        cluster_means[c] = comp[clusters == c].mean()
    # sort clusters by mean ascending
    sorted_clusters = sorted(cluster_means.keys(), key=lambda k: cluster_means[k])
    mapping = {sorted_clusters[0]: 'Good', sorted_clusters[1]: 'Average', sorted_clusters[2]: 'Weak'}
    return pd.Series([mapping[c] for c in clusters])


def train_and_eval(X, y, prefix):
    results = {}
    # label encode
    mapping = {l:i for i,l in enumerate(LABEL_ORDER)}
    y_enc = y.map(mapping).values
    X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_val)
    results['rf'] = compute_class_metrics(y_val, preds_rf)
    joblib.dump(rf, OUTDIR / f'{prefix}_rf.joblib')

    # LightGBM
    lgbc = lgb.LGBMClassifier(n_estimators=200, random_state=42)
    lgbc.fit(X_train, y_train)
    preds_lgb = lgbc.predict(X_val)
    results['lgbm'] = compute_class_metrics(y_val, preds_lgb)
    joblib.dump(lgbc, OUTDIR / f'{prefix}_lgbm.joblib')

    # Keras MLP
    num_classes = len(LABEL_ORDER)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, OUTDIR / f'{prefix}_scaler.joblib')

    tf.random.set_seed(42)
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(X_train_s.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=30, batch_size=32, verbose=0)
    preds_keras = model.predict(X_val_s).argmax(axis=1)
    results['keras'] = compute_class_metrics(y_val, preds_keras)
    model.save(OUTDIR / f'{prefix}_keras.h5')

    # Save confusion matrices
    for name, pred in [('rf', preds_rf), ('lgbm', preds_lgb), ('keras', preds_keras)]:
        cm = confusion_matrix(y_val, pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix {prefix} {name}')
        plt.savefig(OUTDIR / f'confusion_{prefix}_{name}.png')
        plt.close()

    # Save classification reports
    for name, res in results.items():
        with open(OUTDIR / f'{prefix}_{name}_metrics.json', 'w') as f:
            json.dump(res, f, indent=2)
    return results


def compute_class_metrics(y_true, y_pred):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    }
    # per-class accuracy
    per_class = {}
    for i, label in enumerate(LABEL_ORDER):
        idx = (y_true == i)
        if idx.sum() == 0:
            per_class[label] = None
        else:
            per_class[label] = float((y_pred[idx] == i).sum() / idx.sum())
    metrics['per_class_accuracy'] = per_class
    return metrics


def main():
    df = load_data()
    num_df = features_and_base_metrics(df)
    # define composite metric cols
    composite_cols = [
        'Largest_contentful_paint_LCP_ms',
        'First_Contentful_Paint_FCP_ms',
        'Time_to_interactive_TTI_ms',
        'Speed_Index_ms',
        'Cumulative_Layout_Shift_CLS'
    ]

    # Strategy 1: tertiles (existing)
    df['label_tertiles'] = label_by_tertiles(df, composite_cols)

    # Strategy 2: weighted
    df['label_weighted'] = label_by_weighted(df, composite_cols)

    # Strategy 3: kmeans
    df['label_kmeans'] = label_by_kmeans(df, composite_cols)

    summary = {}
    # Train for each strategy
    for col in ['label_tertiles', 'label_weighted', 'label_kmeans']:
        y = df[col]
        # ensure labels in LABEL_ORDER, map if needed
        # if labels differ, map by unique classes
        X = num_df.copy()
        results = train_and_eval(X, y, col)
        summary[col] = results

    # Save summary
    with open(OUTDIR / 'classification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Determine best model per strategy by f1_macro
    best_overall = {}
    for strat, res in summary.items():
        best = None
        best_score = -1
        for model_name, metrics in res.items():
            if metrics['f1_macro'] > best_score:
                best_score = metrics['f1_macro']
                best = model_name
        best_overall[strat] = {'best_model': best, 'f1_macro': best_score}

    with open(OUTDIR / 'best_models_per_strategy.json', 'w') as f:
        json.dump(best_overall, f, indent=2)

    print('Classification training complete. Outputs in', OUTDIR)
    print('Best models per strategy:')
    print(json.dumps(best_overall, indent=2))

if __name__ == '__main__':
    main()
