#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import label_binarize

sns.set(style='whitegrid')
ROOT = Path(__file__).resolve().parents[1]
LABELED_CSV = ROOT / 'All thesis data - set4.labeled.csv'
OUTDIR = ROOT / 'Code' / 'output' / 'classifiers' / 'tuning'
OUTDIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ['Good', 'Average', 'Weak']

# Load labeled data
if not LABELED_CSV.exists():
    raise SystemExit('Labeled CSV not found; run eda_and_label.py first')
df = pd.read_csv(LABELED_CSV)
# features
exclude = ['url', 'label', 'composite_score']
X = df.select_dtypes(include=[np.number]).copy()
for c in exclude:
    if c in X.columns:
        X = X.drop(columns=[c])
# Choose strategy label to tune on (kmeans performed best earlier)
y = df['label_kmeans'] if 'label_kmeans' in df.columns else df['label']

# binarize labels for ROC/PR
classes = sorted(y.unique(), key=lambda v: LABEL_ORDER.index(v) if v in LABEL_ORDER else 0)
Y_bin = label_binarize(y, classes=classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_vals, X_val, y_train_vals, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# RandomForest randomized search
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_params = {
    'n_estimators': [100, 200, 400],
    'max_depth': [None, 10, 20, 40],
    'min_samples_split': [2, 5, 10],
}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=10, scoring='f1_macro', cv=4, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
joblib.dump(rf_search.best_estimator_, OUTDIR / 'rf_best.joblib')
with open(OUTDIR / 'rf_search_best_params.json', 'w') as f:
    json.dump(rf_search.best_params_, f, indent=2)

# LightGBM randomized search
lgbc = lgb.LGBMClassifier(random_state=42)
lgb_params = {
    'num_leaves': [31, 63, 127],
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1]
}
lgb_search = RandomizedSearchCV(lgbc, lgb_params, n_iter=10, scoring='f1_macro', cv=4, random_state=42, n_jobs=-1)
lgb_search.fit(X_train, y_train)
joblib.dump(lgb_search.best_estimator_, OUTDIR / 'lgb_best.joblib')
with open(OUTDIR / 'lgb_search_best_params.json', 'w') as f:
    json.dump(lgb_search.best_params_, f, indent=2)

# Evaluate and plot ROC/PR per class for best estimators
best_models = {'rf': rf_search.best_estimator_, 'lgb': lgb_search.best_estimator_}
for name, model in best_models.items():
    # predict probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)
    else:
        # try decision_function
        try:
            dec = model.decision_function(X_test)
            # convert via softmax
            exp = np.exp(dec - np.max(dec, axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)
        except Exception:
            continue
    # binarize true labels
    classes = sorted(y.unique(), key=lambda v: LABEL_ORDER.index(v) if v in LABEL_ORDER else 0)
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = y_test_bin.shape[1]
    # ROC
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[i] if probs.shape[0]==n_classes else probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC {classes[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC Curves - {name}')
    plt.legend(loc='lower right')
    plt.savefig(OUTDIR / f'roc_{name}.png')
    plt.close()

    # PR curves
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], probs[i] if probs.shape[0]==n_classes else probs[:, i])
        ap = average_precision_score(y_test_bin[:, i], probs[i] if probs.shape[0]==n_classes else probs[:, i])
        plt.plot(recall, precision, label=f'PR {classes[i]} (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {name}')
    plt.legend(loc='lower left')
    plt.savefig(OUTDIR / f'pr_{name}.png')
    plt.close()

print('Tuning complete. Results in', OUTDIR)
