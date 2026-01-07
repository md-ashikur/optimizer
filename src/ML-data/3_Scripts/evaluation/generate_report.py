#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Code' / 'output'
EDA = OUT / 'eda'
CLS = OUT / 'classifiers'
TUNE = CLS / 'tuning'
PDF_OUT = OUT / 'report.pdf'

plt.rcParams.update({'figure.max_open_warning': 0})

def add_image_page(pp, img_path, title=None):
    fig = plt.figure(figsize=(11,8.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img)
    except Exception:
        ax.text(0.5, 0.5, f'Could not load image:\n{img_path}', ha='center')
    if title:
        fig.suptitle(title, fontsize=16)
    pp.savefig(fig)
    plt.close(fig)

with PdfPages(PDF_OUT) as pp:
    # Title page
    fig = plt.figure(figsize=(11,8.5))
    plt.axis('off')
    plt.text(0.5, 0.7, 'Dynamic Web Performance Optimization Measurement', ha='center', fontsize=20, weight='bold')
    plt.text(0.5, 0.62, 'Using Machine Learning Analytics — Report', ha='center', fontsize=14)
    plt.text(0.5, 0.5, 'Author: (generated)', ha='center', fontsize=12)
    plt.text(0.5, 0.45, f'Data: {ROOT / "All thesis data - set4.cleaned.imputed.csv"}', ha='center', fontsize=8)
    pp.savefig(fig)
    plt.close(fig)

    # Basic stats page
    basic_stats = EDA / 'basic_describe.csv'
    if basic_stats.exists():
        df = pd.read_csv(basic_stats)
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')
        ax.set_title('Dataset Basic Statistics', fontsize=16)
        table = ax.table(cellText=df.round(3).values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pp.savefig(fig)
        plt.close(fig)

    # EDA images
    for name in ['composite_boxplot.png', 'composite_hist.png', 'correlation_matrix.png']:
        p = EDA / name
        if p.exists():
            add_image_page(pp, p, title=name.replace('_', ' ').replace('.png','').title())

    # Top feature importances aggregated
    fi = OUT / 'feature_importances.csv'
    if fi.exists():
        df_fi = pd.read_csv(fi)
        agg = df_fi.groupby('feature')['importance'].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(11,8.5))
        agg.plot(kind='barh', ax=ax)
        ax.invert_yaxis()
        ax.set_title('Top 15 Features by Mean Importance')
        ax.set_xlabel('Mean Importance')
        pp.savefig(fig)
        plt.close(fig)

    # Regression model summary
    reg_sum = OUT / 'training_summary.json'
    if reg_sum.exists():
        with open(reg_sum) as f:
            reg = json.load(f)
        # create a small table of per-model MAE/RMSE
        rows = []
        for m in reg.get('tree_results', {}):
            rows.append([m, reg['tree_results'][m]['mae'], reg['tree_results'][m]['rmse']])
        # keras
        if 'keras_results' in reg:
            rows.append(['keras', reg['keras_results']['mae'], reg['keras_results']['rmse']])
        fig, ax = plt.subplots(figsize=(8,4))
        ax.axis('off')
        tbl = ax.table(cellText=[[r[0], f"{r[1]:.2f}", f"{r[2]:.2f}"] for r in rows], colLabels=['Model','MAE','RMSE'], loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        pp.savefig(fig)
        plt.close(fig)

    # Classifier summary
    cls_sum = CLS / 'classification_summary.json'
    best = CLS / 'best_models_per_strategy.json'
    if cls_sum.exists():
        with open(cls_sum) as f:
            csum = json.load(f)
        # For each strategy, compile model f1_macro
        rows = []
        for strat, models in csum.items():
            for mname, metrics in models.items():
                rows.append([strat, mname, metrics['f1_macro']])
        df_rows = pd.DataFrame(rows, columns=['strategy','model','f1_macro'])
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')
        ax.set_title('Classifier F1 Macro Scores')
        tbl = ax.table(cellText=df_rows.round(4).values, colLabels=df_rows.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        pp.savefig(fig)
        plt.close(fig)

    # Include ROC/PR images from tuning if present
    for img in (TUNE.glob('roc_*.png')):
        add_image_page(pp, img, title=img.name)
    for img in (TUNE.glob('pr_*.png')):
        add_image_page(pp, img, title=img.name)

    # Confusion matrices
    import glob
    for img in (CLS.glob('confusion_*.png')):
        add_image_page(pp, img, title=img.name)

    # SHAP example (first shap png if exists)
    shap_dir = OUT / 'shap'
    if shap_dir.exists():
        files = sorted(shap_dir.glob('shap_summary_target_*.png'))
        if files:
            add_image_page(pp, files[0], title='SHAP summary (example)')

    # Final conclusions page
    fig = plt.figure(figsize=(11,8.5))
    plt.axis('off')
    plt.text(0.1, 0.8, 'Conclusions:', fontsize=16, weight='bold')
    plt.text(0.1, 0.65, '- K-means labeling produced best classifier f1 (see classifiers tuning).', fontsize=12)
    plt.text(0.1, 0.60, '- LightGBM and Keras performed well; check per-target regression metrics for LCP/FCP/TTI.', fontsize=12)
    plt.text(0.1, 0.50, '- SHAP explanations available via API /explain to identify feature causes and suggested fixes.', fontsize=12)
    pp.savefig(fig)
    plt.close(fig)

print('Report generated at', PDF_OUT)
