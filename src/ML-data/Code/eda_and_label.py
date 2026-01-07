#!/usr/bin/env python3
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set(style='whitegrid')

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / 'All thesis data - set4.cleaned.imputed.csv'
OUTDIR = ROOT / 'Code' / 'output' / 'eda'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Columns considered in composite performance score (lower is better for all chosen metrics)
COMPOSITE_COLS = [
    'Largest_contentful_paint_LCP_ms',
    'First_Contentful_Paint_FCP_ms',
    'Time_to_interactive_TTI_ms',
    'Speed_Index_ms',
    'Cumulative_Layout_Shift_CLS'
]

# Read data
print('Loading', INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
print('Rows:', len(df), 'Cols:', len(df.columns))

# Basic stats saved
summary = df.describe(include='all').T
summary.to_csv(OUTDIR / 'basic_describe.csv')

# Compute mean, median and percent diff
stats = []
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        mean = df[c].mean()
        median = df[c].median()
        pct_diff = (mean - median) / (median if median != 0 else 1) * 100
        stats.append({'feature': c, 'mean': mean, 'median': median, 'pct_diff_percent': pct_diff})

pd.DataFrame(stats).to_csv(OUTDIR / 'mean_median_stats.csv', index=False)

# Outlier detection via IQR for numeric columns
outliers = {}
for c in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    out_count = ((df[c] < low) | (df[c] > high)).sum()
    outliers[c] = int(out_count)

pd.DataFrame([{'feature': k, 'outlier_count': v} for k, v in outliers.items()]).to_csv(OUTDIR / 'outliers_count.csv', index=False)

# Create composite score (lower is better) using MinMax scaling per column
scaler = MinMaxScaler()
com_df = df[COMPOSITE_COLS].copy()
# Ensure numeric
com_df = com_df.apply(pd.to_numeric, errors='coerce').fillna(com_df.mean())
# MinMax; lower original -> lower scaled
com_scaled = pd.DataFrame(scaler.fit_transform(com_df), columns=COMPOSITE_COLS)
# Composite score = mean of scaled metrics
df['composite_score'] = com_scaled.mean(axis=1)

# Label by tertiles (Good = lowest third composite, Average = middle, Weak = top third)
q1 = df['composite_score'].quantile(1/3)
q2 = df['composite_score'].quantile(2/3)

def label_from_score(s):
    if s <= q1:
        return 'Good'
    if s <= q2:
        return 'Average'
    return 'Weak'

df['label'] = df['composite_score'].apply(label_from_score)

# Save labeled CSV
OUTCSV = ROOT / 'All thesis data - set4.labeled.csv'
df.to_csv(OUTCSV, index=False)
print('Saved labeled CSV to', OUTCSV)

# Univariate plots: boxplot + histogram for composite and each metric
plt.figure(figsize=(8,6))
sns.boxplot(x=df['composite_score'])
plt.title('Composite Score Boxplot')
plt.savefig(OUTDIR / 'composite_boxplot.png')
plt.close()

plt.figure(figsize=(8,6))
sns.histplot(df['composite_score'], kde=True)
plt.title('Composite Score Distribution')
plt.savefig(OUTDIR / 'composite_hist.png')
plt.close()

for c in COMPOSITE_COLS:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[c])
    plt.title(f'Boxplot: {c}')
    plt.savefig(OUTDIR / f'boxplot_{c}.png')
    plt.close()

    plt.figure(figsize=(8,6))
    sns.histplot(df[c], kde=True)
    plt.title(f'Distribution: {c}')
    plt.savefig(OUTDIR / f'hist_{c}.png')
    plt.close()

# Bivariate scatter plots for key pairs
PAIRS = [
    ('Largest_contentful_paint_LCP_ms', 'Time_to_interactive_TTI_ms'),
    ('Largest_contentful_paint_LCP_ms', 'First_Contentful_Paint_FCP_ms'),
    ('Time_to_interactive_TTI_ms', 'First_Contentful_Paint_FCP_ms'),
    ('Largest_contentful_paint_LCP_ms', 'Speed_Index_ms')
]
for x, y in PAIRS:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df[x], y=df[y], hue=df['label'], palette='Set1')
    sns.regplot(x=df[x], y=df[y], scatter=False, color='grey')
    plt.title(f'{x} vs {y}')
    plt.savefig(OUTDIR / f'scatter_{x}_vs_{y}.png')
    plt.close()

# Correlation matrix
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='vlag', center=0)
plt.title('Correlation Matrix')
plt.savefig(OUTDIR / 'correlation_matrix.png')
plt.close()
corr.to_csv(OUTDIR / 'correlation_matrix.csv')

# Encoding example: extract TLD from URL and count label distribution by TLD
if 'url' in df.columns:
    tlds = df['url'].astype(str).str.extract(r"\.([a-zA-Z]{2,})$", expand=False).fillna('other')
    df['tld'] = tlds
    tld_counts = df.groupby(['tld', 'label']).size().unstack(fill_value=0)
    tld_counts.to_csv(OUTDIR / 'tld_label_counts.csv')
    plt.figure(figsize=(10,6))
    tld_counts.sum(axis=1).sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title('Top 10 TLDs by count')
    plt.savefig(OUTDIR / 'top10_tlds.png')
    plt.close()

# Normalization and Standardization on numeric features (save to disk)
num_cols = num_df.columns.tolist()
mm_scaler = MinMaxScaler()
df_mm = pd.DataFrame(mm_scaler.fit_transform(num_df), columns=num_cols)
df_mm.to_csv(OUTDIR / 'data_minmax_scaled.csv', index=False)
ss = StandardScaler()
df_ss = pd.DataFrame(ss.fit_transform(num_df), columns=num_cols)
df_ss.to_csv(OUTDIR / 'data_standard_scaled.csv', index=False)

# Save summary for presentation: label counts, per-label means for composite and key metrics
label_counts = df['label'].value_counts()
label_counts.to_csv(OUTDIR / 'label_counts.csv')
per_label_means = df.groupby('label')[COMPOSITE_COLS + ['composite_score']].mean()
per_label_means.to_csv(OUTDIR / 'per_label_means.csv')

print('EDA & labeling complete. Outputs in', OUTDIR)
