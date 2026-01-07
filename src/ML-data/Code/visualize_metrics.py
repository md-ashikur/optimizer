#!/usr/bin/env python3
"""Generate visualizations and model-specific reports for metrics comparison."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'Code' / 'output' / 'classifiers'
VIZ_DIR = OUTDIR / 'visualizations'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Model-specific directories
MODEL_DIRS = {
    'RandomForest': OUTDIR / 'models' / 'RandomForest',
    'LightGBM': OUTDIR / 'models' / 'LightGBM',
}
for d in MODEL_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load evaluation summary CSV."""
    csv_path = OUTDIR / 'evaluation_summary.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} not found. Run evaluate_models.py first.')
    return pd.read_csv(csv_path)


def clean_model_name(model_str):
    """Extract clean model name from string."""
    if 'rf' in model_str.lower():
        return 'RandomForest'
    elif 'lgbm' in model_str.lower() or 'lightgbm' in model_str.lower():
        return 'LightGBM'
    elif 'keras' in model_str.lower():
        return 'Keras'
    return model_str


def save_model_specific_data(df):
    """Save CSV files organized by model name."""
    df['model_type'] = df['model'].apply(clean_model_name)
    
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type].copy()
        model_df = model_df.sort_values('f1_macro', ascending=False)
        
        # Save to model-specific directory
        if model_type in MODEL_DIRS:
            out_path = MODEL_DIRS[model_type] / f'{model_type}_all_metrics.csv'
            model_df.to_csv(out_path, index=False)
            print(f'Saved {model_type} metrics to {out_path}')
            
            # Save JSON summary
            summary = {
                'model_name': model_type,
                'best_strategy': model_df.iloc[0]['strategy'],
                'metrics': {
                    'accuracy': float(model_df.iloc[0]['accuracy']),
                    'precision_macro': float(model_df.iloc[0]['precision_macro']),
                    'recall_macro': float(model_df.iloc[0]['recall_macro']),
                    'f1_macro': float(model_df.iloc[0]['f1_macro'])
                }
            }
            json_path = MODEL_DIRS[model_type] / f'{model_type}_summary.json'
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f'Saved {model_type} summary to {json_path}')


def plot_metric_comparison(df, metric, title):
    """Create bar chart comparing models for a specific metric."""
    df['model_type'] = df['model'].apply(clean_model_name)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Group by model and strategy
    pivot_data = df.pivot_table(
        values=metric, 
        index='strategy', 
        columns='model_type',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title(f'{title} Comparison Across Models and Strategies', fontsize=16, fontweight='bold')
    ax.set_xlabel('Labeling Strategy', fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.legend(title='Model Type', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    out_path = VIZ_DIR / f'{metric}_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved chart: {out_path}')
    plt.close()


def plot_all_metrics_heatmap(df):
    """Create heatmap showing all metrics for all model-strategy combinations."""
    df['model_type'] = df['model'].apply(clean_model_name)
    df['model_strategy'] = df['model_type'] + ' - ' + df['strategy']
    
    # Select metrics columns
    metrics_cols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    heatmap_data = df.set_index('model_strategy')[metrics_cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, ax=ax, vmin=0.9, vmax=1.0)
    ax.set_title('All Metrics Heatmap: Model Performance', fontsize=16, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Model - Strategy', fontsize=12)
    plt.tight_layout()
    
    out_path = VIZ_DIR / 'all_metrics_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved heatmap: {out_path}')
    plt.close()


def plot_model_comparison_radar(df):
    """Create radar/spider chart comparing models across all metrics."""
    import numpy as np
    
    df['model_type'] = df['model'].apply(clean_model_name)
    
    # Get best performance per model type
    best_per_model = df.groupby('model_type').apply(
        lambda x: x.loc[x['f1_macro'].idxmax()]
    ).reset_index(drop=True)
    
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for _, row in best_per_model.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model_type'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'], fontsize=11)
    ax.set_ylim(0.9, 1.0)
    ax.set_title('Model Performance Comparison (Best Strategy per Model)', 
                 fontsize=16, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    
    out_path = VIZ_DIR / 'model_comparison_radar.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved radar chart: {out_path}')
    plt.close()


def plot_individual_metric_bars(df):
    """Create individual bar charts for each metric showing all models."""
    df['model_type'] = df['model'].apply(clean_model_name)
    df['full_name'] = df['model_type'] + '\n' + df['strategy'].str.replace('label_', '')
    
    metrics = [
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (Macro)'),
        ('recall_macro', 'Recall (Macro)'),
        ('f1_macro', 'F1-Score (Macro)')
    ]
    
    for metric_col, metric_name in metrics:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sorted_df = df.sort_values(metric_col, ascending=False)
        colors = ['#2ecc71' if 'LightGBM' in x else '#3498db' 
                  for x in sorted_df['model_type']]
        
        bars = ax.bar(range(len(sorted_df)), sorted_df[metric_col], color=colors, alpha=0.8)
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['full_name'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} - All Models and Strategies', 
                     fontsize=16, fontweight='bold')
        ax.set_ylim(sorted_df[metric_col].min() - 0.02, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_df[metric_col])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='LightGBM'),
            Patch(facecolor='#3498db', label='RandomForest')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        out_path = VIZ_DIR / f'{metric_col}_individual_bars.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved chart: {out_path}')
        plt.close()


def create_summary_report(df):
    """Create a text summary report."""
    df['model_type'] = df['model'].apply(clean_model_name)
    
    report = []
    report.append("=" * 80)
    report.append("MODEL PERFORMANCE COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall best model
    best_idx = df['f1_macro'].idxmax()
    best = df.loc[best_idx]
    report.append("OVERALL BEST MODEL:")
    report.append(f"  Model: {best['model_type']}")
    report.append(f"  Strategy: {best['strategy']}")
    report.append(f"  Accuracy: {best['accuracy']:.4f}")
    report.append(f"  Precision: {best['precision_macro']:.4f}")
    report.append(f"  Recall: {best['recall_macro']:.4f}")
    report.append(f"  F1-Score: {best['f1_macro']:.4f}")
    report.append("")
    
    # Per model type
    report.append("-" * 80)
    report.append("PERFORMANCE BY MODEL TYPE:")
    report.append("-" * 80)
    for model_type in sorted(df['model_type'].unique()):
        model_df = df[df['model_type'] == model_type]
        best_model = model_df.loc[model_df['f1_macro'].idxmax()]
        
        report.append(f"\n{model_type}:")
        report.append(f"  Best Strategy: {best_model['strategy']}")
        report.append(f"  Accuracy: {best_model['accuracy']:.4f}")
        report.append(f"  Precision: {best_model['precision_macro']:.4f}")
        report.append(f"  Recall: {best_model['recall_macro']:.4f}")
        report.append(f"  F1-Score: {best_model['f1_macro']:.4f}")
        
        avg_metrics = model_df[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']].mean()
        report.append(f"  Average across strategies:")
        report.append(f"    Accuracy: {avg_metrics['accuracy']:.4f}")
        report.append(f"    Precision: {avg_metrics['precision_macro']:.4f}")
        report.append(f"    Recall: {avg_metrics['recall_macro']:.4f}")
        report.append(f"    F1-Score: {avg_metrics['f1_macro']:.4f}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # Save to file
    report_path = VIZ_DIR / 'performance_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nSaved report to: {report_path}")


def main():
    print("Loading evaluation results...")
    df = load_results()
    
    print("\n1. Saving model-specific data...")
    save_model_specific_data(df)
    
    print("\n2. Generating comparison charts...")
    plot_metric_comparison(df, 'accuracy', 'Accuracy')
    plot_metric_comparison(df, 'precision_macro', 'Precision (Macro)')
    plot_metric_comparison(df, 'recall_macro', 'Recall (Macro)')
    plot_metric_comparison(df, 'f1_macro', 'F1-Score (Macro)')
    
    print("\n3. Generating heatmap...")
    plot_all_metrics_heatmap(df)
    
    print("\n4. Generating radar chart...")
    plot_model_comparison_radar(df)
    
    print("\n5. Generating individual metric bar charts...")
    plot_individual_metric_bars(df)
    
    print("\n6. Creating summary report...")
    create_summary_report(df)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {VIZ_DIR}")
    print(f"\nModel-specific data saved to:")
    for model_type, path in MODEL_DIRS.items():
        print(f"  - {model_type}: {path}")


if __name__ == '__main__':
    main()
