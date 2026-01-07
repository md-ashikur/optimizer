# Visualizations

This folder contains all charts, graphs, and visual comparisons of model performance.

## Main Comparison Charts:

### By Metric:
- `accuracy_comparison.png` - Accuracy across models and strategies (bar chart)
- `precision_macro_comparison.png` - Precision comparison
- `recall_macro_comparison.png` - Recall comparison  
- `f1_macro_comparison.png` - F1-score comparison

### Detailed Individual Bars:
- `accuracy_individual_bars.png` - All models sorted by accuracy
- `precision_macro_individual_bars.png` - All models sorted by precision
- `recall_macro_individual_bars.png` - All models sorted by recall
- `f1_macro_individual_bars.png` - All models sorted by F1-score

### Comprehensive Views:
- `all_metrics_heatmap.png` - Heatmap showing all metrics for all models
- `model_comparison_radar.png` - Radar/spider chart comparing best models
- `performance_summary_report.txt` - Text summary with rankings

## Model-Specific Data:

### models/RandomForest/
- `RandomForest_all_metrics.csv` - All RandomForest metrics across strategies
- `RandomForest_summary.json` - Best RandomForest performance summary

### models/LightGBM/
- `LightGBM_all_metrics.csv` - All LightGBM metrics across strategies
- `LightGBM_summary.json` - Best LightGBM performance summary

### models/Keras/
- `Keras_all_metrics.csv` - All Keras (Neural Network) metrics across strategies
- `Keras_summary.json` - Best Keras performance summary

## Best for Presentations:
1. `model_comparison_radar.png` - Shows overall model comparison
2. `all_metrics_heatmap.png` - Shows all results in one view
3. `f1_macro_individual_bars.png` - Shows ranking by most important metric
4. `performance_summary_report.txt` - Quick text summary for slides

## Usage:
All charts are high-resolution (300 DPI) PNG files, ready for thesis/presentation inclusion.
