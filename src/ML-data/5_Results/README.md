# Results

This folder contains all model evaluation results, metrics, and analysis.

## Folder Structure:

### metrics/
Complete performance metrics for all models
- `evaluation_summary.csv` - All models with accuracy, precision, recall, F1
- `classification_summary.json` - Classification results summary
- `best_models_per_strategy.json` - Top performer per labeling strategy
- `accuracy/`, `precision_macro/`, `recall_macro/`, `f1_macro/` - Sorted by each metric

### reports/
Training details and feature analysis
- `training_summary.json` - Model training configuration and results
- `feature_importances.csv` - Which features matter most
- `keras_history.json` - Neural network training progress
- `report.pdf` - Full performance report (if generated)

### confusion_matrices/
Visual confusion matrices for all model-strategy combinations
- Named: `confusion_label_{strategy}_{model}.png`
- Shows prediction accuracy breakdown by class

## Key Files:

**For Quick Overview:**
- `metrics/evaluation_summary.csv` - Compare all models at once

**For Best Results:**
- `metrics/best_models_per_strategy.json` - Top model per approach

**For Detailed Analysis:**
- Check specific metric folders (e.g., `f1_macro/` for F1-sorted results)
