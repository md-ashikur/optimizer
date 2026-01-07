# Model-Specific Metrics Guide

This guide shows you exactly where to find metrics for each model.

## Ì≥ä Organization Structure

Each metric type (accuracy, precision, recall, f1) has:
1. Overall sorted CSV (all models ranked)
2. Model-specific folders with individual CSVs

---

## ÌæØ Accuracy Metrics

### Location: `5_Results/metrics/accuracy/`

**Overall Rankings:**
- `evaluation_sorted_by_accuracy.csv` - All models sorted by accuracy

**By Model:**
- `RandomForest/RandomForest_accuracy.csv` - RandomForest only
- `LightGBM/LightGBM_accuracy.csv` - LightGBM only
- `Keras/Keras_accuracy.csv` - Keras only

---

## ÌæØ Precision Metrics

### Location: `5_Results/metrics/precision_macro/`

**Overall Rankings:**
- `evaluation_sorted_by_precision_macro.csv` - All models sorted by precision

**By Model:**
- `RandomForest/RandomForest_precision_macro.csv`
- `LightGBM/LightGBM_precision_macro.csv`
- `Keras/Keras_precision_macro.csv`

---

## ÌæØ Recall Metrics

### Location: `5_Results/metrics/recall_macro/`

**Overall Rankings:**
- `evaluation_sorted_by_recall_macro.csv` - All models sorted by recall

**By Model:**
- `RandomForest/RandomForest_recall_macro.csv`
- `LightGBM/LightGBM_recall_macro.csv`
- `Keras/Keras_recall_macro.csv`

---

## ÌæØ F1-Score Metrics (Most Important)

### Location: `5_Results/metrics/f1_macro/`

**Overall Rankings:**
- `evaluation_sorted_by_f1_macro.csv` - All models sorted by F1-score ‚≠ê

**By Model:**
- `RandomForest/RandomForest_f1_macro.csv`
- `LightGBM/LightGBM_f1_macro.csv` ‚≠ê BEST
- `Keras/Keras_f1_macro.csv`

---

## Ì≥à Model Summaries

### Location: `6_Visualizations/models/`

Each model has a dedicated folder with:
- `{Model}_all_metrics.csv` - All metrics across all strategies
- `{Model}_summary.json` - Best performance summary

**Available:**
- `RandomForest/` - RandomForest summaries
- `LightGBM/` - LightGBM summaries ‚≠ê
- `Keras/` - Keras summaries

---

## Ì¥ç Example: Finding LightGBM Precision

1. **All LightGBM precision scores:**
   ```
   5_Results/metrics/precision_macro/LightGBM/LightGBM_precision_macro.csv
   ```

2. **Compare LightGBM vs others on precision:**
   ```
   5_Results/metrics/precision_macro/evaluation_sorted_by_precision_macro.csv
   ```

3. **All LightGBM metrics together:**
   ```
   6_Visualizations/models/LightGBM/LightGBM_all_metrics.csv
   ```

---

## Ì≥ä Complete Comparison

**Single file with ALL models and ALL metrics:**
```
5_Results/metrics/evaluation_summary.csv
```

This is your master file - use it for complete comparisons!

---

## ÌøÜ Best Model Quick Access

**LightGBM with K-means labeling:**

- Model file: `4_Trained_Models/classification_models/label_kmeans_lgbm.joblib`
- Scaler: `4_Trained_Models/classification_models/label_kmeans_scaler.joblib`
- All metrics: `6_Visualizations/models/LightGBM/LightGBM_all_metrics.csv`
- Summary: `6_Visualizations/models/LightGBM/LightGBM_summary.json`
- Accuracy: `5_Results/metrics/accuracy/LightGBM/LightGBM_accuracy.csv`
- Precision: `5_Results/metrics/precision_macro/LightGBM/LightGBM_precision_macro.csv`
- Recall: `5_Results/metrics/recall_macro/LightGBM/LightGBM_recall_macro.csv`
- F1-Score: `5_Results/metrics/f1_macro/LightGBM/LightGBM_f1_macro.csv`

**Performance:**
- Accuracy: 97.86%
- Precision: 98.40%
- Recall: 98.53%
- F1-Score: 98.47%

---

## Ì≥Å Individual Model Metrics (JSON)

**Location: `5_Results/metrics/classification_metrics/`**

Detailed JSON files for each model-strategy combination:
- `label_kmeans_lgbm_metrics.json` ‚≠ê
- `label_kmeans_rf_metrics.json`
- `label_kmeans_keras_metrics.json`
- `label_tertiles_lgbm_metrics.json`
- `label_tertiles_rf_metrics.json`
- `label_tertiles_keras_metrics.json`
- `label_weighted_lgbm_metrics.json`
- `label_weighted_rf_metrics.json`
- `label_weighted_keras_metrics.json`

These contain detailed per-class accuracies and all metrics.

---

**For Your Thesis Presentation:**

1. **Show overall comparison:** `5_Results/metrics/evaluation_summary.csv`
2. **Highlight best model:** `6_Visualizations/models/LightGBM/LightGBM_summary.json`
3. **Visual comparisons:** Charts in `6_Visualizations/`
4. **Model-specific deep dive:** Use model-specific CSV files from metric folders

---

Last Updated: January 7, 2026
