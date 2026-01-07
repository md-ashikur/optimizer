# ML-Data Folder Structure (Visual Tree)

```
ML-data/
│
├── README.md (📖 Main documentation - START HERE)
│
├── 1_Raw_Data/                          [Original unprocessed data]
│   ├── README.md
│   └── All thesis data - labeled.csv   (Original dataset: 1167 rows)
│
├── 2_Processed_Data/                    [Cleaned & analyzed data]
│   ├── eda_analysis/                    (Exploratory Data Analysis)
│   │   ├── basic_describe.csv
│   │   ├── correlation_matrix.csv
│   │   ├── data_minmax_scaled.csv
│   │   ├── data_standard_scaled.csv
│   │   ├── label_counts.csv
│   │   ├── mean_median_stats.csv
│   │   ├── outliers_count.csv
│   │   ├── per_label_means.csv
│   │   └── tld_label_counts.csv
│   └── shap_analysis/                   (Model interpretability)
│
├── 3_Scripts/                           [Python scripts for ML pipeline]
│   ├── data_preparation/
│   │   ├── clean_csv.py                 (Data cleaning)
│   │   ├── impute_csv.py                (Missing value handling)
│   │   └── eda_and_label.py             (EDA & labeling)
│   │
│   ├── model_training/
│   │   ├── train_models.py              (Train regression models)
│   │   ├── train_classifiers.py         (Train classifiers - 3 strategies)
│   │   └── tune_classifiers.py          (Hyperparameter tuning)
│   │
│   └── evaluation/
│       ├── evaluate_models.py           (Compute metrics)
│       ├── visualize_metrics.py         (Generate charts)
│       ├── generate_report.py           (Create PDF report)
│       └── api_predict.py               (Prediction API)
│
├── 4_Trained_Models/                    [Saved ML models]
│   ├── README.md
│   │
│   ├── regression_models/               (Continuous prediction)
│   │   ├── model_rf.joblib              (Random Forest)
│   │   ├── model_lgbm.joblib            (LightGBM)
│   │   ├── model_keras.h5               (Neural Network)
│   │   └── scaler.joblib                (Feature scaler)
│   │
│   └── classification_models/           (Category prediction: Good/Avg/Weak)
│       ├── label_tertiles_rf.joblib
│       ├── label_tertiles_lgbm.joblib
│       ├── label_tertiles_keras.h5
│       ├── label_tertiles_scaler.joblib
│       ├── label_weighted_rf.joblib
│       ├── label_weighted_lgbm.joblib
│       ├── label_weighted_keras.h5
│       ├── label_weighted_scaler.joblib
│       ├── label_kmeans_rf.joblib
│       ├── label_kmeans_lgbm.joblib     ⭐ BEST MODEL (98.47% F1)
│       ├── label_kmeans_keras.h5
│       └── label_kmeans_scaler.joblib
│
├── 5_Results/                           [Performance metrics & analysis]
│   ├── README.md
│   │
│   ├── metrics/                         (All performance numbers)
│   │   ├── evaluation_summary.csv       📊 MAIN METRICS FILE
│   │   ├── classification_summary.json
│   │   ├── best_models_per_strategy.json
│   │   ├── accuracy/
│   │   │   └── evaluation_sorted_by_accuracy.csv
│   │   ├── precision_macro/
│   │   │   └── evaluation_sorted_by_precision_macro.csv
│   │   ├── recall_macro/
│   │   │   └── evaluation_sorted_by_recall_macro.csv
│   │   └── f1_macro/
│   │       └── evaluation_sorted_by_f1_macro.csv
│   │
│   ├── reports/                         (Training details)
│   │   ├── training_summary.json
│   │   ├── feature_importances.csv
│   │   ├── keras_history.json
│   │   └── report.pdf
│   │
│   └── confusion_matrices/              (Prediction accuracy breakdowns)
│       ├── confusion_label_tertiles_rf.png
│       ├── confusion_label_tertiles_lgbm.png
│       ├── confusion_label_tertiles_keras.png
│       ├── confusion_label_weighted_rf.png
│       ├── confusion_label_weighted_lgbm.png
│       ├── confusion_label_weighted_keras.png
│       ├── confusion_label_kmeans_rf.png
│       ├── confusion_label_kmeans_lgbm.png
│       └── confusion_label_kmeans_keras.png
│
├── 6_Visualizations/                    [Charts & graphs] 📈
│   ├── README.md
│   │
│   ├── accuracy_comparison.png          (Compare by accuracy)
│   ├── precision_macro_comparison.png   (Compare by precision)
│   ├── recall_macro_comparison.png      (Compare by recall)
│   ├── f1_macro_comparison.png          (Compare by F1-score)
│   │
│   ├── accuracy_individual_bars.png     (Detailed accuracy ranking)
│   ├── precision_macro_individual_bars.png
│   ├── recall_macro_individual_bars.png
│   ├── f1_macro_individual_bars.png
│   │
│   ├── all_metrics_heatmap.png          🔥 COMPREHENSIVE HEATMAP
│   ├── model_comparison_radar.png       🎯 RADAR CHART
│   ├── performance_summary_report.txt   📝 TEXT SUMMARY
│   │
│   └── models/                          (Model-specific data)
│       ├── RandomForest/
│       │   ├── RandomForest_all_metrics.csv
│       │   └── RandomForest_summary.json
│       └── LightGBM/
│           ├── LightGBM_all_metrics.csv
│           └── LightGBM_summary.json
│
├── 7_Documentation/                     [Reference materials]
│   ├── MetricsData.ipynb                (Jupyter notebook)
│   └── original_README.md               (Original setup guide)
│
└── Code/                                [ORIGINAL BACKUP - All files preserved]
    └── (Original unorganized structure - kept for reference)
```

---

## 🎯 Quick Navigation Guide

| What You Need | Where to Look |
|---------------|---------------|
| **See all model performance** | `5_Results/metrics/evaluation_summary.csv` |
| **Best model info** | `6_Visualizations/models/LightGBM/LightGBM_summary.json` |
| **Charts for presentation** | `6_Visualizations/*.png` |
| **Confusion matrices** | `5_Results/confusion_matrices/` |
| **Load a trained model** | `4_Trained_Models/classification_models/label_kmeans_lgbm.joblib` |
| **Run training** | `3_Scripts/model_training/train_classifiers.py` |
| **Generate new charts** | `3_Scripts/evaluation/visualize_metrics.py` |
| **Original data** | `1_Raw_Data/All thesis data - labeled.csv` |

---

## 🏆 Best Results Quick Reference

**Top Performer: LightGBM + K-means Labeling**
- Location: `4_Trained_Models/classification_models/label_kmeans_lgbm.joblib`
- Accuracy: 97.86%
- Precision: 98.40%
- Recall: 98.53%
- F1-Score: 98.47%

**Summary:** `6_Visualizations/models/LightGBM/LightGBM_summary.json`

---

**Legend:**
📖 Documentation | 📊 Data | 📈 Charts | 🔥 Important | ⭐ Best | 🎯 Key File
