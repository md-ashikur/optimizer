# ML-Data Organized Structure

This folder contains all machine learning data, models, scripts, and results for the thesis project.

## 📁 Folder Structure

### **1_Raw_Data/**
Original, unprocessed dataset files
- `All thesis data - labeled.csv` - Original labeled dataset with performance metrics

### **2_Processed_Data/**
Cleaned and processed data, exploratory analysis
- `eda_analysis/` - Exploratory Data Analysis outputs
  - `basic_describe.csv` - Statistical summaries
  - `correlation_matrix.csv` - Feature correlations
  - `data_minmax_scaled.csv` - MinMax scaled data
  - `data_standard_scaled.csv` - Standard scaled data
  - `label_counts.csv` - Distribution of labels
  - `outliers_count.csv` - Outlier analysis
  - `per_label_means.csv` - Mean values per label
- `shap_analysis/` - SHAP (SHapley Additive exPlanations) values for model interpretability

### **3_Scripts/**
Python scripts for data processing, training, and evaluation

#### **3_Scripts/data_preparation/**
Scripts for data cleaning and preprocessing
- `clean_csv.py` - Data cleaning operations
- `impute_csv.py` - Missing value imputation
- `eda_and_label.py` - Exploratory data analysis and labeling

#### **3_Scripts/model_training/**
Scripts for training machine learning models
- `train_models.py` - Train regression models (RF, LightGBM, Keras)
- `train_classifiers.py` - Train classification models with different labeling strategies
- `tune_classifiers.py` - Hyperparameter tuning for classifiers

#### **3_Scripts/evaluation/**
Scripts for model evaluation and visualization
- `evaluate_models.py` - Evaluate trained models and generate metrics CSV
- `visualize_metrics.py` - Create charts and visualizations
- `generate_report.py` - Generate comprehensive performance report
- `api_predict.py` - API endpoint for making predictions

### **4_Trained_Models/**
Saved model files (.joblib, .h5) and scalers

#### **4_Trained_Models/regression_models/**
Models for regression tasks
- `model_rf.joblib` - Random Forest regression model
- `model_lgbm.joblib` - LightGBM regression model
- `model_keras.h5` - Keras neural network regression model
- `scaler.joblib` - Standard scaler for feature normalization

#### **4_Trained_Models/classification_models/**
Models for classification tasks (3 labeling strategies × 3 model types)
- `label_tertiles_*.joblib/.h5` - Models using tertile-based labeling
- `label_weighted_*.joblib/.h5` - Models using weighted scoring labeling
- `label_kmeans_*.joblib/.h5` - Models using K-means clustering labeling
- `*_scaler.joblib` - Scalers for each strategy

**Model naming convention:** `label_{strategy}_{model_type}.{extension}`
- Strategies: `tertiles`, `weighted`, `kmeans`
- Model types: `rf` (RandomForest), `lgbm` (LightGBM), `keras` (Neural Network)

### **5_Results/**
Model performance metrics, reports, and analysis

#### **5_Results/metrics/**
Performance metrics organized by metric type
- `evaluation_summary.csv` - Complete metrics for all models
- `classification_summary.json` - Summary of classification results
- `best_models_per_strategy.json` - Best performing model per labeling strategy
- `accuracy/` - Models sorted by accuracy
- `precision_macro/` - Models sorted by precision
- `recall_macro/` - Models sorted by recall
- `f1_macro/` - Models sorted by F1-score

#### **5_Results/reports/**
Training summaries and feature analysis
- `training_summary.json` - Training process details
- `feature_importances.csv` - Feature importance rankings
- `keras_history.json` - Keras training history (loss, accuracy per epoch)
- `report.pdf` - Comprehensive performance report (if generated)

#### **5_Results/confusion_matrices/**
Confusion matrix visualizations for all model-strategy combinations
- `confusion_label_{strategy}_{model}.png` - Confusion matrix images

### **6_Visualizations/**
Charts, graphs, and model-specific visual comparisons

#### **Main Visualizations:**
- `accuracy_comparison.png` - Accuracy comparison across models and strategies
- `precision_macro_comparison.png` - Precision comparison
- `recall_macro_comparison.png` - Recall comparison
- `f1_macro_comparison.png` - F1-score comparison
- `accuracy_individual_bars.png` - Detailed accuracy bar chart
- `precision_macro_individual_bars.png` - Detailed precision bar chart
- `recall_macro_individual_bars.png` - Detailed recall bar chart
- `f1_macro_individual_bars.png` - Detailed F1-score bar chart
- `all_metrics_heatmap.png` - Heatmap of all metrics
- `model_comparison_radar.png` - Radar chart comparing best models
- `performance_summary_report.txt` - Text summary of all results

#### **models/**
Model-specific data and summaries
- `RandomForest/` - All RandomForest metrics and summary
  - `RandomForest_all_metrics.csv` - Complete metrics CSV
  - `RandomForest_summary.json` - Best performance summary
- `LightGBM/` - All LightGBM metrics and summary
  - `LightGBM_all_metrics.csv` - Complete metrics CSV
  - `LightGBM_summary.json` - Best performance summary

### **7_Documentation/**
Notebooks, README files, and project documentation
- `MetricsData.ipynb` - Jupyter notebook for metrics exploration
- `original_README.md` - Original setup and run instructions

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r ../../requirements.txt
```

### 2. Run Scripts
```bash
# Data preparation
python 3_Scripts/data_preparation/clean_csv.py
python 3_Scripts/data_preparation/impute_csv.py
python 3_Scripts/data_preparation/eda_and_label.py

# Model training
python 3_Scripts/model_training/train_classifiers.py

# Evaluation and visualization
python 3_Scripts/evaluation/evaluate_models.py
python 3_Scripts/evaluation/visualize_metrics.py
```

### 3. View Results
- **Metrics:** Check `5_Results/metrics/evaluation_summary.csv`
- **Charts:** View files in `6_Visualizations/`
- **Model Performance:** See `6_Visualizations/performance_summary_report.txt`

---

## 📊 Best Model Performance

**LightGBM with K-means Labeling:**
- Accuracy: **97.86%**
- Precision: **98.40%**
- Recall: **98.53%**
- F1-Score: **98.47%**

See `6_Visualizations/models/LightGBM/LightGBM_summary.json` for details.

---

## 📝 Notes

- All original files remain in the `Code/` folder (backup)
- The organized structure mirrors the ML workflow: Raw → Processed → Scripts → Models → Results → Visualizations
- Each folder is numbered for easy sequential understanding
- Model files use consistent naming: `{strategy}_{model_type}.{extension}`

---

**Last Updated:** January 7, 2026
**Thesis Project:** Web Performance Optimization ML Models
