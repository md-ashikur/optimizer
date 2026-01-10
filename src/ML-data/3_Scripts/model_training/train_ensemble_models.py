#!/usr/bin/env python3
"""
ENSEMBLE MODELS IMPLEMENTATION
Combines multiple ML models using Voting and Stacking approaches
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ENSEMBLE MODELS TRAINING - COMBINING MULTIPLE MODELS")
print("="*80)

# Paths
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
data_path = base_path / "1_Raw_Data" / "All thesis data - labeled.csv"
output_path = base_path / "8_Advanced_Models" / "ensemble"
results_path = base_path / "9_Advanced_Results" / "ensemble_performance"
viz_path = base_path / "10_Advanced_Visualizations" / "ensemble_comparison"

# Create directories
output_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)
viz_path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Loading data from: {data_path}")

# Load data
data = pd.read_csv(data_path)
print(f"✅ Loaded {len(data)} samples")

# Prepare features and labels
# First, identify the label column
label_col = 'Label' if 'Label' in data.columns else 'label'
if label_col not in data.columns:
    label_col = data.columns[-1]

# Drop non-numeric columns (like URLs, website names, etc.)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\n🔍 Column Analysis:")
print(f"  Total columns: {len(data.columns)}")
print(f"  Numeric columns: {len(numeric_cols)}")
print(f"  Non-numeric columns: {len(non_numeric_cols)}")
if non_numeric_cols:
    print(f"  Non-numeric: {non_numeric_cols}")

# Extract label
y = data[label_col]

# Keep only numeric features, excluding the label
X = data[numeric_cols].copy()
if label_col in X.columns:
    X = X.drop([label_col], axis=1)

print(f"\n📊 Dataset Info:")
print(f"  Features: {X.shape[1]}")
print(f"  Samples: {len(X)}")
print(f"  Classes: {y.nunique()}")
print(f"  Class distribution:\n{y.value_counts()}")

# Encode labels if they're strings
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Train set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# ============================================================================
# MODEL 1: VOTING CLASSIFIER (Soft Voting)
# ============================================================================

print("\n" + "="*80)
print("TRAINING VOTING CLASSIFIER (Soft Voting)")
print("="*80)

# Define base models
print("\n1️⃣  Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_score = f1_score(y_test, rf_pred, average='macro')
print(f"   ✅ Random Forest F1-Score: {rf_score:.4f}")

print("\n2️⃣  Training LightGBM...")
lgbm_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(class_names),
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_train_scaled, y_train)
lgbm_pred = lgbm_model.predict(X_test_scaled)
lgbm_score = f1_score(y_test, lgbm_pred, average='macro')
print(f"   ✅ LightGBM F1-Score: {lgbm_score:.4f}")

print("\n3️⃣  Training Neural Network...")
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

nn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

nn_pred = np.argmax(nn_model.predict(X_test_scaled, verbose=0), axis=1)
nn_score = f1_score(y_test, nn_pred, average='macro')
print(f"   ✅ Neural Network F1-Score: {nn_score:.4f}")

# Create Voting Classifier
print("\n4️⃣  Creating Soft Voting Ensemble...")

voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('lgbm', lgbm_model)
    ],
    voting='soft',
    n_jobs=-1
)

voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)

# Evaluate Voting Classifier
voting_metrics = {
    'accuracy': accuracy_score(y_test, voting_pred),
    'precision': precision_score(y_test, voting_pred, average='macro'),
    'recall': recall_score(y_test, voting_pred, average='macro'),
    'f1_score': f1_score(y_test, voting_pred, average='macro')
}

print(f"\n✅ VOTING CLASSIFIER RESULTS:")
print(f"   Accuracy:  {voting_metrics['accuracy']:.4f}")
print(f"   Precision: {voting_metrics['precision']:.4f}")
print(f"   Recall:    {voting_metrics['recall']:.4f}")
print(f"   F1-Score:  {voting_metrics['f1_score']:.4f}")

# Save Voting Classifier
joblib.dump(voting_clf, output_path / "models" / "voting_classifier.pkl")
joblib.dump(scaler, output_path / "models" / "ensemble_scaler.pkl")
joblib.dump(le, output_path / "models" / "label_encoder.pkl")
print(f"\n💾 Voting Classifier saved")

# ============================================================================
# MODEL 2: STACKING CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("TRAINING STACKING CLASSIFIER")
print("="*80)

# Create Stacking Classifier with LightGBM as meta-learner
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('lgbm', lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100, random_state=42, verbose=-1))
    ],
    final_estimator=lgb.LGBMClassifier(num_leaves=15, learning_rate=0.1, n_estimators=50, random_state=42, verbose=-1),
    cv=5,
    n_jobs=-1
)

print("\n🔄 Training stacking ensemble with 5-fold CV...")
stacking_clf.fit(X_train_scaled, y_train)
stacking_pred = stacking_clf.predict(X_test_scaled)

# Evaluate Stacking Classifier
stacking_metrics = {
    'accuracy': accuracy_score(y_test, stacking_pred),
    'precision': precision_score(y_test, stacking_pred, average='macro'),
    'recall': recall_score(y_test, stacking_pred, average='macro'),
    'f1_score': f1_score(y_test, stacking_pred, average='macro')
}

print(f"\n✅ STACKING CLASSIFIER RESULTS:")
print(f"   Accuracy:  {stacking_metrics['accuracy']:.4f}")
print(f"   Precision: {stacking_metrics['precision']:.4f}")
print(f"   Recall:    {stacking_metrics['recall']:.4f}")
print(f"   F1-Score:  {stacking_metrics['f1_score']:.4f}")

# Save Stacking Classifier
joblib.dump(stacking_clf, output_path / "models" / "stacking_classifier.pkl")
print(f"\n💾 Stacking Classifier saved")

# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================

print("\n" + "="*80)
print("ENSEMBLE PERFORMANCE COMPARISON")
print("="*80)

# Compile all results
results_df = pd.DataFrame({
    'Model': ['Random Forest', 'LightGBM', 'Neural Network', 'Voting Ensemble', 'Stacking Ensemble'],
    'Accuracy': [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, lgbm_pred),
        accuracy_score(y_test, nn_pred),
        voting_metrics['accuracy'],
        stacking_metrics['accuracy']
    ],
    'Precision': [
        precision_score(y_test, rf_pred, average='macro'),
        precision_score(y_test, lgbm_pred, average='macro'),
        precision_score(y_test, nn_pred, average='macro'),
        voting_metrics['precision'],
        stacking_metrics['precision']
    ],
    'Recall': [
        recall_score(y_test, rf_pred, average='macro'),
        recall_score(y_test, lgbm_pred, average='macro'),
        recall_score(y_test, nn_pred, average='macro'),
        voting_metrics['recall'],
        stacking_metrics['recall']
    ],
    'F1-Score': [
        rf_score,
        lgbm_score,
        nn_score,
        voting_metrics['f1_score'],
        stacking_metrics['f1_score']
    ]
})

print("\n📊 Complete Performance Comparison:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(results_path / "ensemble_comparison_results.csv", index=False)
print(f"\n💾 Results saved to: {results_path / 'ensemble_comparison_results.csv'}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n📊 Generating visualizations...")

# 1. Performance Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ensemble Models Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(viz_path / "ensemble_performance_comparison.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: ensemble_performance_comparison.png")

# 2. Confusion Matrices for Ensemble Models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Ensemble Models Confusion Matrices', fontsize=16, fontweight='bold')

# Voting Classifier Confusion Matrix
cm_voting = confusion_matrix(y_test, voting_pred)
sns.heatmap(cm_voting, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=class_names, yticklabels=class_names)
axes[0].set_title(f'Voting Classifier (F1: {voting_metrics["f1_score"]:.4f})')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Stacking Classifier Confusion Matrix
cm_stacking = confusion_matrix(y_test, stacking_pred)
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=class_names, yticklabels=class_names)
axes[1].set_title(f'Stacking Classifier (F1: {stacking_metrics["f1_score"]:.4f})')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(viz_path / "ensemble_confusion_matrices.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: ensemble_confusion_matrices.png")

# 3. F1-Score Improvement Chart
fig, ax = plt.subplots(figsize=(10, 6))

models = results_df['Model'].tolist()
f1_scores = results_df['F1-Score'].tolist()

bars = ax.barh(models, f1_scores, color=colors)
ax.set_xlabel('F1-Score', fontweight='bold')
ax.set_title('F1-Score Comparison: Individual vs Ensemble Models', fontweight='bold', fontsize=14)
ax.set_xlim([0.85, 1.0])
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax.text(score + 0.002, bar.get_y() + bar.get_height()/2,
            f'{score:.4f}',
            va='center', fontweight='bold')

# Highlight ensemble models
for i in [3, 4]:  # Voting and Stacking indices
    bars[i].set_edgecolor('red')
    bars[i].set_linewidth(2)

plt.tight_layout()
plt.savefig(viz_path / "f1_score_improvement.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: f1_score_improvement.png")

# 4. Metrics Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='RdYlGn', 
            vmin=0.85, vmax=1.0, cbar_kws={'label': 'Score'},
            linewidths=0.5, ax=ax)
ax.set_title('All Metrics Heatmap: Ensemble vs Individual Models', fontweight='bold', fontsize=14)
ax.set_xlabel('Model')
ax.set_ylabel('Metric')

plt.tight_layout()
plt.savefig(viz_path / "ensemble_metrics_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: ensemble_metrics_heatmap.png")

# ============================================================================
# DETAILED CLASSIFICATION REPORTS
# ============================================================================

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS")
print("="*80)

print("\n🔹 VOTING CLASSIFIER:")
print(classification_report(y_test, voting_pred, target_names=class_names))

print("\n🔹 STACKING CLASSIFIER:")
print(classification_report(y_test, stacking_pred, target_names=class_names))

# Save reports
with open(results_path / "voting_classifier_report.txt", 'w') as f:
    f.write("VOTING CLASSIFIER CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_test, voting_pred, target_names=class_names))

with open(results_path / "stacking_classifier_report.txt", 'w') as f:
    f.write("STACKING CLASSIFIER CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(classification_report(y_test, stacking_pred, target_names=class_names))

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ ENSEMBLE MODELS TRAINING COMPLETE")
print("="*80)

best_model = results_df.loc[results_df['F1-Score'].idxmax()]
print(f"\n🏆 BEST MODEL: {best_model['Model']}")
print(f"   F1-Score: {best_model['F1-Score']:.4f}")
print(f"   Accuracy: {best_model['Accuracy']:.4f}")

improvement = (results_df.loc[results_df['Model'].isin(['Voting Ensemble', 'Stacking Ensemble']), 'F1-Score'].max() - 
               results_df.loc[~results_df['Model'].isin(['Voting Ensemble', 'Stacking Ensemble']), 'F1-Score'].max())
print(f"\n📈 Ensemble Improvement over Best Individual Model: +{improvement:.4f}")

print(f"\n📁 Output Locations:")
print(f"   Models: {output_path / 'models'}")
print(f"   Results: {results_path}")
print(f"   Visualizations: {viz_path}")
print("\n" + "="*80)
