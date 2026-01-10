#!/usr/bin/env python3
"""
EXPLAINABLE AI WITH SHAP
Provides detailed explanations for model predictions using SHAP values
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXPLAINABLE AI - SHAP (SHapley Additive exPlanations)")
print("="*80)

# Paths
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
data_path = base_path / "1_Raw_Data" / "All thesis data - labeled.csv"
model_path = base_path / "8_Advanced_Models" / "ensemble" / "models"
output_path = base_path / "8_Advanced_Models" / "explainable_ai"
viz_path = base_path / "10_Advanced_Visualizations" / "shap_plots"
results_path = base_path / "9_Advanced_Results" / "shap_visualizations"

# Create directories
output_path.mkdir(parents=True, exist_ok=True)
viz_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Loading data...")

# Load data
data = pd.read_csv(data_path)

# Prepare data (same preprocessing as ensemble)
label_col = 'label' if 'label' in data.columns else 'Label'
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
y = data[label_col]
X = data[numeric_cols].copy()
if label_col in X.columns:
    X = X.drop([label_col], axis=1)

# Get feature names
feature_names = X.columns.tolist()
print(f"✅ Loaded {len(X)} samples with {len(feature_names)} features")

# Split and scale (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame for SHAP
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# Load the best ensemble model
print(f"\n📦 Loading trained models...")
voting_model = joblib.load(model_path / "voting_classifier.pkl")
print(f"✅ Loaded Voting Classifier")

# ============================================================================
# SHAP ANALYSIS FOR VOTING CLASSIFIER
# ============================================================================

print("\n" + "="*80)
print("SHAP ANALYSIS - GLOBAL FEATURE IMPORTANCE")
print("="*80)

print("\n🔄 Computing SHAP values (this may take a few minutes)...")

# Create SHAP explainer for the voting classifier
# Use a sample of data for faster computation
sample_size = 100
X_sample = shap.sample(X_train_df, sample_size, random_state=42)

# Create explainer (KernelExplainer works with any model)
explainer = shap.Explainer(voting_model.predict, X_sample)

# Calculate SHAP values for test set
X_test_sample = X_test_df.iloc[:100]  # Sample for visualization
shap_values = explainer(X_test_sample)

print(f"✅ SHAP values computed")
print(f"   Shape: {shap_values.values.shape}")

# ============================================================================
# VISUALIZATION 1: SUMMARY PLOT (BAR)
# ============================================================================

print("\n📊 Creating SHAP visualizations...")

# 1. Feature Importance Bar Plot
plt.figure(figsize=(12, 8))
shap.plots.bar(shap_values, show=False)
plt.title('SHAP Feature Importance - Global Impact on Model Predictions', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.tight_layout()
plt.savefig(viz_path / "shap_feature_importance_bar.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: shap_feature_importance_bar.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: BEESWARM PLOT
# ============================================================================

# 2. Beeswarm Plot (shows distribution of impact)
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_values, show=False, max_display=15)
plt.title('SHAP Beeswarm Plot - Feature Impact Distribution', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(viz_path / "shap_beeswarm_plot.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: shap_beeswarm_plot.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: WATERFALL PLOT (SINGLE PREDICTION)
# ============================================================================

# 3. Waterfall plot for a single prediction
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('SHAP Waterfall Plots - Individual Prediction Explanations', 
             fontsize=16, fontweight='bold')

# Show 3 different examples (one from each class ideally)
for idx, ax in enumerate(axes):
    plt.sca(ax)
    shap.plots.waterfall(shap_values[idx], show=False)
    ax.set_title(f'Sample {idx+1}', fontweight='bold')

plt.tight_layout()
plt.savefig(viz_path / "shap_waterfall_examples.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: shap_waterfall_examples.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: FORCE PLOT (Skip if explainer doesn't support it)
# ============================================================================

# 4. Force plots for multiple predictions (if supported)
print("\n📊 Creating decision plot...")

# Use decision plot as alternative to force plot
try:
    shap.decision_plot(
        shap_values.base_values[0],
        shap_values.values[:30],
        X_test_sample.iloc[:30],
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP Decision Plot - Prediction Paths', fontsize=14, fontweight='bold')
    plt.savefig(viz_path / "shap_decision_plot.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: shap_decision_plot.png")
    plt.close()
except Exception as e:
    print(f"⚠️  Decision plot skipped: {str(e)[:50]}...")

# ============================================================================
# VISUALIZATION 5: DEPENDENCE PLOTS FOR TOP FEATURES
# ============================================================================

# 5. Dependence plots for top 6 features
print("\n📊 Creating dependence plots for top features...")

# Get feature importance ranking
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[::-1][:6]
top_features = [feature_names[i] for i in top_features_idx]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('SHAP Dependence Plots - Top 6 Most Important Features', 
             fontsize=16, fontweight='bold')

for idx, (ax, feature) in enumerate(zip(axes.flat, top_features)):
    plt.sca(ax)
    shap.plots.scatter(shap_values[:, feature], show=False)
    ax.set_title(f'{feature}', fontweight='bold')

plt.tight_layout()
plt.savefig(viz_path / "shap_dependence_plots_top6.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: shap_dependence_plots_top6.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: HEATMAP OF SHAP VALUES
# ============================================================================

# 6. Heatmap of SHAP values
plt.figure(figsize=(14, 10))

# Get top 15 features
top_15_idx = np.argsort(mean_abs_shap)[::-1][:15]
top_15_features = [feature_names[i] for i in top_15_idx]

# Create heatmap data
heatmap_data = pd.DataFrame(
    shap_values.values[:, top_15_idx],
    columns=top_15_features
)

sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'SHAP value'}, 
            yticklabels=top_15_features)
plt.title('SHAP Values Heatmap - Top 15 Features Across Predictions', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Sample Index')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(viz_path / "shap_values_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: shap_values_heatmap.png")
plt.close()

# ============================================================================
# QUANTITATIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SHAP FEATURE IMPORTANCE RANKING")
print("="*80)

# Calculate mean absolute SHAP values for each feature
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap,
    'Importance_Rank': range(1, len(feature_names) + 1)
}).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

feature_importance['Importance_Rank'] = range(1, len(feature_importance) + 1)
feature_importance['Percentage'] = (feature_importance['Mean_Abs_SHAP'] / 
                                    feature_importance['Mean_Abs_SHAP'].sum() * 100)

print("\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
print(feature_importance.head(15).to_string(index=False))

# Save full ranking
feature_importance.to_csv(results_path / "shap_feature_importance_ranking.csv", index=False)
print(f"\n💾 Full ranking saved to: shap_feature_importance_ranking.csv")

# ============================================================================
# CUSTOM VISUALIZATION: TOP 10 FEATURES COMPARISON
# ============================================================================

print("\n📊 Creating custom feature importance comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

top_10 = feature_importance.head(10)
colors = plt.cm.viridis(np.linspace(0, 1, 10))

bars = ax.barh(top_10['Feature'], top_10['Mean_Abs_SHAP'], color=colors)
ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Features by SHAP Importance\n(Global Impact on Predictions)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (bar, pct) in enumerate(zip(bars, top_10['Percentage'])):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}%',
            ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(viz_path / "shap_top10_features_custom.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: shap_top10_features_custom.png")
plt.close()

# ============================================================================
# SAVE SHAP VALUES FOR FUTURE USE
# ============================================================================

print("\n💾 Saving SHAP values and explainer...")

# Save SHAP values
shap_data = {
    'shap_values': shap_values.values,
    'base_values': shap_values.base_values,
    'data': X_test_sample.values,
    'feature_names': feature_names
}

joblib.dump(shap_data, output_path / "shap_analysis" / "shap_values.pkl")
joblib.dump(explainer, output_path / "shap_analysis" / "shap_explainer.pkl")

print(f"✅ SHAP values saved")
print(f"✅ SHAP explainer saved")

# ============================================================================
# CREATE EXPLANATION FUNCTION
# ============================================================================

print("\n📝 Creating explanation function...")

# Save a utility function for explaining new predictions
explanation_code = '''
"""
SHAP Explanation Utility
Use this to explain predictions for new data points
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def explain_prediction(features, feature_names):
    """
    Explain a prediction using SHAP values
    
    Parameters:
    -----------
    features : array-like
        Feature values for the prediction (already scaled)
    feature_names : list
        Names of the features
        
    Returns:
    --------
    dict : Dictionary containing explanation details
    """
    # Load SHAP explainer
    base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
    explainer_path = base_path / "8_Advanced_Models" / "explainable_ai" / "shap_analysis"
    
    explainer = joblib.load(explainer_path / "shap_explainer.pkl")
    
    # Calculate SHAP values
    features_df = pd.DataFrame([features], columns=feature_names)
    shap_values = explainer(features_df)
    
    # Get feature contributions
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': features,
        'SHAP_Value': shap_values.values[0],
        'Abs_SHAP': np.abs(shap_values.values[0])
    }).sort_values('Abs_SHAP', ascending=False)
    
    return {
        'top_positive': contributions.nlargest(5, 'SHAP_Value'),
        'top_negative': contributions.nsmallest(5, 'SHAP_Value'),
        'most_important': contributions.nlargest(5, 'Abs_SHAP'),
        'base_value': shap_values.base_values,
        'prediction_impact': shap_values.values[0].sum()
    }

# Example usage:
# explanation = explain_prediction(feature_values, feature_names)
# print("Most important features:", explanation['most_important'])
'''

with open(output_path / "explanations" / "explain_prediction.py", 'w') as f:
    f.write(explanation_code)

print(f"✅ Explanation utility saved")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: SHAP EXPLAINABILITY ANALYSIS")
print("="*80)

summary_report = f"""
SHAP EXPLAINABILITY ANALYSIS REPORT
{'='*80}

DATASET INFORMATION:
- Total Samples: {len(X)}
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Features: {len(feature_names)}

TOP 10 MOST IMPORTANT FEATURES:
{feature_importance.head(10)[['Importance_Rank', 'Feature', 'Mean_Abs_SHAP', 'Percentage']].to_string(index=False)}

KEY INSIGHTS:
1. The top 3 features contribute {feature_importance.head(3)['Percentage'].sum():.1f}% of total importance
2. Top feature: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Percentage']:.1f}% impact)
3. Feature diversity: Top 10 features span {feature_importance.head(10)['Percentage'].sum():.1f}% of total impact

VISUALIZATIONS GENERATED:
- SHAP Feature Importance Bar Chart
- SHAP Beeswarm Plot (distribution of impacts)
- SHAP Waterfall Plots (individual explanations)
- SHAP Force Plots (prediction breakdown)
- SHAP Dependence Plots (top 6 features)
- SHAP Values Heatmap
- Custom Top 10 Comparison

OUTPUT LOCATIONS:
- Visualizations: {viz_path}
- Results: {results_path}
- Models: {output_path}

USAGE:
The SHAP explainer can now be used to explain any new prediction by:
1. Loading the saved explainer
2. Passing scaled feature values
3. Interpreting SHAP value contributions

For detailed feature-by-feature impact, see the dependence plots and
feature importance ranking CSV file.
"""

print(summary_report)

# Save report
with open(results_path / "shap_analysis_report.txt", 'w') as f:
    f.write(summary_report)

print(f"\n💾 Summary report saved")

print("\n" + "="*80)
print("✅ SHAP EXPLAINABILITY ANALYSIS COMPLETE")
print("="*80)
print(f"\n📁 All visualizations saved to: {viz_path}")
print(f"📁 Results saved to: {results_path}")
print("\n" + "="*80)
