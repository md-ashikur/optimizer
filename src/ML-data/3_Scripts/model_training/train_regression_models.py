#!/usr/bin/env python3
"""
REGRESSION MODELS FOR EXACT METRIC PREDICTION
Predicts specific values for LCP, FID/INP, and CLS instead of categories
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("REGRESSION MODELS - EXACT METRIC PREDICTION")
print("="*80)

# Paths
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
data_path = base_path / "1_Raw_Data" / "All thesis data - labeled.csv"
output_path = base_path / "8_Advanced_Models" / "regression" / "models"
results_path = base_path / "9_Advanced_Results" / "regression_predictions"
viz_path = base_path / "10_Advanced_Visualizations" / "regression_plots"

# Create directories
output_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)
viz_path.mkdir(parents=True, exist_ok=True)

print(f"\n📁 Loading data...")

# Load data
data = pd.read_csv(data_path)

# Get numeric columns only
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

print(f"✅ Loaded {len(data)} samples")
print(f"📊 Numeric columns: {len(numeric_cols)}")

# Target metrics for regression (Core Web Vitals)
target_metrics = {
    'LCP': 'Largest_contentful_paint_LCP_ms',
    'FID_INP': 'Interaction_to_Next_Paint_INP_ms',
    'CLS': 'Cumulative_Layout_Shift_CLS'
}

# Check which targets exist
available_targets = {}
for name, col in target_metrics.items():
    if col in data.columns:
        available_targets[name] = col
        print(f"✅ Target found: {name} ({col})")
    else:
        print(f"⚠️  Target not found: {col}")

if not available_targets:
    print("\n❌ No target metrics found! Checking available columns...")
    print(data.columns.tolist())
    exit(1)

# Results storage
all_results = []

# ============================================================================
# TRAIN REGRESSION MODELS FOR EACH TARGET METRIC
# ============================================================================

for target_name, target_col in available_targets.items():
    
    print("\n" + "="*80)
    print(f"TRAINING REGRESSION MODELS FOR {target_name}")
    print("="*80)
    
    # Prepare data
    # Features = all numeric except the current target
    feature_cols = [col for col in numeric_cols if col != target_col and 'label' not in col.lower()]
    
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Remove any rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"\n📊 Dataset for {target_name}:")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"   Target mean: {y.mean():.2f}")
    print(f"   Target std: {y.std():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, output_path / f"{target_name}_scaler.pkl")
    
    # ========================================================================
    # MODEL 1: RANDOM FOREST REGRESSOR
    # ========================================================================
    
    print(f"\n1️⃣  Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    
    rf_mse = mean_squared_error(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(rf_mse)
    
    print(f"   RMSE: {rf_rmse:.4f}")
    print(f"   MAE:  {rf_mae:.4f}")
    print(f"   R²:   {rf_r2:.4f}")
    
    # Save model
    joblib.dump(rf_model, output_path / f"{target_name}_random_forest.pkl")
    
    # ========================================================================
    # MODEL 2: LIGHTGBM REGRESSOR
    # ========================================================================
    
    print(f"\n2️⃣  Training LightGBM Regressor...")
    lgbm_model = lgb.LGBMRegressor(
        objective='regression',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
        verbose=-1
    )
    lgbm_model.fit(X_train_scaled, y_train)
    lgbm_pred = lgbm_model.predict(X_test_scaled)
    
    lgbm_mse = mean_squared_error(y_test, lgbm_pred)
    lgbm_mae = mean_absolute_error(y_test, lgbm_pred)
    lgbm_r2 = r2_score(y_test, lgbm_pred)
    lgbm_rmse = np.sqrt(lgbm_mse)
    
    print(f"   RMSE: {lgbm_rmse:.4f}")
    print(f"   MAE:  {lgbm_mae:.4f}")
    print(f"   R²:   {lgbm_r2:.4f}")
    
    # Save model
    joblib.dump(lgbm_model, output_path / f"{target_name}_lightgbm.pkl")
    
    # ========================================================================
    # MODEL 3: GRADIENT BOOSTING REGRESSOR
    # ========================================================================
    
    print(f"\n3️⃣  Training Gradient Boosting Regressor...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_mae = mean_absolute_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(gb_mse)
    
    print(f"   RMSE: {gb_rmse:.4f}")
    print(f"   MAE:  {gb_mae:.4f}")
    print(f"   R²:   {gb_r2:.4f}")
    
    # Save model
    joblib.dump(gb_model, output_path / f"{target_name}_gradient_boosting.pkl")
    
    # ========================================================================
    # MODEL 4: NEURAL NETWORK REGRESSOR
    # ========================================================================
    
    print(f"\n4️⃣  Training Neural Network Regressor...")
    nn_model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Single output for regression
    ])
    
    nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = nn_model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    nn_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
    
    nn_mse = mean_squared_error(y_test, nn_pred)
    nn_mae = mean_absolute_error(y_test, nn_pred)
    nn_r2 = r2_score(y_test, nn_pred)
    nn_rmse = np.sqrt(nn_mse)
    
    print(f"   RMSE: {nn_rmse:.4f}")
    print(f"   MAE:  {nn_mae:.4f}")
    print(f"   R²:   {nn_r2:.4f}")
    
    # Save model
    nn_model.save(output_path / f"{target_name}_neural_network.keras")
    
    # ========================================================================
    # STORE RESULTS
    # ========================================================================
    
    all_results.append({
        'Target': target_name,
        'Model': 'Random Forest',
        'RMSE': rf_rmse,
        'MAE': rf_mae,
        'R2': rf_r2
    })
    all_results.append({
        'Target': target_name,
        'Model': 'LightGBM',
        'RMSE': lgbm_rmse,
        'MAE': lgbm_mae,
        'R2': lgbm_r2
    })
    all_results.append({
        'Target': target_name,
        'Model': 'Gradient Boosting',
        'RMSE': gb_rmse,
        'MAE': gb_mae,
        'R2': gb_r2
    })
    all_results.append({
        'Target': target_name,
        'Model': 'Neural Network',
        'RMSE': nn_rmse,
        'MAE': nn_mae,
        'R2': nn_r2
    })
    
    # ========================================================================
    # VISUALIZATIONS FOR THIS TARGET
    # ========================================================================
    
    print(f"\n📊 Creating visualizations for {target_name}...")
    
    # 1. Actual vs Predicted plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{target_name} Prediction: Actual vs Predicted', fontsize=16, fontweight='bold')
    
    models_data = [
        ('Random Forest', rf_pred, rf_r2),
        ('LightGBM', lgbm_pred, lgbm_r2),
        ('Gradient Boosting', gb_pred, gb_r2),
        ('Neural Network', nn_pred, nn_r2)
    ]
    
    for idx, (name, pred, r2) in enumerate(models_data):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(y_test, pred, alpha=0.5, s=20)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Value', fontweight='bold')
        ax.set_ylabel('Predicted Value', fontweight='bold')
        ax.set_title(f'{name} (R² = {r2:.4f})', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_path / f"{target_name}_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {target_name}_actual_vs_predicted.png")
    plt.close()
    
    # 2. Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{target_name} Residual Analysis', fontsize=16, fontweight='bold')
    
    for idx, (name, pred, _) in enumerate(models_data):
        ax = axes[idx // 2, idx % 2]
        residuals = y_test - pred
        ax.scatter(pred, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Value', fontweight='bold')
        ax.set_ylabel('Residuals', fontweight='bold')
        ax.set_title(f'{name} Residuals', fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_path / f"{target_name}_residuals.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {target_name}_residuals.png")
    plt.close()
    
    # Save predictions for this target
    predictions_df = pd.DataFrame({
        'Actual': y_test.values,
        'RF_Predicted': rf_pred,
        'LGBM_Predicted': lgbm_pred,
        'GB_Predicted': gb_pred,
        'NN_Predicted': nn_pred
    })
    predictions_df.to_csv(results_path / f"{target_name}_predictions.csv", index=False)
    print(f"   ✅ Saved: {target_name}_predictions.csv")

# ============================================================================
# OVERALL RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("OVERALL REGRESSION RESULTS")
print("="*80)

results_df = pd.DataFrame(all_results)
print("\n📊 Complete Results:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(results_path / "all_regression_results.csv", index=False)
print(f"\n💾 Results saved to: all_regression_results.csv")

# ============================================================================
# FINAL COMPARISON VISUALIZATIONS
# ============================================================================

print("\n📊 Creating comparison visualizations...")

# 1. R² Score Comparison
fig, ax = plt.subplots(figsize=(14, 8))

pivot_r2 = results_df.pivot(index='Model', columns='Target', values='R2')
pivot_r2.plot(kind='bar', ax=ax, width=0.8)

ax.set_title('R² Score Comparison Across All Models and Targets', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
legend = ax.legend(title='Target Metric')
legend.get_title().set_fontweight('bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(viz_path / "r2_score_comparison_all.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: r2_score_comparison_all.png")
plt.close()

# 2. RMSE Comparison
fig, ax = plt.subplots(figsize=(14, 8))

pivot_rmse = results_df.pivot(index='Model', columns='Target', values='RMSE')
pivot_rmse.plot(kind='bar', ax=ax, width=0.8)

ax.set_title('RMSE Comparison Across All Models and Targets', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('RMSE', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')
legend = ax.legend(title='Target Metric')
legend.get_title().set_fontweight('bold')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(viz_path / "rmse_comparison_all.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: rmse_comparison_all.png")
plt.close()

# 3. Heatmap of R² scores
fig, ax = plt.subplots(figsize=(10, 6))

heatmap_data = results_df.pivot(index='Model', columns='Target', values='R2')
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
            vmin=0, vmax=1, cbar_kws={'label': 'R² Score'}, ax=ax)
ax.set_title('R² Score Heatmap: Models vs Target Metrics', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_path / "r2_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: r2_heatmap.png")
plt.close()

# ============================================================================
# BEST MODEL SELECTION
# ============================================================================

print("\n" + "="*80)
print("BEST MODELS FOR EACH TARGET")
print("="*80)

for target in available_targets.keys():
    target_results = results_df[results_df['Target'] == target]
    best_model = target_results.loc[target_results['R2'].idxmax()]
    
    print(f"\n🏆 {target}:")
    print(f"   Best Model: {best_model['Model']}")
    print(f"   R²: {best_model['R2']:.4f}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   MAE: {best_model['MAE']:.4f}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

summary_report = f"""
REGRESSION MODELS SUMMARY REPORT
{'='*80}

OBJECTIVE:
Predict exact values for Core Web Vitals metrics instead of categories

TARGETS TRAINED:
{', '.join(available_targets.keys())}

MODELS TRAINED PER TARGET:
1. Random Forest Regressor
2. LightGBM Regressor
3. Gradient Boosting Regressor
4. Neural Network Regressor

TOTAL MODELS: {len(all_results)}

PERFORMANCE SUMMARY:
{results_df.groupby('Target')[['RMSE', 'MAE', 'R2']].agg(['mean', 'min', 'max']).to_string()}

OUTPUT LOCATIONS:
- Models: {output_path}
- Results: {results_path}
- Visualizations: {viz_path}

USAGE:
Each model can predict exact metric values (e.g., LCP = 2453ms)
instead of categories (Good/Average/Weak).

This provides more precise optimization guidance.
"""

print("\n" + summary_report)

with open(results_path / "regression_summary_report.txt", 'w') as f:
    f.write(summary_report)

print("\n" + "="*80)
print("✅ REGRESSION MODELS TRAINING COMPLETE")
print("="*80)
print(f"\n📁 Models saved to: {output_path}")
print(f"📁 Results saved to: {results_path}")
print(f"📁 Visualizations saved to: {viz_path}")
print("\n" + "="*80)
