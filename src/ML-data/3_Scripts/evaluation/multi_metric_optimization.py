#!/usr/bin/env python3
"""
MULTI-METRIC OPTIMIZATION
Balances tradeoffs between different performance metrics using Pareto optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MULTI-METRIC OPTIMIZATION - PARETO ANALYSIS")
print("="*80)

# Paths
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
data_path = base_path / "1_Raw_Data" / "All thesis data - labeled.csv"
output_path = base_path / "8_Advanced_Models" / "multi_metric_optimizer" / "models"
results_path = base_path / "9_Advanced_Results" / "optimization_reports"
viz_path = base_path / "10_Advanced_Visualizations" / "pareto_frontiers"

# Create directories
for path in [output_path, results_path, viz_path]:
    path.mkdir(parents=True, exist_ok=True)

print("\nLoading data...")

# Load data
data = pd.read_csv(data_path)

# Get Core Web Vitals metrics
core_metrics = {
    'LCP': 'Largest_contentful_paint_LCP_ms',
    'FID': 'Interaction_to_Next_Paint_INP_ms',
    'CLS': 'Cumulative_Layout_Shift_CLS'
}

# Verify columns exist
available_metrics = {}
for name, col in core_metrics.items():
    if col in data.columns:
        available_metrics[name] = col
        print(f"Found: {name} ({col})")

if len(available_metrics) < 2:
    print("ERROR: Need at least 2 metrics for optimization")
    exit(1)

# Extract metrics
metrics_df = data[[col for col in available_metrics.values()]].copy()
metrics_df.columns = list(available_metrics.keys())

# Remove missing values
metrics_df = metrics_df.dropna()

print(f"\nDataset: {len(metrics_df)} samples")
print(f"Metrics: {list(available_metrics.keys())}")

# ============================================================================
# PARETO FRONTIER ANALYSIS
# ============================================================================

print("\nComputing Pareto frontier...")

def is_pareto_efficient(costs, return_mask=True):
    """
    Find the Pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of Pareto-efficient points or a mask
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point with a lower cost
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep self
    if return_mask:
        return is_efficient
    else:
        return np.where(is_efficient)[0]

# For optimization, we want to MINIMIZE all metrics
# So we use metrics as is (lower is better)
costs = metrics_df.values

# Find Pareto-efficient points
pareto_mask = is_pareto_efficient(costs)
pareto_points = metrics_df[pareto_mask].copy()
non_pareto_points = metrics_df[~pareto_mask].copy()

print(f"\nPareto-efficient points: {len(pareto_points)}")
print(f"Non-Pareto points: {len(non_pareto_points)}")
print(f"Pareto ratio: {len(pareto_points)/len(metrics_df)*100:.1f}%")

# Save Pareto points
pareto_points.to_csv(results_path / "pareto_optimal_websites.csv", index=False)
print("\nSaved Pareto-optimal websites")

# ============================================================================
# ANALYZE PARETO FRONTIER
# ============================================================================

print("\nAnalyzing Pareto frontier...")

# Statistics for Pareto vs Non-Pareto
comparison = pd.DataFrame({
    'Metric': list(available_metrics.keys()),
    'Pareto_Mean': pareto_points.mean().values,
    'Pareto_Std': pareto_points.std().values,
    'NonPareto_Mean': non_pareto_points.mean().values,
    'NonPareto_Std': non_pareto_points.std().values
})

comparison['Improvement'] = ((comparison['NonPareto_Mean'] - comparison['Pareto_Mean']) / 
                             comparison['NonPareto_Mean'] * 100)

print("\nPareto vs Non-Pareto Comparison:")
print(comparison.to_string(index=False))

comparison.to_csv(results_path / "pareto_comparison.csv", index=False)

# ============================================================================
# OPTIMIZATION STRATEGIES
# ============================================================================

print("\nDefining optimization strategies...")

# Define common optimization scenarios
optimization_strategies = {
    'BALANCED': {
        'description': 'Equal weight to all metrics',
        'weights': {metric: 1/len(available_metrics) for metric in available_metrics.keys()},
        'use_case': 'General-purpose optimization'
    },
    
    'LCP_FOCUSED': {
        'description': 'Prioritize loading performance',
        'weights': {'LCP': 0.6, 'FID': 0.2, 'CLS': 0.2},
        'use_case': 'Content-heavy sites (blogs, news)'
    },
    
    'INTERACTIVITY_FOCUSED': {
        'description': 'Prioritize responsiveness',
        'weights': {'LCP': 0.2, 'FID': 0.6, 'CLS': 0.2},
        'use_case': 'Interactive apps (SPAs, web apps)'
    },
    
    'STABILITY_FOCUSED': {
        'description': 'Prioritize visual stability',
        'weights': {'LCP': 0.2, 'FID': 0.2, 'CLS': 0.6},
        'use_case': 'E-commerce, forms'
    }
}

# Calculate composite scores for each strategy
for strategy_name, strategy in optimization_strategies.items():
    weights = strategy['weights']
    
    # Normalize metrics first (0-1 scale)
    scaler = StandardScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(metrics_df),
        columns=metrics_df.columns,
        index=metrics_df.index
    )
    
    # Calculate weighted score
    score = sum(normalized[metric] * weights.get(metric, 0) 
                for metric in available_metrics.keys())
    
    metrics_df[f'{strategy_name}_Score'] = score
    pareto_points[f'{strategy_name}_Score'] = score[pareto_mask]

print(f"\nDefined {len(optimization_strategies)} optimization strategies")

# Save strategies
import json
strategies_export = {
    name: {
        'description': details['description'],
        'weights': details['weights'],
        'use_case': details['use_case']
    }
    for name, details in optimization_strategies.items()
}

with open(output_path / "optimization_strategies.json", 'w') as f:
    json.dump(strategies_export, f, indent=2)

print("Saved optimization strategies")

# ============================================================================
# FIND BEST EXAMPLES FOR EACH STRATEGY
# ============================================================================

print("\nFinding best examples for each strategy...")

best_examples = {}

for strategy_name in optimization_strategies.keys():
    score_col = f'{strategy_name}_Score'
    
    # Find top 5 Pareto-optimal points for this strategy
    best_5 = pareto_points.nsmallest(5, score_col)[list(available_metrics.keys())]
    
    best_examples[strategy_name] = best_5.mean().to_dict()
    
    print(f"\n{strategy_name}:")
    print(f"  Best average metrics:")
    for metric, value in best_examples[strategy_name].items():
        print(f"    {metric}: {value:.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

# 1. 2D Pareto Frontiers (pairwise)
if len(available_metrics) >= 2:
    metrics_list = list(available_metrics.keys())
    
    # Plot all pairwise combinations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Pareto Frontiers - Pairwise Metric Comparisons', 
                 fontsize=16, fontweight='bold')
    
    combinations = [
        (metrics_list[0], metrics_list[1]),
        (metrics_list[0], metrics_list[2]) if len(metrics_list) > 2 else (metrics_list[0], metrics_list[1]),
        (metrics_list[1], metrics_list[2]) if len(metrics_list) > 2 else (metrics_list[0], metrics_list[1])
    ]
    
    for idx, (metric_x, metric_y) in enumerate(combinations):
        ax = axes[idx]
        
        # Plot non-Pareto points
        ax.scatter(non_pareto_points[metric_x], non_pareto_points[metric_y],
                  alpha=0.3, s=20, c='gray', label='Non-Pareto')
        
        # Plot Pareto points
        ax.scatter(pareto_points[metric_x], pareto_points[metric_y],
                  alpha=0.7, s=50, c='red', marker='*', label='Pareto Optimal')
        
        ax.set_xlabel(metric_x, fontweight='bold')
        ax.set_ylabel(metric_y, fontweight='bold')
        ax.set_title(f'{metric_x} vs {metric_y}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_path / "pareto_frontiers_2d.png", dpi=300, bbox_inches='tight')
    print("Saved: pareto_frontiers_2d.png")
    plt.close()

# 2. 3D Pareto Frontier (if we have 3 metrics)
if len(available_metrics) >= 3:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    metrics_list = list(available_metrics.keys())
    
    # Plot non-Pareto points
    ax.scatter(non_pareto_points[metrics_list[0]], 
               non_pareto_points[metrics_list[1]],
               non_pareto_points[metrics_list[2]],
               c='lightblue', alpha=0.3, s=20, label='Non-Pareto')
    
    # Plot Pareto points
    ax.scatter(pareto_points[metrics_list[0]], 
               pareto_points[metrics_list[1]],
               pareto_points[metrics_list[2]],
               c='red', alpha=0.8, s=100, marker='*', label='Pareto Optimal')
    
    ax.set_xlabel(metrics_list[0], fontweight='bold')
    ax.set_ylabel(metrics_list[1], fontweight='bold')
    ax.set_zlabel(metrics_list[2], fontweight='bold')
    ax.set_title('3D Pareto Frontier - Core Web Vitals', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(viz_path / "pareto_frontier_3d.png", dpi=300, bbox_inches='tight')
    print("Saved: pareto_frontier_3d.png")
    plt.close()

# 3. Strategy Comparison Heatmap
strategy_scores = pd.DataFrame(best_examples).T

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(strategy_scores, annot=True, fmt='.1f', cmap='RdYlGn_r',
            cbar_kws={'label': 'Metric Value (lower is better)'}, ax=ax)
ax.set_title('Optimization Strategies - Best Achievable Metrics', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Metric')
ax.set_ylabel('Strategy')
plt.tight_layout()
plt.savefig(viz_path / "optimization_strategies_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved: optimization_strategies_heatmap.png")
plt.close()

# 4. Improvement Potential Chart
fig, ax = plt.subplots(figsize=(10, 6))

comparison_subset = comparison[['Metric', 'Improvement']]
bars = ax.bar(comparison_subset['Metric'], comparison_subset['Improvement'], 
              color=['#e74c3c', '#3498db', '#2ecc71'])

ax.set_ylabel('Improvement Potential (%)', fontweight='bold')
ax.set_title('Average Improvement by Moving to Pareto Frontier', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(viz_path / "improvement_potential.png", dpi=300, bbox_inches='tight')
print("Saved: improvement_potential.png")
plt.close()

# ============================================================================
# CREATE OPTIMIZATION GUIDE
# ============================================================================

print("\nCreating optimization guide...")

guide_code = '''
"""
Multi-Metric Optimization Guide
Use this to find optimal balance between metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
results_path = base_path / "9_Advanced_Results" / "optimization_reports"
model_path = base_path / "8_Advanced_Models" / "multi_metric_optimizer" / "models"

# Load Pareto-optimal examples
pareto_sites = pd.read_csv(results_path / "pareto_optimal_websites.csv")

# Load optimization strategies
with open(model_path / "optimization_strategies.json", 'r') as f:
    strategies = json.load(f)

def recommend_strategy(site_type):
    """
    Recommend optimization strategy based on site type
    
    Parameters:
    -----------
    site_type : str
        One of: 'content', 'app', 'ecommerce', 'general'
        
    Returns:
    --------
    dict : Recommended strategy details
    """
    
    strategy_map = {
        'content': 'LCP_FOCUSED',
        'blog': 'LCP_FOCUSED',
        'news': 'LCP_FOCUSED',
        'app': 'INTERACTIVITY_FOCUSED',
        'spa': 'INTERACTIVITY_FOCUSED',
        'webapp': 'INTERACTIVITY_FOCUSED',
        'ecommerce': 'STABILITY_FOCUSED',
        'shop': 'STABILITY_FOCUSED',
        'form': 'STABILITY_FOCUSED',
        'general': 'BALANCED'
    }
    
    strategy_name = strategy_map.get(site_type.lower(), 'BALANCED')
    strategy = strategies[strategy_name]
    
    # Get benchmarks from Pareto-optimal sites
    benchmarks = {
        'LCP': pareto_sites['LCP'].quantile(0.25),  # Top 25%
        'FID': pareto_sites['FID'].quantile(0.25),
        'CLS': pareto_sites['CLS'].quantile(0.25)
    }
    
    return {
        'strategy': strategy_name,
        'description': strategy['description'],
        'weights': strategy['weights'],
        'use_case': strategy['use_case'],
        'target_benchmarks': benchmarks
    }

def find_similar_pareto_sites(current_metrics, n=5):
    """
    Find Pareto-optimal sites similar to current metrics
    
    Parameters:
    -----------
    current_metrics : dict
        Dictionary with 'LCP', 'FID', 'CLS' values
    n : int
        Number of similar sites to return
        
    Returns:
    --------
    DataFrame : Similar Pareto-optimal sites
    """
    
    # Calculate Euclidean distance
    distances = np.sqrt(
        (pareto_sites['LCP'] - current_metrics['LCP'])**2 +
        (pareto_sites['FID'] - current_metrics.get('FID', 0))**2 +
        (pareto_sites['CLS'] - current_metrics.get('CLS', 0))**2
    )
    
    # Get n closest
    closest_idx = distances.nsmallest(n).index
    
    return pareto_sites.loc[closest_idx]

# Example usage:
# recommendation = recommend_strategy('ecommerce')
# print(f"Strategy: {recommendation['strategy']}")
# print(f"Target LCP: {recommendation['target_benchmarks']['LCP']:.0f}ms")
#
# similar_sites = find_similar_pareto_sites({'LCP': 3000, 'FID': 200, 'CLS': 0.15})
# print(f"Found {len(similar_sites)} similar Pareto-optimal sites")
'''

with open(output_path / "optimization_guide.py", 'w') as f:
    f.write(guide_code)

print("Saved: optimization_guide.py")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

summary = f"""
MULTI-METRIC OPTIMIZATION SUMMARY
{'='*80}

PARETO ANALYSIS:
- Total samples analyzed: {len(metrics_df)}
- Pareto-efficient points: {len(pareto_points)} ({len(pareto_points)/len(metrics_df)*100:.1f}%)
- Metrics optimized: {', '.join(available_metrics.keys())}

AVERAGE IMPROVEMENT POTENTIAL:
{comparison[['Metric', 'Improvement']].to_string(index=False)}

OPTIMIZATION STRATEGIES:
{chr(10).join([f'- {name}: {details["description"]} ({details["use_case"]})' for name, details in optimization_strategies.items()])}

BEST ACHIEVABLE METRICS (Pareto Frontier):
{pd.DataFrame(best_examples).to_string()}

KEY INSIGHTS:
1. Moving to Pareto frontier offers {comparison['Improvement'].mean():.1f}% average improvement
2. {comparison.loc[comparison['Improvement'].idxmax(), 'Metric']} has highest improvement potential ({comparison['Improvement'].max():.1f}%)
3. Different strategies achieve different metric balances

OUTPUT LOCATIONS:
- Strategies: {output_path}
- Results: {results_path}
- Visualizations: {viz_path}

USAGE:
Use optimization_guide.py to:
1. Get strategy recommendations based on site type
2. Find Pareto-optimal sites similar to yours
3. Get target benchmarks for optimization

The Pareto frontier represents websites that achieve optimal balance - 
no metric can be improved without degrading another.
"""

print("\n" + summary)

with open(results_path / "multi_metric_optimization_summary.txt", 'w') as f:
    f.write(summary)

print("\n" + "="*80)
print("MULTI-METRIC OPTIMIZATION COMPLETE")
print("="*80)
print(f"\nResults: {results_path}")
print(f"Visualizations: {viz_path}")
print(f"Strategies: {output_path}")
print("="*80)
