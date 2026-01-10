
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
