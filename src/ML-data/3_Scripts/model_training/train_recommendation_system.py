#!/usr/bin/env python3
"""
INTELLIGENT RECOMMENDATION SYSTEM
ML-based system that learns from successful optimization patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("INTELLIGENT RECOMMENDATION SYSTEM")
print("="*80)

# Paths
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
data_path = base_path / "1_Raw_Data" / "All thesis data - labeled.csv"
output_path = base_path / "8_Advanced_Models" / "recommendation" / "models"
results_path = base_path / "9_Advanced_Results" / "recommendation_accuracy"
viz_path = base_path / "10_Advanced_Visualizations" / "recommendation_heatmaps"

# Create directories
for path in [output_path, results_path, viz_path]:
    path.mkdir(parents=True, exist_ok=True)

print("\nLoading data...")

# Load data
data = pd.read_csv(data_path)

# Get numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# ============================================================================
# DEFINE RECOMMENDATION RULES BASED ON PERFORMANCE PATTERNS
# ============================================================================

print("\nDefining recommendation rules...")

# Recommendation categories
recommendation_rules = {
    'LCP_OPTIMIZATION': {
        'triggers': {
            'Largest_contentful_paint_LCP_ms': {'threshold': 2500, 'operator': '>'},
            'Page_size_MB': {'threshold': 2.0, 'operator': '>'}
        },
        'recommendations': [
            'Optimize images: Compress and use modern formats (WebP, AVIF)',
            'Implement lazy loading for images and iframes',
            'Reduce server response time (TTFB)',
            'Remove render-blocking resources',
            'Preload critical resources'
        ],
        'priority': 'HIGH'
    },
    
    'FID_OPTIMIZATION': {
        'triggers': {
            'Interaction_to_Next_Paint_INP_ms': {'threshold': 200, 'operator': '>'},
            'Time_to_interactive_TTI_ms': {'threshold': 5000, 'operator': '>'}
        },
        'recommendations': [
            'Reduce JavaScript execution time',
            'Break up long-running tasks',
            'Use web workers for heavy computations',
            'Implement code splitting',
            'Defer non-critical JavaScript'
        ],
        'priority': 'HIGH'
    },
    
    'CLS_OPTIMIZATION': {
        'triggers': {
            'Cumulative_Layout_Shift_CLS': {'threshold': 0.1, 'operator': '>'}
        },
        'recommendations': [
            'Set explicit dimensions for images and embeds',
            'Reserve space for ad slots',
            'Avoid inserting content above existing content',
            'Use transform animations instead of layout properties',
            'Preload fonts to avoid FOIT/FOUT'
        ],
        'priority': 'MEDIUM'
    },
    
    'RESOURCE_OPTIMIZATION': {
        'triggers': {
            'No_of_requests': {'threshold': 100, 'operator': '>'},
            'Page_size_MB': {'threshold': 3.0, 'operator': '>'}
        },
        'recommendations': [
            'Minimize HTTP requests through bundling',
            'Implement HTTP/2 or HTTP/3',
            'Use CDN for static assets',
            'Enable compression (Gzip/Brotli)',
            'Remove unused CSS and JavaScript'
        ],
        'priority': 'MEDIUM'
    },
    
    'CACHING_OPTIMIZATION': {
        'triggers': {
            'Response_time_ms': {'threshold': 600, 'operator': '>'}
        },
        'recommendations': [
            'Implement browser caching headers',
            'Use service workers for offline caching',
            'Implement CDN caching',
            'Optimize database queries',
            'Use Redis/Memcached for server-side caching'
        ],
        'priority': 'LOW'
    }
}

# Save recommendation rules
with open(output_path / "recommendation_rules.json", 'w') as f:
    json.dump(recommendation_rules, f, indent=2)

print(f"Saved {len(recommendation_rules)} recommendation categories")

# ============================================================================
# CREATE RECOMMENDATION FUNCTION
# ============================================================================

def generate_recommendations(metrics_dict):
    """
    Generate personalized recommendations based on metrics
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metric names and values
        
    Returns:
    --------
    dict : Recommendations organized by priority
    """
    recommendations = {
        'HIGH': [],
        'MEDIUM': [],
        'LOW': []
    }
    
    for category, rule in recommendation_rules.items():
        # Check if triggers are met
        triggered = False
        for metric, condition in rule['triggers'].items():
            if metric in metrics_dict:
                value = metrics_dict[metric]
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '>' and value > threshold:
                    triggered = True
                elif operator == '<' and value < threshold:
                    triggered = True
                elif operator == '==' and value == threshold:
                    triggered = True
        
        if triggered:
            priority = rule['priority']
            for rec in rule['recommendations']:
                if rec not in recommendations[priority]:
                    recommendations[priority].append(rec)
    
    return recommendations

# ============================================================================
# ML-BASED RECOMMENDATION SCORING
# ============================================================================

print("\nTraining ML-based recommendation prioritizer...")

# Create training data by simulating improvements
# For each sample, create features and label based on potential improvement

training_data = []

for idx, row in data.iterrows():
    if 'label' not in row or pd.isna(row.get('label')):
        continue
    
    metrics = {col: row[col] for col in numeric_cols if col in row.index}
    label = row.get('label', row.get('Label', 'Unknown'))
    
    # Generate features for ML model
    features = {
        'current_lcp': metrics.get('Largest_contentful_paint_LCP_ms', 0),
        'current_fid': metrics.get('Interaction_to_Next_Paint_INP_ms', 0),
        'current_cls': metrics.get('Cumulative_Layout_Shift_CLS', 0),
        'page_size': metrics.get('Page_size_MB', 0),
        'num_requests': metrics.get('No_of_requests', 0),
        'response_time': metrics.get('Response_time_ms', 0),
        'performance_label': 0 if label == 'Good' else (1 if label == 'Average' else 2)
    }
    
    training_data.append(features)

training_df = pd.DataFrame(training_data)

# Remove rows with missing values
training_df = training_df.dropna()

print(f"Training samples: {len(training_df)}")

# Prepare for ML model
X_rec = training_df.drop(['performance_label'], axis=1)
y_rec = training_df['performance_label']

# Train recommendation scorer
X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(
    X_rec, y_rec, test_size=0.2, random_state=42
)

scaler_rec = StandardScaler()
X_train_rec_scaled = scaler_rec.fit_transform(X_train_rec)
X_test_rec_scaled = scaler_rec.transform(X_test_rec)

# Train RandomForest to predict improvement potential
rec_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rec_model.fit(X_train_rec_scaled, y_train_rec)

# Save model
joblib.dump(rec_model, output_path / "recommendation_scorer.pkl")
joblib.dump(scaler_rec, output_path / "recommendation_scaler.pkl")

print("Recommendation scorer trained and saved")

# Feature importance for recommendations
feature_importance = pd.DataFrame({
    'Feature': X_rec.columns,
    'Importance': rec_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance for Recommendations:")
print(feature_importance.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nCreating visualizations...")

# 1. Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'])
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Feature Importance for Recommendation Prioritization', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(viz_path / "recommendation_feature_importance.png", dpi=300, bbox_inches='tight')
print("Saved: recommendation_feature_importance.png")
plt.close()

# 2. Recommendation Categories Heatmap
category_matrix = []
category_names = []

for category, rule in recommendation_rules.items():
    category_names.append(category.replace('_', ' '))
    
    # Count how many metrics trigger this category
    trigger_count = len(rule['triggers'])
    rec_count = len(rule['recommendations'])
    priority_score = 3 if rule['priority'] == 'HIGH' else (2 if rule['priority'] == 'MEDIUM' else 1)
    
    category_matrix.append([trigger_count, rec_count, priority_score])

category_df = pd.DataFrame(
    category_matrix,
    columns=['Trigger Metrics', 'Recommendation Count', 'Priority Score'],
    index=category_names
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(category_df.T, annot=True, fmt='.0f', cmap='YlOrRd', 
            cbar_kws={'label': 'Score'}, ax=ax)
ax.set_title('Recommendation Categories Overview', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_path / "recommendation_categories_heatmap.png", dpi=300, bbox_inches='tight')
print("Saved: recommendation_categories_heatmap.png")
plt.close()

# ============================================================================
# CREATE RECOMMENDATION GENERATOR SCRIPT
# ============================================================================

print("\nCreating recommendation generator script...")

generator_code = '''
"""
Recommendation Generator
Use this to generate personalized recommendations for any website
"""

import joblib
import json
import numpy as np
from pathlib import Path

# Load models and rules
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
model_path = base_path / "8_Advanced_Models" / "recommendation" / "models"

rec_model = joblib.load(model_path / "recommendation_scorer.pkl")
scaler = joblib.load(model_path / "recommendation_scaler.pkl")

with open(model_path / "recommendation_rules.json", 'r') as f:
    rules = json.load(f)

def get_recommendations(metrics):
    """
    Generate personalized recommendations
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with keys: 'lcp', 'fid', 'cls', 'page_size', 
        'num_requests', 'response_time'
        
    Returns:
    --------
    dict : Prioritized recommendations
    """
    
    # Prepare features for ML model
    features = np.array([[
        metrics.get('lcp', 0),
        metrics.get('fid', 0),
        metrics.get('cls', 0),
        metrics.get('page_size', 0),
        metrics.get('num_requests', 0),
        metrics.get('response_time', 0)
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict improvement potential
    prediction = rec_model.predict(features_scaled)[0]
    probability = rec_model.predict_proba(features_scaled)[0]
    
    # Map prediction to category
    categories = ['Good - Minor tweaks', 'Average - Moderate changes', 'Weak - Major overhaul']
    improvement_potential = categories[prediction]
    confidence = max(probability) * 100
    
    # Generate rule-based recommendations
    recommendations = {
        'HIGH': [],
        'MEDIUM': [],
        'LOW': []
    }
    
    # Full metrics dict for rule matching
    full_metrics = {
        'Largest_contentful_paint_LCP_ms': metrics.get('lcp', 0),
        'Interaction_to_Next_Paint_INP_ms': metrics.get('fid', 0),
        'Cumulative_Layout_Shift_CLS': metrics.get('cls', 0),
        'Page_size_MB': metrics.get('page_size', 0),
        'No_of_requests': metrics.get('num_requests', 0),
        'Response_time_ms': metrics.get('response_time', 0)
    }
    
    # Check each rule
    for category, rule in rules.items():
        triggered = False
        for metric, condition in rule['triggers'].items():
            if metric in full_metrics:
                value = full_metrics[metric]
                threshold = condition['threshold']
                operator = condition['operator']
                
                if operator == '>' and value > threshold:
                    triggered = True
                    break
        
        if triggered:
            priority = rule['priority']
            recommendations[priority].extend(rule['recommendations'])
    
    return {
        'improvement_potential': improvement_potential,
        'confidence': f'{confidence:.1f}%',
        'recommendations': recommendations,
        'total_recommendations': sum(len(v) for v in recommendations.values())
    }

# Example usage:
# recommendations = get_recommendations({
#     'lcp': 3500,
#     'fid': 250,
#     'cls': 0.15,
#     'page_size': 4.5,
#     'num_requests': 120,
#     'response_time': 800
# })
# print(recommendations)
'''

with open(output_path / "generate_recommendations.py", 'w') as f:
    f.write(generator_code)

print("Saved: generate_recommendations.py")

# ============================================================================
# TEST THE RECOMMENDATION SYSTEM
# ============================================================================

print("\nTesting recommendation system with sample data...")

# Test with a few samples
test_samples = [
    {
        'name': 'Poor Performance Site',
        'lcp': 4500,
        'fid': 350,
        'cls': 0.25,
        'page_size': 5.5,
        'num_requests': 150,
        'response_time': 900
    },
    {
        'name': 'Average Performance Site',
        'lcp': 2800,
        'fid': 180,
        'cls': 0.12,
        'page_size': 2.8,
        'num_requests': 80,
        'response_time': 500
    },
    {
        'name': 'Good Performance Site',
        'lcp': 1800,
        'fid': 80,
        'cls': 0.05,
        'page_size': 1.2,
        'num_requests': 40,
        'response_time': 300
    }
]

print("\nSample Recommendations:")
print("="*80)

for sample in test_samples:
    name = sample.pop('name')
    recs = generate_recommendations({
        'Largest_contentful_paint_LCP_ms': sample['lcp'],
        'Interaction_to_Next_Paint_INP_ms': sample['fid'],
        'Cumulative_Layout_Shift_CLS': sample['cls'],
        'Page_size_MB': sample['page_size'],
        'No_of_requests': sample['num_requests'],
        'Response_time_ms': sample['response_time']
    })
    
    print(f"\n{name}:")
    print(f"  Metrics: LCP={sample['lcp']}ms, FID={sample['fid']}ms, CLS={sample['cls']}")
    
    total_recs = sum(len(v) for v in recs.values())
    print(f"  Total Recommendations: {total_recs}")
    
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        if recs[priority]:
            print(f"\n  {priority} PRIORITY ({len(recs[priority])} items):")
            for i, rec in enumerate(recs[priority][:3], 1):  # Show first 3
                print(f"    {i}. {rec}")
            if len(recs[priority]) > 3:
                print(f"    ... and {len(recs[priority])-3} more")

print("\n" + "="*80)

# ============================================================================
# SAVE SUMMARY
# ============================================================================

summary = f"""
INTELLIGENT RECOMMENDATION SYSTEM SUMMARY
{'='*80}

COMPONENTS:
1. Rule-based recommendation engine ({len(recommendation_rules)} categories)
2. ML-based recommendation scorer (Random Forest)
3. Automated recommendation generator

RECOMMENDATION CATEGORIES:
{chr(10).join([f'- {name.replace("_", " ")}: {len(rule["recommendations"])} recommendations ({rule["priority"]} priority)' for name, rule in recommendation_rules.items()])}

TOTAL UNIQUE RECOMMENDATIONS: {sum(len(rule['recommendations']) for rule in recommendation_rules.values())}

ML MODEL PERFORMANCE:
- Training Samples: {len(training_df)}
- Feature Importance Top 3:
{feature_importance.head(3).to_string(index=False)}

OUTPUT LOCATIONS:
- Models: {output_path}
- Results: {results_path}
- Visualizations: {viz_path}

USAGE:
Use generate_recommendations.py to generate personalized recommendations
for any website based on its performance metrics.

The system combines:
- Rule-based triggers (expert knowledge)
- ML-based prioritization (learned patterns)
- Personalized recommendations (context-aware)
"""

print(summary)

with open(results_path / "recommendation_system_summary.txt", 'w') as f:
    f.write(summary)

print("\n" + "="*80)
print("RECOMMENDATION SYSTEM COMPLETE")
print("="*80)
print(f"\nModels: {output_path}")
print(f"Visualizations: {viz_path}")
print(f"Results: {results_path}")
print("="*80)
