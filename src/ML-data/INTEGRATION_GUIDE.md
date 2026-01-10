# Advanced ML Features - Integration Guide
**How to Use All 5 Features Together**

## Table of Contents
1. [Quick Start](#quick-start)
2. [Feature-by-Feature Usage](#feature-by-feature-usage)
3. [Complete Integration Example](#complete-integration-example)
4. [API Integration](#api-integration)
5. [Frontend Integration](#frontend-integration)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites
```bash
# Ensure Python environment is activated
cd f:/client/Optimizer/optimizer
.venv\Scripts\activate  # Windows

# Verify installations
python src/ML-data/fix_tensorflow_imports.py
```

### Start Advanced ML API
```bash
cd src/api
python ml_server_advanced.py
```

**API will be available at:**
- Endpoint: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Feature-by-Feature Usage

### 1. Ensemble Classification (100% Accuracy)

**Purpose:** Classify website performance as Good/Average/Weak with perfect accuracy

**Script:**
```python
import joblib
import numpy as np

# Load models
voting_model = joblib.load('src/ML-data/8_Advanced_Models/ensemble/models/voting_classifier.pkl')
scaler = joblib.load('src/ML-data/8_Advanced_Models/ensemble/models/ensemble_scaler.pkl')
label_encoder = joblib.load('src/ML-data/8_Advanced_Models/ensemble/models/label_encoder.pkl')

# Prepare features (22 features)
features = np.array([[
    composite_score, response_time, dom_load_time, ttfb, total_links,
    load_time, num_requests, byte_size, lcp, page_size, fcp, tti,
    speed_index, cls, fid, tbt, 0, 0, 0, 0, 0, 0  # 22 features total
]])

# Scale features
features_scaled = scaler.transform(features)

# Predict
prediction_encoded = voting_model.predict(features_scaled)[0]
prediction = label_encoder.inverse_transform([prediction_encoded])[0]
confidence = voting_model.predict_proba(features_scaled)[0].max()

print(f"Classification: {prediction}")
print(f"Confidence: {confidence:.2%}")
```

**Expected Output:**
```
Classification: Good
Confidence: 100.00%
```

**When to Use:**
- Need simple Good/Average/Weak classification
- Require high confidence predictions
- Building dashboard status indicators

---

### 2. SHAP Explanations

**Purpose:** Understand WHY the model made a specific prediction

**Script:**
```python
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load explainer
explainer = joblib.load('src/ML-data/8_Advanced_Models/explainable_ai/shap_analysis/shap_explainer.pkl')

# Feature names
feature_names = [
    'composite_score', 'response_time', 'dom_load_time', 'ttfb', 
    'total_links', 'load_time', 'num_requests', 'byte_size',
    'lcp', 'page_size', 'fcp', 'tti', 'speed_index', 'cls', 
    'fid', 'tbt', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21'
]

# Create DataFrame
features_df = pd.DataFrame(features_scaled, columns=feature_names)

# Compute SHAP values
shap_values = explainer(features_df)

# Visualize - Waterfall plot (for single prediction)
shap.plots.waterfall(shap_values[0])
plt.savefig('shap_explanation.png', dpi=300, bbox_inches='tight')
plt.show()

# Get top influencing features
feature_impacts = []
for i, (fname, shap_val) in enumerate(zip(feature_names, shap_values.values[0])):
    feature_impacts.append({
        'feature': fname,
        'shap_value': shap_val,
        'impact': abs(shap_val)
    })

# Sort by impact
feature_impacts_sorted = sorted(feature_impacts, key=lambda x: x['impact'], reverse=True)

print("\nTop 5 Most Impactful Features:")
for i, fi in enumerate(feature_impacts_sorted[:5], 1):
    direction = "↑ increases" if fi['shap_value'] > 0 else "↓ decreases"
    print(f"{i}. {fi['feature']}: {direction} prediction by {fi['impact']:.4f}")
```

**Expected Output:**
```
Top 5 Most Impactful Features:
1. composite_score: ↑ increases prediction by 0.4523
2. response_time: ↓ decreases prediction by 0.0012
3. dom_load_time: ↓ decreases prediction by 0.0008
...
```

**When to Use:**
- Need to explain predictions to stakeholders
- Debugging unexpected model behavior
- Understanding feature importance for specific cases
- Building trust in ML predictions

---

### 3. Regression Predictions (Exact Values)

**Purpose:** Predict exact LCP (ms), FID (ms), CLS values instead of categories

**Script:**
```python
import joblib
import numpy as np

# Load best models for each metric
lcp_model = joblib.load('src/ML-data/8_Advanced_Models/regression/models/LCP_gradient_boosting.pkl')
fid_model = joblib.load('src/ML-data/8_Advanced_Models/regression/models/FID_INP_random_forest.pkl')
cls_model = joblib.load('src/ML-data/8_Advanced_Models/regression/models/CLS_random_forest.pkl')

# Load scalers
lcp_scaler = joblib.load('src/ML-data/8_Advanced_Models/regression/models/LCP_scaler.pkl')
fid_scaler = joblib.load('src/ML-data/8_Advanced_Models/regression/models/FID_INP_scaler.pkl')
cls_scaler = joblib.load('src/ML-data/8_Advanced_Models/regression/models/CLS_scaler.pkl')

# Prepare features (first 21 features only for regression)
features = np.array([[
    composite_score, response_time, dom_load_time, ttfb, total_links,
    load_time, num_requests, byte_size, lcp, page_size, fcp, tti,
    speed_index, cls, fid, tbt, 0, 0, 0, 0, 0
]])

# Predict LCP
lcp_scaled = lcp_scaler.transform(features)
predicted_lcp = lcp_model.predict(lcp_scaled)[0]

# Predict FID
fid_scaled = fid_scaler.transform(features)
predicted_fid = fid_model.predict(fid_scaled)[0]

# Predict CLS
cls_scaled = cls_scaler.transform(features)
predicted_cls = cls_model.predict(cls_scaled)[0]

print(f"Predicted LCP: {predicted_lcp:.2f} ms")
print(f"Predicted FID: {predicted_fid:.2f} ms")
print(f"Predicted CLS: {predicted_cls:.4f}")

# Check against thresholds
print("\nCore Web Vitals Assessment:")
print(f"LCP: {'✓ Good' if predicted_lcp <= 2500 else '✗ Needs Improvement' if predicted_lcp <= 4000 else '✗ Poor'}")
print(f"FID: {'✓ Good' if predicted_fid <= 100 else '✗ Needs Improvement' if predicted_fid <= 300 else '✗ Poor'}")
print(f"CLS: {'✓ Good' if predicted_cls <= 0.1 else '✗ Needs Improvement' if predicted_cls <= 0.25 else '✗ Poor'}")
```

**Expected Output:**
```
Predicted LCP: 2345.67 ms
Predicted FID: 87.32 ms
Predicted CLS: 0.0834

Core Web Vitals Assessment:
LCP: ✓ Good
FID: ✓ Good
CLS: ✓ Good
```

**When to Use:**
- Need precise metric values for optimization planning
- Setting specific performance goals
- Tracking incremental improvements
- A/B testing optimization strategies

---

### 4. Intelligent Recommendations

**Purpose:** Get personalized optimization suggestions based on ML analysis

**Script:**
```python
import json
import joblib

# Load recommendation system
with open('src/ML-data/8_Advanced_Models/recommendation/models/recommendation_rules.json', 'r') as f:
    recommendation_rules = json.load(f)

rec_model = joblib.load('src/ML-data/8_Advanced_Models/recommendation/models/recommendation_scorer.pkl')
rec_scaler = joblib.load('src/ML-data/8_Advanced_Models/recommendation/models/recommendation_scaler.pkl')

# Current metrics
current_metrics = {
    'Largest_contentful_paint_LCP_ms': 3500,
    'Interaction_to_Next_Paint_INP_ms': 200,
    'Cumulative_Layout_Shift_CLS': 0.15,
    'Page_size_MB': 3.5,
    'No_of_requests': 120,
    'Response_time_ms': 500
}

# Generate recommendations
recommendations = {
    'HIGH': [],
    'MEDIUM': [],
    'LOW': []
}

for category, rule in recommendation_rules.items():
    triggered = False
    for metric, condition in rule['triggers'].items():
        if metric in current_metrics:
            value = current_metrics[metric]
            threshold = condition['threshold']
            operator = condition['operator']
            
            if operator == '>' and value > threshold:
                triggered = True
                break
    
    if triggered:
        priority = rule['priority']
        recommendations[priority].extend(rule['recommendations'])

# Remove duplicates
for priority in recommendations:
    recommendations[priority] = list(set(recommendations[priority]))

# Display recommendations
total_recs = sum(len(v) for v in recommendations.values())
print(f"Generated {total_recs} recommendations:\n")

for priority in ['HIGH', 'MEDIUM', 'LOW']:
    if recommendations[priority]:
        print(f"{priority} Priority ({len(recommendations[priority])} items):")
        for i, rec in enumerate(recommendations[priority], 1):
            print(f"  {i}. {rec}")
        print()
```

**Expected Output:**
```
Generated 12 recommendations:

HIGH Priority (5 items):
  1. Implement server-side rendering for faster initial page load
  2. Optimize and compress images (use WebP format)
  3. Implement lazy loading for images and videos
  4. Reduce main thread work by code splitting
  5. Reserve space for dynamic content to prevent layout shifts

MEDIUM Priority (7 items):
  1. Enable text compression (gzip or brotli)
  2. Implement browser caching strategies
  3. Use a Content Delivery Network (CDN)
  ...
```

**When to Use:**
- Providing actionable optimization steps
- Prioritizing optimization efforts
- Creating automated optimization reports
- Guiding developers on improvements

---

### 5. Multi-Metric Optimization Strategies

**Purpose:** Find optimal balance between LCP, FID, CLS using Pareto analysis

**Script:**
```python
import json

# Load optimization strategies
with open('src/ML-data/8_Advanced_Models/multi_metric_optimizer/models/optimization_strategies.json', 'r') as f:
    strategies = json.load(f)

# Select strategy based on site type
def recommend_strategy(site_type='general'):
    """
    site_type options:
    - 'content', 'blog', 'news' → LCP_FOCUSED
    - 'app', 'spa', 'interactive' → INTERACTIVITY_FOCUSED
    - 'ecommerce', 'shop', 'checkout' → STABILITY_FOCUSED
    - 'general' → BALANCED
    """
    strategy_map = {
        'content': 'LCP_FOCUSED',
        'blog': 'LCP_FOCUSED',
        'news': 'LCP_FOCUSED',
        'app': 'INTERACTIVITY_FOCUSED',
        'spa': 'INTERACTIVITY_FOCUSED',
        'interactive': 'INTERACTIVITY_FOCUSED',
        'ecommerce': 'STABILITY_FOCUSED',
        'shop': 'STABILITY_FOCUSED',
        'checkout': 'STABILITY_FOCUSED',
        'general': 'BALANCED'
    }
    
    strategy_key = strategy_map.get(site_type.lower(), 'BALANCED')
    return strategies[strategy_key]

# Example usage
site_type = 'ecommerce'
strategy = recommend_strategy(site_type)

print(f"Recommended Strategy for '{site_type}' site:\n")
print(f"Strategy: {strategy['name']}")
print(f"Description: {strategy['description']}")
print(f"\nOptimization Weights:")
print(f"  LCP: {strategy['weights']['LCP']*100:.0f}%")
print(f"  FID: {strategy['weights']['FID']*100:.0f}%")
print(f"  CLS: {strategy['weights']['CLS']*100:.0f}%")
print(f"\nTarget Metrics:")
print(f"  LCP ≤ {strategy['targets']['LCP']} ms")
print(f"  FID ≤ {strategy['targets']['FID']} ms")
print(f"  CLS ≤ {strategy['targets']['CLS']}")
print(f"\nExpected Improvements:")
for metric, improvement in strategy['expected_improvements'].items():
    print(f"  {metric}: {improvement}")
```

**Expected Output:**
```
Recommended Strategy for 'ecommerce' site:

Strategy: Stability-Focused Optimization
Description: Prioritizes visual stability (CLS) while maintaining acceptable loading speed

Optimization Weights:
  LCP: 20%
  FID: 20%
  CLS: 60%

Target Metrics:
  LCP ≤ 3000 ms
  FID ≤ 150 ms
  CLS ≤ 0.05

Expected Improvements:
  LCP: Moderate improvement (20-40%)
  FID: Moderate improvement (20-40%)
  CLS: Significant improvement (60-80%)
```

**When to Use:**
- Planning comprehensive optimization strategy
- Balancing tradeoffs between metrics
- Setting realistic performance targets
- Aligning optimization with business goals

---

## Complete Integration Example

**Real-World Scenario:** Analyze a website and get complete optimization plan

```python
"""
Complete Workflow: From Analysis to Action Plan
"""
import joblib
import numpy as np
import pandas as pd
import json

# ============================================================================
# INPUT: Website Performance Data
# ============================================================================

website_data = {
    'url': 'https://example.com',
    'site_type': 'ecommerce',
    
    # Current metrics
    'composite_score': 45.2,
    'response_time': 450,
    'dom_load_time': 1200,
    'ttfb': 350,
    'total_links': 85,
    'load_time': 3500,
    'num_requests': 120,
    'byte_size': 2500000,
    'lcp': 3200,
    'page_size': 3.5,
    'fcp': 1800,
    'tti': 4500,
    'speed_index': 3800,
    'cls': 0.18,
    'fid': 180,
    'tbt': 450
}

print("="*80)
print(f"COMPLETE PERFORMANCE ANALYSIS: {website_data['url']}")
print("="*80)

# ============================================================================
# STEP 1: Ensemble Classification
# ============================================================================

print("\n1. CLASSIFICATION (Ensemble Model)")
print("-" * 80)

voting_model = joblib.load('src/ML-data/8_Advanced_Models/ensemble/models/voting_classifier.pkl')
scaler = joblib.load('src/ML-data/8_Advanced_Models/ensemble/models/ensemble_scaler.pkl')
label_encoder = joblib.load('src/ML-data/8_Advanced_Models/ensemble/models/label_encoder.pkl')

features = np.array([[
    website_data['composite_score'], website_data['response_time'],
    website_data['dom_load_time'], website_data['ttfb'],
    website_data['total_links'], website_data['load_time'],
    website_data['num_requests'], website_data['byte_size'],
    website_data['lcp'], website_data['page_size'],
    website_data['fcp'], website_data['tti'],
    website_data['speed_index'], website_data['cls'],
    website_data['fid'], website_data['tbt'],
    0, 0, 0, 0, 0, 0
]])

features_scaled = scaler.transform(features)
prediction_encoded = voting_model.predict(features_scaled)[0]
prediction = label_encoder.inverse_transform([prediction_encoded])[0]
confidence = voting_model.predict_proba(features_scaled)[0].max()

print(f"Performance Category: {prediction}")
print(f"Confidence: {confidence:.2%}")

# ============================================================================
# STEP 2: Regression Predictions
# ============================================================================

print("\n2. EXACT METRIC PREDICTIONS (Regression Models)")
print("-" * 80)

lcp_model = joblib.load('src/ML-data/8_Advanced_Models/regression/models/LCP_gradient_boosting.pkl')
fid_model = joblib.load('src/ML-data/8_Advanced_Models/regression/models/FID_INP_random_forest.pkl')
cls_model = joblib.load('src/ML-data/8_Advanced_Models/regression/models/CLS_random_forest.pkl')

lcp_scaler = joblib.load('src/ML-data/8_Advanced_Models/regression/models/LCP_scaler.pkl')
fid_scaler = joblib.load('src/ML-data/8_Advanced_Models/regression/models/FID_INP_scaler.pkl')
cls_scaler = joblib.load('src/ML-data/8_Advanced_Models/regression/models/CLS_scaler.pkl')

reg_features = features[:, :21]

predicted_lcp = lcp_model.predict(lcp_scaler.transform(reg_features))[0]
predicted_fid = fid_model.predict(fid_scaler.transform(reg_features))[0]
predicted_cls = cls_model.predict(cls_scaler.transform(reg_features))[0]

print(f"Current LCP: {website_data['lcp']} ms → Optimized Prediction: {predicted_lcp:.2f} ms")
print(f"Current FID: {website_data['fid']} ms → Optimized Prediction: {predicted_fid:.2f} ms")
print(f"Current CLS: {website_data['cls']} → Optimized Prediction: {predicted_cls:.4f}")

# ============================================================================
# STEP 3: Recommendations
# ============================================================================

print("\n3. OPTIMIZATION RECOMMENDATIONS")
print("-" * 80)

with open('src/ML-data/8_Advanced_Models/recommendation/models/recommendation_rules.json', 'r') as f:
    recommendation_rules = json.load(f)

current_metrics = {
    'Largest_contentful_paint_LCP_ms': website_data['lcp'],
    'Interaction_to_Next_Paint_INP_ms': website_data['fid'],
    'Cumulative_Layout_Shift_CLS': website_data['cls'],
    'Page_size_MB': website_data['page_size'],
    'No_of_requests': website_data['num_requests'],
    'Response_time_ms': website_data['response_time']
}

recommendations = {'HIGH': [], 'MEDIUM': [], 'LOW': []}

for category, rule in recommendation_rules.items():
    triggered = False
    for metric, condition in rule['triggers'].items():
        if metric in current_metrics:
            value = current_metrics[metric]
            threshold = condition['threshold']
            operator = condition['operator']
            if operator == '>' and value > threshold:
                triggered = True
                break
    if triggered:
        priority = rule['priority']
        recommendations[priority].extend(rule['recommendations'])

for priority in recommendations:
    recommendations[priority] = list(set(recommendations[priority]))

total_recs = sum(len(v) for v in recommendations.values())
print(f"Total Recommendations: {total_recs}\n")

for priority in ['HIGH', 'MEDIUM', 'LOW']:
    if recommendations[priority]:
        print(f"{priority} Priority ({len(recommendations[priority])} items):")
        for i, rec in enumerate(recommendations[priority][:3], 1):  # Show top 3
            print(f"  {i}. {rec}")
        if len(recommendations[priority]) > 3:
            print(f"  ... and {len(recommendations[priority]) - 3} more")
        print()

# ============================================================================
# STEP 4: Optimization Strategy
# ============================================================================

print("\n4. RECOMMENDED OPTIMIZATION STRATEGY")
print("-" * 80)

with open('src/ML-data/8_Advanced_Models/multi_metric_optimizer/models/optimization_strategies.json', 'r') as f:
    strategies = json.load(f)

strategy_map = {
    'content': 'LCP_FOCUSED',
    'blog': 'LCP_FOCUSED',
    'news': 'LCP_FOCUSED',
    'app': 'INTERACTIVITY_FOCUSED',
    'spa': 'INTERACTIVITY_FOCUSED',
    'ecommerce': 'STABILITY_FOCUSED',
    'shop': 'STABILITY_FOCUSED',
    'general': 'BALANCED'
}

strategy_key = strategy_map.get(website_data['site_type'].lower(), 'BALANCED')
strategy = strategies[strategy_key]

print(f"Strategy: {strategy['name']}")
print(f"Description: {strategy['description']}")
print(f"\nTarget Metrics:")
print(f"  LCP ≤ {strategy['targets']['LCP']} ms")
print(f"  FID ≤ {strategy['targets']['FID']} ms")
print(f"  CLS ≤ {strategy['targets']['CLS']}")

# ============================================================================
# STEP 5: SHAP Explanation
# ============================================================================

print("\n5. EXPLANATION (Why this classification?)")
print("-" * 80)

explainer = joblib.load('src/ML-data/8_Advanced_Models/explainable_ai/shap_analysis/shap_explainer.pkl')

feature_names = [
    'composite_score', 'response_time', 'dom_load_time', 'ttfb', 
    'total_links', 'load_time', 'num_requests', 'byte_size',
    'lcp', 'page_size', 'fcp', 'tti', 'speed_index', 'cls', 
    'fid', 'tbt', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21'
]

features_df = pd.DataFrame(features_scaled, columns=feature_names)
shap_values = explainer(features_df)

feature_impacts = []
for fname, shap_val in zip(feature_names, shap_values.values[0]):
    feature_impacts.append({
        'feature': fname,
        'shap_value': shap_val,
        'impact': abs(shap_val)
    })

feature_impacts_sorted = sorted(feature_impacts, key=lambda x: x['impact'], reverse=True)

print("Top 5 Most Impactful Features:")
for i, fi in enumerate(feature_impacts_sorted[:5], 1):
    direction = "pushes toward higher category" if fi['shap_value'] > 0 else "pushes toward lower category"
    print(f"  {i}. {fi['feature']}: {direction} (impact: {fi['impact']:.4f})")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY & ACTION PLAN")
print("="*80)

print(f"\n✓ Classification: {prediction} ({confidence:.0%} confidence)")
print(f"✓ Predicted Improvements: LCP {predicted_lcp:.0f}ms, FID {predicted_fid:.0f}ms, CLS {predicted_cls:.3f}")
print(f"✓ Recommendations: {total_recs} actionable items")
print(f"✓ Strategy: {strategy['name']}")
print(f"\nNext Steps:")
print(f"  1. Implement {len(recommendations['HIGH'])} HIGH priority recommendations")
print(f"  2. Target {strategy['targets']['CLS']} CLS (current: {website_data['cls']})")
print(f"  3. Monitor improvements using regression predictions")
print(f"  4. Re-analyze after optimizations to track progress")
```

---

## API Integration

### Using FastAPI Server

**Start Server:**
```bash
cd src/api
python ml_server_advanced.py
```

**Complete Prediction (JavaScript/TypeScript):**
```typescript
interface PerformanceMetrics {
  lcp: number;
  fid: number;
  cls: number;
  fcp?: number;
  tti?: number;
  // ... other metrics
}

async function analyzePerformance(metrics: PerformanceMetrics) {
  const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(metrics)
  });
  
  const result = await response.json();
  
  console.log('Ensemble Prediction:', result.ensemble_prediction);
  console.log('Confidence:', result.ensemble_confidence);
  console.log('Regression Predictions:', result.regression_predictions);
  console.log('Recommendations:', result.recommendations);
  console.log('Strategy:', result.optimization_strategy);
  console.log('SHAP Explanation:', result.shap_explanation);
  
  return result;
}

// Example usage
const metrics = {
  lcp: 3200,
  fid: 180,
  cls: 0.18,
  // ... other metrics
};

const analysis = await analyzePerformance(metrics);
```

**Recommendations Only:**
```javascript
async function getRecommendations(metrics) {
  const response = await fetch('http://localhost:8000/api/recommendations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(metrics)
  });
  
  const result = await response.json();
  return result.recommendations;
}
```

**Regression Predictions:**
```javascript
async function predictExactMetrics(metrics) {
  const response = await fetch('http://localhost:8000/api/regression', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(metrics)
  });
  
  const predictions = await response.json();
  console.log(`Predicted LCP: ${predictions.predicted_lcp_ms} ms`);
  console.log(`Predicted FID: ${predictions.predicted_fid_ms} ms`);
  console.log(`Predicted CLS: ${predictions.predicted_cls}`);
  
  return predictions;
}
```

---

## Frontend Integration

### React Component Example

```tsx
import React, { useState } from 'react';

interface AnalysisResult {
  ensemble_prediction: string;
  ensemble_confidence: number;
  regression_predictions: {
    predicted_lcp_ms: number;
    predicted_fid_ms: number;
    predicted_cls: number;
  };
  recommendations: {
    HIGH: string[];
    MEDIUM: string[];
    LOW: string[];
  };
  optimization_strategy: string;
}

export function PerformanceAnalyzer() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  
  const analyzePerformance = async (metrics: any) => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metrics)
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="performance-analyzer">
      {loading && <div>Analyzing...</div>}
      
      {result && (
        <div className="results">
          <div className="classification">
            <h2>Performance: {result.ensemble_prediction}</h2>
            <div className="confidence">
              Confidence: {(result.ensemble_confidence * 100).toFixed(0)}%
            </div>
          </div>
          
          <div className="predictions">
            <h3>Optimized Predictions</h3>
            <div>LCP: {result.regression_predictions.predicted_lcp_ms.toFixed(0)} ms</div>
            <div>FID: {result.regression_predictions.predicted_fid_ms.toFixed(0)} ms</div>
            <div>CLS: {result.regression_predictions.predicted_cls.toFixed(4)}</div>
          </div>
          
          <div className="recommendations">
            <h3>Recommendations</h3>
            {result.recommendations.HIGH.length > 0 && (
              <div className="high-priority">
                <h4>HIGH Priority</h4>
                <ul>
                  {result.recommendations.HIGH.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## Troubleshooting

### Common Issues

#### 1. TensorFlow/Keras Import Errors
**Problem:** VS Code shows "Import tensorflow.keras could not be resolved"

**Solution:**
```bash
# Run verification script
python src/ML-data/fix_tensorflow_imports.py

# If still showing errors in VS Code:
# 1. Ctrl+Shift+P → "Python: Select Interpreter" → Choose .venv
# 2. Ctrl+Shift+P → "Developer: Reload Window"
# 3. Optional: pip install tensorflow-stubs
```

#### 2. Model Loading Errors
**Problem:** `FileNotFoundError: model not found`

**Solution:**
```python
from pathlib import Path

# Use absolute paths
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")
model_path = base_path / "8_Advanced_Models/ensemble/models/voting_classifier.pkl"

model = joblib.load(model_path)
```

#### 3. Feature Shape Mismatch
**Problem:** `ValueError: X has 20 features but model expects 22`

**Solution:**
```python
# Ensure exactly 22 features for classification
features = np.array([[
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, 0, 0, 0, 0, 0, 0
]])  # Pad with zeros if needed

# Regression needs 21 features
reg_features = features[:, :21]
```

#### 4. API Connection Errors
**Problem:** `Connection refused` when calling API

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Restart server
cd src/api
python ml_server_advanced.py

# Check firewall/port availability
```

#### 5. SHAP Visualization Issues
**Problem:** SHAP plots not displaying

**Solution:**
```python
import matplotlib.pyplot as plt
import shap

# Ensure matplotlib backend is set
plt.switch_backend('Agg')  # For non-interactive

# Or force display
shap.plots.waterfall(shap_values[0])
plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.close()
```

---

## Performance Tips

### 1. Batch Predictions
```python
# Instead of looping
for sample in samples:
    prediction = model.predict([sample])  # Slow

# Batch predict
all_predictions = model.predict(samples)  # Fast
```

### 2. Cache Model Loading
```python
# Load models once at startup
class ModelCache:
    def __init__(self):
        self.voting_model = None
        self.scaler = None
        
    def load_models(self):
        if self.voting_model is None:
            self.voting_model = joblib.load('...')
            self.scaler = joblib.load('...')
    
    def predict(self, features):
        self.load_models()
        return self.voting_model.predict(self.scaler.transform(features))

cache = ModelCache()
```

### 3. Use Async API Calls
```javascript
// Parallel requests
const [classification, regression, recommendations] = await Promise.all([
  fetch('/api/predict'),
  fetch('/api/regression'),
  fetch('/api/recommendations')
]);
```

---

## Next Steps

1. **Test Integration:** Run the complete example above
2. **Deploy API:** Host ml_server_advanced.py on production server
3. **Build Dashboard:** Create frontend using React/Vue/Angular
4. **Monitor Performance:** Track API response times
5. **Collect Feedback:** Gather user feedback on recommendations

---

**For More Information:**
- Master Report: [MASTER_ADVANCED_ML_REPORT.md](MASTER_ADVANCED_ML_REPORT.md)
- API Documentation: http://localhost:8000/docs
- Model Details: See individual feature READMEs in 8_Advanced_Models/

**Support:**
- Check [TROUBLESHOOTING.md](../../TROUBLESHOOTING.md)
- Review model metrics in 9_Advanced_Results/
- Examine visualizations in 10_Advanced_Visualizations/
