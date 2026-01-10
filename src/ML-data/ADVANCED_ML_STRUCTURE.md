# Advanced ML Features - File Structure

## Directory Organization

```
src/ML-data/
│
├── 8_Advanced_Models/          # NEW: Advanced ML implementations
│   ├── ensemble/               # Ensemble model combinations
│   │   ├── voting_classifier/
│   │   ├── stacking_classifier/
│   │   └── models/
│   │
│   ├── regression/             # Regression models for exact predictions
│   │   ├── lcp_predictor/
│   │   ├── fid_predictor/
│   │   ├── cls_predictor/
│   │   └── models/
│   │
│   ├── explainable_ai/         # SHAP & LIME implementations
│   │   ├── shap_analysis/
│   │   ├── lime_analysis/
│   │   └── explanations/
│   │
│   ├── recommendation/         # ML-based recommendation system
│   │   ├── models/
│   │   ├── training_data/
│   │   └── rules_engine/
│   │
│   └── multi_metric_optimizer/ # Multi-metric optimization
│       ├── models/
│       ├── pareto_analysis/
│       └── optimization_strategies/
│
├── 9_Advanced_Results/         # NEW: Results from advanced models
│   ├── ensemble_performance/
│   ├── regression_predictions/
│   ├── shap_visualizations/
│   ├── recommendation_accuracy/
│   └── optimization_reports/
│
└── 10_Advanced_Visualizations/ # NEW: Advanced visualizations
    ├── ensemble_comparison/
    ├── shap_plots/
    ├── regression_plots/
    ├── recommendation_heatmaps/
    └── pareto_frontiers/
```

## Feature Implementation Order

1. **Ensemble Models** - Combine existing models for better accuracy
2. **Explainable AI (SHAP)** - Understand model decisions
3. **Regression Models** - Predict exact metric values
4. **Recommendation System** - Generate personalized optimization suggestions
5. **Multi-Metric Optimizer** - Balance competing performance goals

## File Naming Conventions

- Scripts: `train_[feature]_model.py`
- Models: `[feature]_[algorithm]_model.pkl`
- Results: `[feature]_results_[date].csv`
- Visualizations: `[feature]_[type]_plot.png`
