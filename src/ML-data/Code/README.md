Setup and run instructions for ML evaluation

1) Create virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2) Run classifier training and evaluation (produces metrics and saved models):

```powershell
python Code\train_classifiers.py
```

Outputs are written to `src/ML-data/Code/output/classifiers` including:
- `*_metrics.json` for accuracy/precision/recall/f1 per model
- `classification_summary.json` aggregate results
- `best_models_per_strategy.json` best model by f1
- saved models and scalers

If you only want to evaluate saved models against a held-out test split, ask me and I will add a small `evaluate_models.py` script that loads models and prints a CSV of metrics.
