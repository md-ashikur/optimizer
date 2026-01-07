import sys
sys.path.insert(0, 'src/api')
from pathlib import Path
import joblib

MODEL_DIR = Path('src/ML-data/4_Trained_Models/classification_models')
MODEL_PATH = MODEL_DIR / 'label_kmeans_lgbm.joblib'
SCALER_PATH = MODEL_DIR / 'label_kmeans_scaler.joblib'

print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")
print(f"Scaler exists: {SCALER_PATH.exists()}")

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")

if SCALER_PATH.exists():
    try:
        scaler = joblib.load(SCALER_PATH)
        print("✓ Scaler loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
