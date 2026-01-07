# Trained Models

This folder contains all saved machine learning models and scalers.

## Folder Structure:

### regression_models/
Models for predicting continuous values (composite performance scores)
- `model_rf.joblib` - Random Forest
- `model_lgbm.joblib` - LightGBM  
- `model_keras.h5` - Keras Neural Network
- `scaler.joblib` - Feature scaler

### classification_models/
Models for classifying performance into categories (Good/Average/Weak)

**3 Labeling Strategies:**
1. **Tertiles** - Split by 33rd/66th percentiles
2. **Weighted** - Weighted composite score (emphasizes LCP, TTI, Speed Index)
3. **K-means** - Clustering-based labels (BEST PERFORMANCE)

**3 Model Types per Strategy:**
- Random Forest (`*_rf.joblib`)
- LightGBM (`*_lgbm.joblib`)  
- Keras Neural Network (`*_keras.h5`)
- Scalers for preprocessing (`*_scaler.joblib`)

## Loading Models:

```python
import joblib
from tensorflow import keras

# Load classifier
model = joblib.load('classification_models/label_kmeans_lgbm.joblib')
scaler = joblib.load('classification_models/label_kmeans_scaler.joblib')

# Load Keras model
keras_model = keras.models.load_model('classification_models/label_kmeans_keras.h5')

# Load regression model
reg_model = joblib.load('regression_models/model_lgbm.joblib')
```

## Best Model:
**LightGBM with K-means labeling** (`label_kmeans_lgbm.joblib`)
- F1-Score: 98.47%
