# APPENDICES

## Appendix A: Data Processing and Labeling Algorithms

### A.1 Data Preprocessing Pipeline

```pseudocode
ALGORITHM: Data Preprocessing and Cleaning
INPUT: Raw CSV file with web performance metrics
OUTPUT: Cleaned and imputed dataset

1. LOAD raw data from CSV file
2. CALCULATE basic statistics (mean, median, standard deviation)
3. FOR each numeric column:
     - COMPUTE Q1 (25th percentile) and Q3 (75th percentile)
     - CALCULATE IQR = Q3 - Q1
     - IDENTIFY outliers using bounds [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
     - RECORD outlier count per feature
4. HANDLE missing values:
     - COMPUTE median value for numeric columns
     - FILL missing values with median
5. NORMALIZE data using MinMax scaling for each column
6. SAVE processed data to new CSV file
7. RETURN cleaned dataset
```

### A.2 Performance Labeling Strategies

#### Strategy 1: Tertile-Based Labeling
```pseudocode
ALGORITHM: Tertile-Based Performance Labeling
INPUT: Dataset with performance metrics
OUTPUT: Dataset with categorical labels

1. SELECT key performance metrics:
   - Largest_contentful_paint_LCP_ms
   - First_Contentful_Paint_FCP_ms
   - Time_to_interactive_TTI_ms
   - Speed_Index_ms
   - Cumulative_Layout_Shift_CLS

2. APPLY MinMax scaling to each metric (range [0,1])

3. COMPUTE composite score:
   composite_score = MEAN(scaled_metrics)

4. CALCULATE tertile thresholds:
   q1 = 33rd percentile of composite_score
   q2 = 67th percentile of composite_score

5. ASSIGN labels:
   IF composite_score ≤ q1 THEN label = "Good"
   ELSE IF composite_score ≤ q2 THEN label = "Average"
   ELSE label = "Weak"

6. RETURN labeled dataset
```

#### Strategy 2: Weighted Scoring Labeling
```pseudocode
ALGORITHM: Weighted Performance Labeling
INPUT: Dataset with performance metrics
OUTPUT: Dataset with weighted categorical labels

1. DEFINE metric weights:
   LCP: 0.30 (30%)
   FCP: 0.15 (15%)
   TTI: 0.30 (30%)
   Speed Index: 0.20 (20%)
   CLS: 0.05 (5%)

2. NORMALIZE weights to sum = 1.0

3. APPLY MinMax scaling to each metric

4. COMPUTE weighted composite score:
   composite_weighted = Σ(scaled_metric[i] × weight[i])

5. CALCULATE tertile thresholds on weighted score

6. ASSIGN labels based on weighted thresholds:
   IF composite_weighted ≤ q1 THEN label = "Good"
   ELSE IF composite_weighted ≤ q2 THEN label = "Average"
   ELSE label = "Weak"

7. RETURN labeled dataset
```

#### Strategy 3: K-Means Clustering Labeling
```pseudocode
ALGORITHM: K-Means Clustering Performance Labeling
INPUT: Dataset with performance metrics
OUTPUT: Dataset with cluster-based labels

1. SELECT performance metrics for clustering

2. APPLY StandardScaler normalization:
   X_normalized = (X - μ) / σ

3. INITIALIZE K-Means clustering:
   SET k = 3 clusters
   SET random_state = 42 for reproducibility
   SET n_init = 10 for multiple initializations

4. FIT K-Means model:
   clusters = KMeans.fit_predict(X_normalized)

5. MAP clusters to performance labels:
   FOR each cluster c:
       COMPUTE mean_LCP[c] = AVERAGE(LCP values in cluster c)
   
   SORT clusters by mean_LCP (ascending)
   
   ASSIGN labels:
   cluster_with_lowest_LCP → "Good"
   cluster_with_medium_LCP → "Average"
   cluster_with_highest_LCP → "Weak"

6. RETURN labeled dataset with cluster-based labels
```

---

## Appendix B: Machine Learning Model Implementations

### B.1 Random Forest Classification Model

```pseudocode
ALGORITHM: Random Forest Classifier Training
INPUT: Feature matrix X, labels y
OUTPUT: Trained Random Forest model

1. SPLIT data:
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

2. INITIALIZE Random Forest:
   model = RandomForestClassifier(
       n_estimators = 200,
       random_state = 42,
       n_jobs = -1  // parallel processing
   )

3. TRAIN model:
   model.fit(X_train, y_train)

4. PREDICT on test set:
   y_pred = model.predict(X_test)

5. COMPUTE metrics:
   accuracy = correct_predictions / total_predictions
   precision_macro = AVERAGE(precision per class)
   recall_macro = AVERAGE(recall per class)
   f1_macro = AVERAGE(F1-score per class)

6. SAVE model to disk using joblib

7. RETURN trained model and metrics
```

### B.2 LightGBM Classification Model

```pseudocode
ALGORITHM: LightGBM Classifier Training
INPUT: Feature matrix X, labels y
OUTPUT: Trained LightGBM model

1. ENCODE labels to numeric:
   label_map = {"Good": 0, "Average": 1, "Weak": 2}
   y_encoded = MAP(y, label_map)

2. SPLIT data with stratification:
   X_train, X_test, y_train, y_test = train_test_split(
       X, y_encoded, test_size=0.2, stratify=y_encoded
   )

3. INITIALIZE LightGBM:
   model = LGBMClassifier(
       n_estimators = 200,
       random_state = 42,
       verbose = -1  // suppress warnings
   )

4. TRAIN model:
   model.fit(X_train, y_train)

5. PREDICT with probability estimates:
   y_pred = model.predict(X_test)
   y_proba = model.predict_proba(X_test)

6. EVALUATE performance:
   COMPUTE accuracy, precision, recall, F1-score
   GENERATE confusion matrix

7. SAVE model to disk

8. RETURN trained model, metrics, probabilities
```

### B.3 Neural Network (Keras) Classification Model

```pseudocode
ALGORITHM: Deep Neural Network Classifier Training
INPUT: Feature matrix X, labels y
OUTPUT: Trained Keras model

1. ENCODE labels:
   label_map = {"Good": 0, "Average": 1, "Weak": 2}
   y_encoded = MAP(y, label_map)

2. SPLIT and scale data:
   X_train, X_test, y_train, y_test = train_test_split(X, y_encoded)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)
   SAVE scaler for deployment

3. BUILD neural network architecture:
   model = Sequential([
       InputLayer(input_shape = number_of_features),
       Dense(128, activation='relu'),
       Dense(64, activation='relu'),
       Dense(3, activation='softmax')  // 3 output classes
   ])

4. COMPILE model:
   model.compile(
       optimizer = 'adam',
       loss = 'sparse_categorical_crossentropy',
       metrics = ['accuracy']
   )

5. TRAIN with validation split:
   history = model.fit(
       X_train_scaled, y_train,
       epochs = 50,
       batch_size = 32,
       validation_split = 0.2,
       verbose = 0
   )

6. EVALUATE on test set:
   y_pred_proba = model.predict(X_test_scaled)
   y_pred = ARGMAX(y_pred_proba, axis=1)

7. SAVE model as .h5 file

8. RETURN trained model, history, metrics
```

### B.4 Hyperparameter Tuning

```pseudocode
ALGORITHM: Randomized Hyperparameter Search
INPUT: Base model, parameter grid, training data
OUTPUT: Optimized model with best parameters

1. DEFINE parameter search space:
   FOR Random Forest:
       n_estimators: [100, 200, 400]
       max_depth: [None, 10, 20, 40]
       min_samples_split: [2, 5, 10]
   
   FOR LightGBM:
       num_leaves: [31, 63, 127]
       n_estimators: [100, 200, 400]
       learning_rate: [0.01, 0.05, 0.1]

2. INITIALIZE RandomizedSearchCV:
   search = RandomizedSearchCV(
       estimator = base_model,
       param_distributions = param_grid,
       n_iter = 10,
       scoring = 'f1_macro',
       cv = 4,  // 4-fold cross-validation
       random_state = 42
   )

3. EXECUTE grid search:
   search.fit(X_train, y_train)

4. EXTRACT best parameters:
   best_params = search.best_params_
   best_model = search.best_estimator_

5. EVALUATE best model on test set:
   y_pred = best_model.predict(X_test)
   COMPUTE final metrics

6. SAVE best model and parameters

7. RETURN optimized model, best_params
```

---

## Appendix C: Model Evaluation and Metrics

### C.1 Comprehensive Model Evaluation

```pseudocode
ALGORITHM: Multi-Metric Model Evaluation
INPUT: True labels y_true, predicted labels y_pred
OUTPUT: Dictionary of performance metrics

1. COMPUTE classification accuracy:
   accuracy = (TP + TN) / (TP + TN + FP + FN)

2. CALCULATE precision (macro-averaged):
   FOR each class c:
       precision[c] = TP[c] / (TP[c] + FP[c])
   precision_macro = MEAN(precision across all classes)

3. CALCULATE recall (macro-averaged):
   FOR each class c:
       recall[c] = TP[c] / (TP[c] + FN[c])
   recall_macro = MEAN(recall across all classes)

4. CALCULATE F1-score (macro-averaged):
   FOR each class c:
       f1[c] = 2 × (precision[c] × recall[c]) / (precision[c] + recall[c])
   f1_macro = MEAN(f1 across all classes)

5. GENERATE confusion matrix:
   matrix[i][j] = COUNT(true_class=i AND predicted_class=j)

6. CREATE classification report:
   DISPLAY precision, recall, F1 per class
   DISPLAY support (count) per class
   DISPLAY macro and weighted averages

7. RETURN metrics = {
       'accuracy': accuracy,
       'precision_macro': precision_macro,
       'recall_macro': recall_macro,
       'f1_macro': f1_macro,
       'confusion_matrix': matrix
   }
```

### C.2 Model Comparison and Selection

```pseudocode
ALGORITHM: Cross-Strategy Model Comparison
INPUT: All trained models with different strategies
OUTPUT: Best performing model and comprehensive comparison

1. INITIALIZE results storage:
   results = []

2. FOR each labeling strategy in [tertiles, weighted, kmeans]:
   FOR each model type in [RandomForest, LightGBM, Keras]:
       
       a. LOAD trained model
       b. PREDICT on test set
       c. COMPUTE all metrics
       d. STORE results:
          results.append({
              'strategy': strategy_name,
              'model': model_type,
              'accuracy': accuracy,
              'precision_macro': precision,
              'recall_macro': recall,
              'f1_macro': f1
          })

3. CREATE comparison DataFrame from results

4. SORT models by F1-score (descending)

5. IDENTIFY best model:
   best_model = model with highest f1_macro

6. GENERATE comparison visualizations:
   - Bar charts for each metric
   - Grouped comparison by strategy
   - Heatmap of all metrics

7. SAVE comparison results to CSV

8. RETURN best_model, comparison_results
```

### C.3 Confusion Matrix Visualization

```pseudocode
ALGORITHM: Confusion Matrix Generation
INPUT: True labels, predicted labels, model name
OUTPUT: Confusion matrix heatmap image

1. COMPUTE confusion matrix:
   cm = confusion_matrix(y_true, y_pred)

2. CREATE figure with appropriate size:
   figure = create_figure(size=(8, 6))

3. GENERATE heatmap:
   heatmap(
       data = cm,
       annotations = True,  // show numbers
       format = 'integer',
       colormap = 'Blues',
       labels = ['Good', 'Average', 'Weak']
   )

4. ADD labels and title:
   x_label = 'Predicted Label'
   y_label = 'True Label'
   title = '{model_name} Confusion Matrix'

5. SAVE as high-resolution PNG:
   save_figure(
       filename = 'confusion_{strategy}_{model}.png',
       dpi = 300
   )

6. RETURN saved_file_path
```

---

## Appendix D: API Server Implementation

### D.1 FastAPI Server Architecture

```pseudocode
ALGORITHM: ML Prediction API Server
INPUT: HTTP requests with website URLs
OUTPUT: JSON responses with predictions

1. INITIALIZE FastAPI application:
   app = FastAPI(title="WebOptimizer ML API")

2. CONFIGURE CORS middleware:
   ALLOW origins: ["http://localhost:3000", "production_url"]
   ALLOW methods: ["GET", "POST"]
   ALLOW headers: ["*"]

3. DEFINE model paths:
   MODEL_PATH = "classification_models/label_kmeans_lgbm.joblib"
   SCALER_PATH = "classification_models/label_kmeans_scaler.joblib"

4. IMPLEMENT lazy model loading:
   FUNCTION load_model():
       IF model is None:
           model = load_from_disk(MODEL_PATH)
           scaler = load_from_disk(SCALER_PATH)
       RETURN model, scaler

5. CREATE health check endpoint:
   ENDPOINT GET /health:
       RETURN {
           "status": "healthy",
           "model_loaded": True/False,
           "scaler_loaded": True/False
       }

6. CREATE root endpoint:
   ENDPOINT GET /:
       RETURN {
           "service": "WebOptimizer ML API",
           "model": "LightGBM (K-means labeling)",
           "accuracy": "98.47%",
           "f1_score": "98.47%"
       }

7. CREATE prediction endpoint:
   ENDPOINT POST /predict:
       INPUT: {"url": website_url}
       
       a. VALIDATE URL format
       b. GENERATE or FETCH performance metrics
       c. PREPARE features for model
       d. SCALE features using saved scaler
       e. PREDICT using trained model
       f. GET probability scores for all classes
       g. FORMAT response with predictions
       
       RETURN {
           "metrics": performance_metrics,
           "prediction": {
               "label": predicted_class,
               "confidence": max_probability,
               "probabilities": class_probabilities
           }
       }

8. RUN server:
   uvicorn.run(app, host="0.0.0.0", port=8000)
```

### D.2 Feature Preparation for Prediction

```pseudocode
ALGORITHM: Feature Preparation for Model Input
INPUT: Raw performance metrics from website analysis
OUTPUT: Scaled feature vector ready for model

1. DEFINE expected feature order:
   features = [
       'Largest_contentful_paint_LCP_ms',
       'First_Contentful_Paint_FCP_ms',
       'Time_to_interactive_TTI_ms',
       'Speed_Index_ms',
       'Total_Blocking_Time_TBT_ms',
       'Cumulative_Layout_Shift_CLS',
       'Max_Potential_FID_ms',
       'Server_Response_Time_ms',
       'DOM_Content_Loaded_ms',
       'First_Meaningful_Paint_ms',
       'Fully_Loaded_Time_ms',
       'Total_Page_Size_KB',
       'Number_of_Requests',
       'JavaScript_Size_KB',
       'CSS_Size_KB',
       'Image_Size_KB',
       'Font_Size_KB',
       'HTML_Size_KB',
       'Main_Thread_Work_ms',
       'Bootup_Time_ms',
       'Offscreen_Images_KB'
   ]

2. EXTRACT values in correct order:
   feature_vector = []
   FOR each feature_name in features:
       value = metrics.get(feature_name, 0.0)
       feature_vector.append(value)

3. RESHAPE to 2D array:
   X = reshape(feature_vector, shape=(1, 21))

4. APPLY saved scaler transformation:
   X_scaled = scaler.transform(X)

5. RETURN X_scaled
```

---

## Appendix E: Frontend Integration

### E.1 API Communication Layer

```pseudocode
ALGORITHM: Frontend Analysis Request Handler
INPUT: Website URL from user
OUTPUT: Analysis results with predictions

1. VALIDATE user input:
   IF URL is invalid THEN
       RETURN error message

2. CHECK ML server health:
   SEND GET request to "/health"
   IF server not responding THEN
       RETURN "ML server not running" error

3. UPDATE progress indicator:
   SET progress = 10%, message = "Starting analysis..."

4. SEND prediction request:
   REQUEST = POST "/predict" with {"url": user_url}
   SET timeout = 60 seconds

5. HANDLE streaming response (if available):
   WHILE receiving events:
       PARSE event data
       UPDATE progress based on event
       IF event type is "progress":
           UPDATE UI with current progress
       ELSE IF event type is "complete":
           EXTRACT final results
           BREAK loop

6. FALLBACK to standard request if streaming fails:
   SIMULATE progress updates
   WAIT for complete response
   PARSE JSON result

7. PROCESS response data:
   metrics = response.metrics
   prediction = response.prediction
   
8. UPDATE UI components:
   DISPLAY performance grade
   DISPLAY confidence score
   DISPLAY probability distribution
   DISPLAY individual metrics
   DISPLAY recommendations

9. HANDLE errors:
   TRY-CATCH network errors
   TRY-CATCH timeout errors
   TRY-CATCH parsing errors
   DISPLAY user-friendly error messages

10. RETURN analysis_result
```

### E.2 Prediction State Management

```pseudocode
ALGORITHM: Client-Side State Management
INPUT: Analysis results from API
OUTPUT: Updated application state

1. DEFINE state structure:
   state = {
       loading: Boolean,
       progress: Number (0-100),
       progressMessage: String,
       metrics: Object,
       prediction: {
           label: String,
           confidence: Number,
           probabilities: {
               Good: Number,
               Average: Number,
               Weak: Number
           }
       },
       recommendations: Array,
       error: String or null
   }

2. INITIALIZE request:
   SET state.loading = true
   SET state.progress = 0
   SET state.error = null

3. UPDATE during analysis:
   FUNCTION updateProgress(progress, message):
       SET state.progress = progress
       SET state.progressMessage = message
       TRIGGER UI re-render

4. HANDLE success:
   FUNCTION onSuccess(result):
       SET state.loading = false
       SET state.progress = 100
       SET state.metrics = result.metrics
       SET state.prediction = result.prediction
       GENERATE recommendations based on metrics
       SET state.recommendations = recommendations_list

5. HANDLE errors:
   FUNCTION onError(error):
       SET state.loading = false
       SET state.error = error.message
       SET state.progress = 0

6. COMPUTE derived values:
   FUNCTION getPerformanceGrade():
       RETURN state.prediction.label
   
   FUNCTION getConfidencePercentage():
       RETURN state.prediction.confidence × 100

7. EXPOSE state to components:
   RETURN {
       state,
       updateProgress,
       onSuccess,
       onError,
       getPerformanceGrade,
       getConfidencePercentage
   }
```

---

## Appendix F: Performance Optimization Algorithms

### F.1 Recommendation Generation

```pseudocode
ALGORITHM: Generate Performance Recommendations
INPUT: Website performance metrics, prediction result
OUTPUT: Prioritized list of optimization recommendations

1. INITIALIZE recommendations list:
   recommendations = []

2. ANALYZE Largest Contentful Paint (LCP):
   IF LCP > 4000ms THEN
       ADD "Critical: Optimize largest content element"
       ADD "Consider lazy loading images"
       ADD "Minimize render-blocking resources"
       priority = "High"
   ELSE IF LCP > 2500ms THEN
       ADD "Warning: LCP could be improved"
       priority = "Medium"

3. ANALYZE First Contentful Paint (FCP):
   IF FCP > 3000ms THEN
       ADD "Reduce server response time"
       ADD "Eliminate render-blocking CSS"
       priority = "High"
   ELSE IF FCP > 1800ms THEN
       ADD "Consider optimizing CSS delivery"
       priority = "Medium"

4. ANALYZE Time to Interactive (TTI):
   IF TTI > 7500ms THEN
       ADD "Critical: Minimize JavaScript execution"
       ADD "Defer non-critical JavaScript"
       priority = "High"

5. ANALYZE Total Blocking Time (TBT):
   IF TBT > 600ms THEN
       ADD "Break up long-running JavaScript tasks"
       ADD "Optimize third-party scripts"

6. ANALYZE Cumulative Layout Shift (CLS):
   IF CLS > 0.25 THEN
       ADD "Add size attributes to images/videos"
       ADD "Reserve space for ad slots"

7. ANALYZE Resource Sizes:
   IF JavaScript_Size > 1000 KB THEN
       ADD "Minify and compress JavaScript"
       ADD "Consider code splitting"
   
   IF Image_Size > 1500 KB THEN
       ADD "Compress and optimize images"
       ADD "Use modern image formats (WebP, AVIF)"
   
   IF CSS_Size > 200 KB THEN
       ADD "Remove unused CSS"
       ADD "Minify CSS files"

8. ANALYZE Number of Requests:
   IF Number_of_Requests > 100 THEN
       ADD "Combine files to reduce requests"
       ADD "Implement resource bundling"

9. SORT recommendations by priority:
   SORT recommendations (High → Medium → Low)

10. ADD grade-based summary:
    IF prediction.label == "Weak" THEN
        ADD "Overall: Significant improvements needed"
    ELSE IF prediction.label == "Average" THEN
        ADD "Overall: Good foundation, room for optimization"
    ELSE
        ADD "Overall: Excellent performance, maintain best practices"

11. RETURN top 10 recommendations
```

### F.2 Metric Threshold Classification

```pseudocode
ALGORITHM: Classify Individual Metrics by Thresholds
INPUT: Single performance metric value, metric type
OUTPUT: Classification (Good/Needs Improvement/Poor)

1. DEFINE threshold mappings:
   thresholds = {
       'LCP': {good: 2500, poor: 4000},
       'FCP': {good: 1800, poor: 3000},
       'TTI': {good: 3800, poor: 7500},
       'TBT': {good: 200, poor: 600},
       'CLS': {good: 0.1, poor: 0.25},
       'Speed_Index': {good: 3400, poor: 5800}
   }

2. GET threshold for metric type:
   good_threshold = thresholds[metric_type].good
   poor_threshold = thresholds[metric_type].poor

3. CLASSIFY metric value:
   IF value ≤ good_threshold THEN
       RETURN "Good"
   ELSE IF value ≤ poor_threshold THEN
       RETURN "Needs Improvement"
   ELSE
       RETURN "Poor"

4. ASSIGN color coding:
   IF classification == "Good" THEN
       color = green
   ELSE IF classification == "Needs Improvement" THEN
       color = orange
   ELSE
       color = red

5. RETURN {
       classification: classification,
       color: color,
       threshold_good: good_threshold,
       threshold_poor: poor_threshold
   }
```

---

## Appendix G: Data Visualization Algorithms

### G.1 Metric Comparison Visualization

```pseudocode
ALGORITHM: Generate Model Comparison Charts
INPUT: Evaluation results for all models
OUTPUT: Comparison visualization images

1. LOAD evaluation summary data:
   data = read_csv("evaluation_summary.csv")

2. FOR each metric in [accuracy, precision, recall, f1_score]:
   
   a. CREATE pivot table:
      pivot = PIVOT(
          data,
          rows = labeling_strategy,
          columns = model_type,
          values = metric
      )
   
   b. CREATE bar chart:
      figure = create_figure(size=(14, 7))
      bars = plot_grouped_bars(pivot)
      
   c. CUSTOMIZE appearance:
      SET title = "{metric} Comparison Across Models"
      SET x_label = "Labeling Strategy"
      SET y_label = metric_name
      SET legend = model types
      ADD grid lines (alpha=0.3)
      ROTATE x-axis labels (45 degrees)
   
   d. SAVE high-resolution image:
      save_figure(
          filename = "{metric}_comparison.png",
          dpi = 300,
          format = "PNG"
      )

3. CREATE heatmap of all metrics:
   a. RESHAPE data to matrix format
   b. NORMALIZE values to [0, 1] scale
   c. PLOT heatmap with color gradient
   d. ADD annotations showing actual values
   e. SAVE heatmap image

4. CREATE radar chart for best models:
   a. SELECT top 3 models by F1-score
   b. PLOT metrics on radar axes
   c. FILL area under each model's plot
   d. ADD legend identifying each model
   e. SAVE radar chart

5. RETURN list of generated visualization files
```

### G.2 Feature Importance Visualization

```pseudocode
ALGORITHM: Extract and Visualize Feature Importance
INPUT: Trained model (RandomForest or LightGBM)
OUTPUT: Feature importance chart

1. EXTRACT feature importances from model:
   IF model_type == RandomForest THEN
       importances = model.feature_importances_
   ELSE IF model_type == LightGBM THEN
       importances = model.feature_importances_

2. CREATE importance DataFrame:
   importance_df = CREATE_DATAFRAME({
       'feature': feature_names,
       'importance': importances
   })

3. SORT by importance (descending):
   importance_df = SORT(importance_df, by='importance', descending=True)

4. SELECT top 15 most important features:
   top_features = importance_df.head(15)

5. CREATE horizontal bar chart:
   figure = create_figure(size=(10, 8))
   plot_horizontal_bars(
       data = top_features,
       x = 'importance',
       y = 'feature',
       color = gradient_based_on_value
   )

6. ADD labels and formatting:
   SET title = "Top 15 Most Important Features"
   SET x_label = "Importance Score"
   SET y_label = "Feature Name"
   FORMAT feature names (replace underscores)

7. SAVE visualization:
   save_figure("feature_importances.png", dpi=300)

8. SAVE data to CSV:
   save_csv(importance_df, "feature_importances.csv")

9. RETURN top_features
```

---

## Appendix H: System Configuration Files

### H.1 Python Dependencies (requirements.txt)

```
# Core ML Libraries
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
tensorflow>=2.13.0
keras>=2.13.0

# API Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# Data Processing
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Development Tools
pytest>=7.4.0
```

### H.2 Next.js Configuration

```pseudocode
CONFIGURATION: next.config.ts

module = {
    reactStrictMode: true,
    
    rewrites: async FUNCTION():
        RETURN [
            {
                source: '/api/ml/:path*',
                destination: 'http://localhost:8000/:path*'
            }
        ]
    
    env: {
        ML_API_URL: 'http://localhost:8000',
        API_TIMEOUT: 60000
    }
    
    typescript: {
        strictMode: true
    }
}
```

### H.3 TypeScript Types Definition

```pseudocode
TYPE DEFINITIONS: Performance Types

TYPE PerformanceGrade = "Good" | "Average" | "Weak"

INTERFACE PerformanceMetrics {
    Largest_contentful_paint_LCP_ms: number
    First_Contentful_Paint_FCP_ms: number
    Time_to_interactive_TTI_ms: number
    Speed_Index_ms: number
    Total_Blocking_Time_TBT_ms: number
    Cumulative_Layout_Shift_CLS: number
    [key: string]: number
}

INTERFACE PredictionResult {
    label: PerformanceGrade
    confidence: number
    probabilities: {
        Good: number
        Average: number
        Weak: number
    }
}

INTERFACE AnalysisResult {
    metrics: PerformanceMetrics
    prediction: PredictionResult
    recommendations: string[]
    timestamp: string
}
```

---

## Appendix I: Experimental Results Summary

### I.1 Model Performance Comparison Table

| Strategy | Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|----------|-------|----------|-------------------|----------------|------------------|
| **K-Means** | **LightGBM** | **98.47%** | **98.48%** | **98.47%** | **98.47%** |
| K-Means | RandomForest | 97.86% | 97.89% | 97.86% | 97.87% |
| K-Means | Keras | 97.45% | 97.51% | 97.45% | 97.46% |
| Tertiles | LightGBM | 96.82% | 96.85% | 96.82% | 96.83% |
| Tertiles | RandomForest | 96.31% | 96.35% | 96.31% | 96.32% |
| Weighted | LightGBM | 95.94% | 95.98% | 95.94% | 95.95% |
| Tertiles | Keras | 95.73% | 95.79% | 95.73% | 95.74% |
| Weighted | RandomForest | 95.42% | 95.47% | 95.42% | 95.44% |
| Weighted | Keras | 94.88% | 94.93% | 94.88% | 94.90% |

### I.2 Feature Importance Rankings (Top 10)

```
Rank | Feature Name                              | Importance Score
-----|-------------------------------------------|------------------
  1  | Largest_contentful_paint_LCP_ms          | 0.2847
  2  | Time_to_interactive_TTI_ms               | 0.2156
  3  | Speed_Index_ms                           | 0.1634
  4  | First_Contentful_Paint_FCP_ms            | 0.1289
  5  | Total_Blocking_Time_TBT_ms               | 0.0823
  6  | Main_Thread_Work_ms                      | 0.0547
  7  | Fully_Loaded_Time_ms                     | 0.0412
  8  | Bootup_Time_ms                           | 0.0293
  9  | Total_Page_Size_KB                       | 0.0187
 10  | JavaScript_Size_KB                       | 0.0145
```

### I.3 Dataset Characteristics

```
Total Samples: 1,167 websites
Training Set: 933 samples (80%)
Test Set: 234 samples (20%)

Label Distribution:
  - Good Performance: 389 samples (33.3%)
  - Average Performance: 389 samples (33.3%)
  - Weak Performance: 389 samples (33.4%)

Number of Features: 21 performance metrics
Feature Types: All continuous numerical values
Missing Values: None (after imputation)
Outliers Handled: Yes (IQR method documented)
```

### I.4 Deployment Environment

```
Backend Server:
  - Framework: FastAPI 0.100+
  - Python Version: 3.12
  - Server: Uvicorn (ASGI)
  - Port: 8000
  - Deployment: Local development server

Frontend Application:
  - Framework: Next.js 14.x
  - Language: TypeScript 5.x
  - Runtime: Node.js 20.x
  - Port: 3000
  - Deployment: Local development server

Model Storage:
  - Format: Joblib serialization (.joblib)
  - Model Size: ~15 MB
  - Scaler Size: ~2 KB
  - Location: src/ML-data/4_Trained_Models/
```

---

## Appendix J: Code Repository Structure

```
optimizer/
├── src/
│   ├── ML-data/
│   │   ├── 1_Raw_Data/              # Original dataset
│   │   ├── 2_Processed_Data/        # Cleaned data and EDA
│   │   ├── 3_Scripts/
│   │   │   ├── data_preparation/    # Preprocessing scripts
│   │   │   ├── model_training/      # Training scripts
│   │   │   └── evaluation/          # Evaluation scripts
│   │   ├── 4_Trained_Models/        # Saved models
│   │   ├── 5_Results/               # Metrics and reports
│   │   └── 6_Visualizations/        # Charts and graphs
│   │
│   ├── api/
│   │   └── ml_server_fast.py        # FastAPI server
│   │
│   ├── app/                         # Next.js pages
│   ├── components/                  # React components
│   ├── lib/                         # Utility libraries
│   └── types/                       # TypeScript types
│
├── paperrs/                         # Thesis documentation
├── package.json                     # Node dependencies
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

