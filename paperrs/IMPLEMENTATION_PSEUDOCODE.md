# Implementation Pseudocode

## Training Pipeline

```pseudocode
ALGORITHM: Complete LightGBM Training Pipeline
INPUT: Labeled dataset CSV file
OUTPUT: Trained model and scaler saved to disk

1. LOAD dataset from CSV file:
   data = READ_CSV('All_thesis_data_labeled.csv')

2. SEPARATE features and labels:
   X = ALL_COLUMNS except 'Label'
   y = 'Label' column

3. SPLIT data with stratification:
   X_train, X_test, y_train, y_test = SPLIT_DATA(
       data = X, y,
       test_size = 0.2,
       random_state = 42,
       stratify = y  // ensure balanced class distribution
   )

4. NORMALIZE features using StandardScaler:
   scaler = CREATE_SCALER()
   X_train_scaled = scaler.FIT_TRANSFORM(X_train)
   X_test_scaled = scaler.TRANSFORM(X_test)

5. CONFIGURE LightGBM parameters:
   params = {
       objective: 'multiclass',
       num_class: 3,  // Good, Average, Weak
       metric: 'multi_logloss',
       learning_rate: 0.05,
       num_leaves: 31,
       random_state: 42
   }

6. CREATE training dataset:
   train_data = LIGHTGBM_DATASET(X_train_scaled, y_train)

7. TRAIN model:
   model = TRAIN_LIGHTGBM(
       parameters = params,
       training_data = train_data,
       num_boost_round = 100
   )

8. PREDICT on test set:
   y_pred_probabilities = model.PREDICT(X_test_scaled)
   y_pred_labels = ARGMAX(y_pred_probabilities, axis=1)

9. EVALUATE performance:
   accuracy = CALCULATE_ACCURACY(y_test, y_pred_labels)
   f1_macro = CALCULATE_F1_SCORE(y_test, y_pred_labels, average='macro')

10. DISPLAY results:
    PRINT "Accuracy: {accuracy}"
    PRINT "F1-Score: {f1_macro}"
    PRINT CLASSIFICATION_REPORT(y_test, y_pred_labels)

11. SAVE model and scaler:
    SAVE_TO_DISK(model, 'lightgbm_kmeans_model.pkl')
    SAVE_TO_DISK(scaler, 'scaler.pkl')

12. RETURN trained model, accuracy, f1_score
```

## Backend API Implementation

```pseudocode
ALGORITHM: FastAPI Backend Server
INPUT: HTTP requests from frontend
OUTPUT: Prediction results and recommendations

1. INITIALIZE FastAPI application:
   app = CREATE_FASTAPI(title="Web Performance Prediction API")

2. CONFIGURE CORS middleware:
   ADD_MIDDLEWARE(
       type = CORS,
       allow_origins = ["*"],  // all origins for development
       allow_methods = ["*"],  // all HTTP methods
       allow_headers = ["*"]   // all headers
   )

3. LOAD trained model and scaler at startup:
   model = LOAD_FROM_DISK('models/lightgbm_kmeans_model.pkl')
   scaler = LOAD_FROM_DISK('models/scaler.pkl')

4. DEFINE data model for request:
   STRUCTURE PerformanceMetrics {
       lcp: Float,
       fid: Float,
       cls: Float,
       fcp: Float,
       speed_index: Float,
       tbt: Float,
       server_response: Float,
       dom_content_loaded: Float,
       // ... 13 more features (total 21 features)
   }

5. ENDPOINT: POST /api/predict
   FUNCTION predict_performance(metrics: PerformanceMetrics):
   
   TRY:
       a. PREPARE feature vector:
          features = CREATE_ARRAY([
              metrics.lcp,
              metrics.fid,
              metrics.cls,
              metrics.fcp,
              metrics.speed_index,
              // ... all 21 features in correct order
          ])
          features = RESHAPE(features, shape=(1, 21))
       
       b. NORMALIZE features:
          features_scaled = scaler.TRANSFORM(features)
       
       c. PREDICT using model:
          prediction_probs = model.PREDICT(features_scaled)
          predicted_class_index = ARGMAX(prediction_probs)
          confidence = MAX(prediction_probs)
       
       d. MAP class index to label:
          label_mapping = ['Good', 'Average', 'Weak']
          predicted_label = label_mapping[predicted_class_index]
       
       e. GENERATE recommendations:
          recommendations = GENERATE_RECOMMENDATIONS(metrics)
       
       f. FORMAT response:
          result = {
              "prediction": predicted_label,
              "confidence": CONVERT_TO_PERCENTAGE(confidence),
              "probabilities": {
                  "Good": prediction_probs[0],
                  "Average": prediction_probs[1],
                  "Weak": prediction_probs[2]
              },
              "recommendations": recommendations
          }
       
       g. RETURN JSON_RESPONSE(result, status=200)
   
   CATCH Exception as e:
       RETURN ERROR_RESPONSE(
           status_code = 500,
           message = "Prediction failed: " + e.message
       )

6. ENDPOINT: GET /health
   FUNCTION health_check():
       RETURN {
           "status": "healthy",
           "model_loaded": model IS NOT NULL,
           "scaler_loaded": scaler IS NOT NULL,
           "service": "Web Performance Prediction API"
       }

7. ENDPOINT: GET /
   FUNCTION root():
       RETURN {
           "service": "WebOptimizer ML API",
           "model": "LightGBM (K-means labeling)",
           "accuracy": "98.47%",
           "f1_score": "98.47%",
           "version": "1.0"
       }

8. FUNCTION generate_recommendations(metrics: PerformanceMetrics):
   recommendations = []
   
   IF metrics.lcp > 4000 THEN:
       ADD "Critical: Optimize Largest Contentful Paint (LCP > 4s)"
       ADD "Consider lazy loading images and optimizing server response"
   
   IF metrics.fcp > 3000 THEN:
       ADD "Reduce First Contentful Paint by minimizing render-blocking resources"
   
   IF metrics.tbt > 600 THEN:
       ADD "High Total Blocking Time detected - optimize JavaScript execution"
   
   IF metrics.cls > 0.25 THEN:
       ADD "Poor Cumulative Layout Shift - add size attributes to media"
   
   IF metrics.speed_index > 5800 THEN:
       ADD "Slow Speed Index - optimize critical rendering path"
   
   // Additional checks for resource sizes, requests, etc.
   
   RETURN top_10_recommendations(recommendations)

9. START server:
   RUN_SERVER(
       app = app,
       host = "0.0.0.0",
       port = 8000,
       reload = True  // auto-reload during development
   )
```

## Model Prediction Flow

```pseudocode
ALGORITHM: Complete Prediction Workflow
INPUT: Website URL from user
OUTPUT: Performance grade and recommendations

1. FRONTEND submits URL to backend:
   REQUEST = POST /api/predict with performance_metrics

2. BACKEND receives request:
   metrics = PARSE_REQUEST_BODY()

3. VALIDATE input data:
   IF any required feature is missing THEN:
       RETURN ERROR("Missing required metrics")
   
   IF any feature value is negative THEN:
       RETURN ERROR("Invalid metric values")

4. PREPARE features in exact order:
   feature_vector = [
       metrics.Largest_contentful_paint_LCP_ms,
       metrics.First_Contentful_Paint_FCP_ms,
       metrics.Time_to_interactive_TTI_ms,
       metrics.Speed_Index_ms,
       metrics.Total_Blocking_Time_TBT_ms,
       metrics.Cumulative_Layout_Shift_CLS,
       metrics.Max_Potential_FID_ms,
       metrics.Server_Response_Time_ms,
       metrics.DOM_Content_Loaded_ms,
       metrics.First_Meaningful_Paint_ms,
       metrics.Fully_Loaded_Time_ms,
       metrics.Total_Page_Size_KB,
       metrics.Number_of_Requests,
       metrics.JavaScript_Size_KB,
       metrics.CSS_Size_KB,
       metrics.Image_Size_KB,
       metrics.Font_Size_KB,
       metrics.HTML_Size_KB,
       metrics.Main_Thread_Work_ms,
       metrics.Bootup_Time_ms,
       metrics.Offscreen_Images_KB
   ]

5. APPLY feature scaling:
   X_scaled = scaler.TRANSFORM(feature_vector)

6. PERFORM prediction:
   probabilities = model.PREDICT_PROBA(X_scaled)
   predicted_class = ARGMAX(probabilities)

7. INTERPRET results:
   grade = MAP_CLASS_TO_LABEL(predicted_class)
   confidence = probabilities[predicted_class]

8. GENERATE actionable recommendations:
   recommendations = ANALYZE_METRICS_AND_RECOMMEND(metrics)

9. FORMAT response:
   response = {
       "grade": grade,
       "confidence": confidence,
       "probabilities": {
           "Good": probabilities[0],
           "Average": probabilities[1],
           "Weak": probabilities[2]
       },
       "metrics": metrics,
       "recommendations": recommendations
   }

10. SEND response to frontend:
    RETURN JSON(response)

11. FRONTEND displays results:
    - Show performance grade badge
    - Display confidence percentage
    - Render probability bars
    - List recommendations with priorities
    - Show individual metric cards
```

## Error Handling

```pseudocode
ALGORITHM: Comprehensive Error Handling
INPUT: Various system states and user inputs
OUTPUT: User-friendly error messages with solutions

1. HANDLE model loading errors:
   TRY:
       model = LOAD_MODEL(path)
   CATCH FileNotFoundError:
       LOG "Model file not found at {path}"
       USE_FALLBACK_MODEL() or RETURN_ERROR(
           "Model not initialized. Please contact administrator."
       )
   CATCH Exception as e:
       LOG "Model loading failed: {e}"
       RETURN_ERROR("System initialization failed")

2. HANDLE prediction errors:
   TRY:
       prediction = model.PREDICT(features)
   CATCH ValueError:
       RETURN_ERROR("Invalid feature values provided")
   CATCH Exception as e:
       LOG_ERROR(e)
       RETURN_ERROR("Prediction failed. Please try again.")

3. HANDLE network errors (Frontend):
   TRY:
       response = SEND_API_REQUEST(url, data)
   CATCH ConnectionError:
       DISPLAY "Cannot connect to ML server. Please ensure it's running."
   CATCH TimeoutError:
       DISPLAY "Analysis timed out. Please try again."
   CATCH HTTPError as e:
       IF e.status_code == 500 THEN:
           DISPLAY "Server error. Please contact support."
       ELSE IF e.status_code == 400 THEN:
           DISPLAY "Invalid input. Please check the URL."

4. HANDLE validation errors:
   FUNCTION validate_metrics(metrics):
       errors = []
       
       IF metrics.lcp < 0 THEN:
           ADD_ERROR("LCP cannot be negative")
       
       IF metrics.cls < 0 OR metrics.cls > 1 THEN:
           ADD_ERROR("CLS must be between 0 and 1")
       
       IF LENGTH(feature_vector) != 21 THEN:
           ADD_ERROR("Incomplete feature set")
       
       IF errors IS NOT EMPTY THEN:
           RETURN VALIDATION_ERROR(errors)
       
       RETURN VALID

5. HANDLE missing dependencies:
   AT_STARTUP:
       REQUIRED_PACKAGES = ['lightgbm', 'sklearn', 'joblib', 'fastapi']
       
       FOR package IN REQUIRED_PACKAGES:
           IF NOT PACKAGE_INSTALLED(package) THEN:
               DISPLAY "Missing dependency: {package}"
               DISPLAY "Install with: pip install {package}"
               EXIT_WITH_ERROR
```
