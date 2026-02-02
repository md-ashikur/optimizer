# Research Methodology Diagram

```mermaid
flowchart TB
    Start([Research Start]) --> Problem[Problem Definition:<br/>Website Performance<br/>Analysis & Prediction]
    
    Problem --> DataCollection[Data Collection Phase]
    
    DataCollection --> DC1[Selenium WebDriver<br/>Navigation Timing API]
    DataCollection --> DC2[Lighthouse CLI<br/>Performance Audits]
    DataCollection --> DC3[Broken Links Analysis<br/>BeautifulSoup]
    
    DC1 --> Metrics1[Response Time<br/>Load Time<br/>DOM Content Loaded<br/>TTFB]
    DC2 --> Metrics2[Core Web Vitals<br/>LCP, FCP, CLS, INP<br/>Performance Score]
    DC3 --> Metrics3[Link Quality<br/>Metrics]
    
    Metrics1 --> DataPrep[Data Preparation<br/>& Feature Engineering]
    Metrics2 --> DataPrep
    Metrics3 --> DataPrep
    
    DataPrep --> Features[22 Performance Features<br/>Extracted & Normalized]
    
    Features --> Labeling{Labeling Strategy}
    
    Labeling --> KMeans[K-Means Clustering<br/>Pattern-Based Labels]
    Labeling --> Tertiles[Composite Score Tertiles<br/>Threshold-Based Labels]
    
    KMeans --> Labels[Performance Labels:<br/>Good / Average / Weak]
    Tertiles --> Labels
    
    Labels --> Split[Train/Test Split<br/>80/20 Ratio]
    
    Split --> Training[Model Training Phase]
    
    Training --> LGBM[LightGBM Classifier<br/>Gradient Boosting]
    Training --> Keras[Deep Neural Network<br/>Keras/TensorFlow]
    
    LGBM --> Tuning[Hyperparameter Tuning<br/>Cross-Validation]
    Keras --> Tuning
    
    Tuning --> Evaluation[Model Evaluation]
    
    Evaluation --> Metrics[Performance Metrics:<br/>Accuracy, Precision<br/>Recall, F1-Score]
    
    Metrics --> Validation[Model Validation<br/>Confusion Matrix<br/>SHAP Analysis]
    
    Validation --> Selection[Model Selection<br/>K-Means LGBM:<br/>Best Accuracy]
    
    Selection --> Deployment[Production Deployment]
    
    Deployment --> API[FastAPI Server<br/>Real-time Predictions]
    
    API --> Frontend[Next.js Dashboard<br/>User Interface]
    
    Frontend --> Results[Performance Analysis<br/>Recommendations<br/>Insights]
    
    Results --> End([Research Complete])
    
    style Start fill:#e1f5e1
    style Problem fill:#fff3cd
    style DataCollection fill:#cfe2ff
    style Features fill:#f8d7da
    style Labels fill:#d1ecf1
    style Selection fill:#d4edda
    style Results fill:#d1ecf1
    style End fill:#e1f5e1
```

## Research Methodology Overview

### Phase 1: Problem Definition
- Identify the need for automated website performance analysis
- Define performance classification categories (Good, Average, Weak)

### Phase 2: Data Collection
- **Selenium WebDriver**: Collect navigation timing metrics
- **Lighthouse**: Perform comprehensive performance audits
- **Web Scraping**: Analyze link quality and page structure

### Phase 3: Feature Engineering
- Extract 22 performance features across 6 categories
- Normalize and preprocess data
- Handle missing values and outliers

### Phase 4: Labeling Strategy
- **K-Means Clustering**: Unsupervised pattern-based labeling
- **Tertiles Method**: Composite score threshold-based labeling

### Phase 5: Model Development
- Train multiple models (LightGBM, Neural Networks)
- Hyperparameter optimization
- Cross-validation for generalization

### Phase 6: Evaluation & Selection
- Compare model performance
- SHAP analysis for explainability
- Select best-performing model (K-Means LGBM)

### Phase 7: Deployment
- FastAPI backend for predictions
- Next.js frontend for visualization
- Real-time performance analysis

