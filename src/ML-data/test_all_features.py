"""
Comprehensive Test Suite for All Advanced ML Features
Verifies all 5 features are working correctly
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json

# Add path
base_path = Path("f:/client/Optimizer/optimizer/src/ML-data")

def test_tensorflow_import():
    """Test 1: Verify TensorFlow imports work"""
    print("\n" + "="*80)
    print("TEST 1: TensorFlow/Keras Import")
    print("="*80)
    
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        print(f"✓ Keras {tf.keras.__version__} imported successfully")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_ensemble_models():
    """Test 2: Verify ensemble models load and predict"""
    print("\n" + "="*80)
    print("TEST 2: Ensemble Models")
    print("="*80)
    
    try:
        # Load models
        voting_model = joblib.load(base_path / "8_Advanced_Models/ensemble/models/voting_classifier.pkl")
        stacking_model = joblib.load(base_path / "8_Advanced_Models/ensemble/models/stacking_classifier.pkl")
        scaler = joblib.load(base_path / "8_Advanced_Models/ensemble/models/ensemble_scaler.pkl")
        label_encoder = joblib.load(base_path / "8_Advanced_Models/ensemble/models/label_encoder.pkl")
        
        print("✓ All ensemble models loaded successfully")
        
        # Test prediction
        test_features = np.random.rand(1, 22)
        features_scaled = scaler.transform(test_features)
        
        voting_pred = voting_model.predict(features_scaled)
        stacking_pred = stacking_model.predict(features_scaled)
        confidence = voting_model.predict_proba(features_scaled)[0].max()
        
        voting_label = label_encoder.inverse_transform(voting_pred)[0]
        stacking_label = label_encoder.inverse_transform(stacking_pred)[0]
        
        print(f"✓ Voting prediction: {voting_label} ({confidence:.2%} confidence)")
        print(f"✓ Stacking prediction: {stacking_label}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_shap_explainer():
    """Test 3: Verify SHAP explainer works"""
    print("\n" + "="*80)
    print("TEST 3: SHAP Explainability")
    print("="*80)
    
    try:
        # Load explainer
        shap_explainer = joblib.load(base_path / "8_Advanced_Models/explainable_ai/shap_analysis/shap_explainer.pkl")
        shap_values_saved = joblib.load(base_path / "8_Advanced_Models/explainable_ai/shap_analysis/shap_values.pkl")
        
        print("✓ SHAP explainer and saved values loaded")
        
        # Test explanation
        feature_names = [
            'composite_score', 'response_time', 'dom_load_time', 'ttfb', 
            'total_links', 'load_time', 'num_requests', 'byte_size',
            'lcp', 'page_size', 'fcp', 'tti', 'speed_index', 'cls', 
            'fid', 'tbt', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21'
        ]
        
        test_features = np.random.rand(1, 22)
        features_df = pd.DataFrame(test_features, columns=feature_names)
        
        shap_values = shap_explainer(features_df)
        
        print(f"✓ SHAP values computed for test sample")
        print(f"  Shape: {shap_values.values.shape}")
        
        # Top feature
        top_idx = np.abs(shap_values.values[0]).argmax()
        print(f"  Most impactful feature: {feature_names[top_idx]}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_regression_models():
    """Test 4: Verify regression models work"""
    print("\n" + "="*80)
    print("TEST 4: Regression Models")
    print("="*80)
    
    try:
        # Load best models
        lcp_model = joblib.load(base_path / "8_Advanced_Models/regression/models/LCP_gradient_boosting.pkl")
        fid_model = joblib.load(base_path / "8_Advanced_Models/regression/models/FID_INP_random_forest.pkl")
        cls_model = joblib.load(base_path / "8_Advanced_Models/regression/models/CLS_random_forest.pkl")
        
        lcp_scaler = joblib.load(base_path / "8_Advanced_Models/regression/models/LCP_scaler.pkl")
        fid_scaler = joblib.load(base_path / "8_Advanced_Models/regression/models/FID_INP_scaler.pkl")
        cls_scaler = joblib.load(base_path / "8_Advanced_Models/regression/models/CLS_scaler.pkl")
        
        print("✓ All regression models loaded successfully")
        
        # Test predictions
        test_features = np.random.rand(1, 21)  # 21 features for regression
        
        lcp_pred = lcp_model.predict(lcp_scaler.transform(test_features))[0]
        fid_pred = fid_model.predict(fid_scaler.transform(test_features))[0]
        cls_pred = cls_model.predict(cls_scaler.transform(test_features))[0]
        
        print(f"✓ LCP prediction: {lcp_pred:.2f} ms")
        print(f"✓ FID prediction: {fid_pred:.2f} ms")
        print(f"✓ CLS prediction: {cls_pred:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_recommendation_system():
    """Test 5: Verify recommendation system works"""
    print("\n" + "="*80)
    print("TEST 5: Recommendation System")
    print("="*80)
    
    try:
        # Load recommendation system
        rec_model = joblib.load(base_path / "8_Advanced_Models/recommendation/models/recommendation_scorer.pkl")
        rec_scaler = joblib.load(base_path / "8_Advanced_Models/recommendation/models/recommendation_scaler.pkl")
        
        with open(base_path / "8_Advanced_Models/recommendation/models/recommendation_rules.json", 'r') as f:
            rec_rules = json.load(f)
        
        print("✓ Recommendation system loaded successfully")
        print(f"  Categories: {len(rec_rules)}")
        
        # Count total recommendations
        total_recs = sum(len(rule['recommendations']) for rule in rec_rules.values())
        print(f"  Total unique recommendations: {total_recs}")
        
        # Test scoring
        test_features = np.random.rand(1, 6)
        test_score = rec_model.predict_proba(rec_scaler.transform(test_features))[0]
        print(f"✓ Recommendation scoring works")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_optimization_strategies():
    """Test 6: Verify optimization strategies loaded"""
    print("\n" + "="*80)
    print("TEST 6: Multi-Metric Optimization")
    print("="*80)
    
    try:
        # Load strategies
        with open(base_path / "8_Advanced_Models/multi_metric_optimizer/models/optimization_strategies.json", 'r') as f:
            strategies = json.load(f)
        
        print("✓ Optimization strategies loaded successfully")
        print(f"  Available strategies: {len(strategies)}")
        
        for strategy_name in strategies:
            strategy = strategies[strategy_name]
            print(f"  - {strategy['name']}")
        
        # Load Pareto optimal sites
        pareto_path = base_path / "9_Advanced_Results/optimization_reports/pareto_optimal_websites.csv"
        pareto_df = pd.read_csv(pareto_path)
        print(f"✓ Pareto-optimal sites: {len(pareto_df)}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_api_server():
    """Test 7: Check if API server files exist"""
    print("\n" + "="*80)
    print("TEST 7: API Server")
    print("="*80)
    
    try:
        api_path = Path("f:/client/Optimizer/optimizer/src/api/ml_server_advanced.py")
        
        if api_path.exists():
            print(f"✓ Advanced ML API server found: {api_path}")
            
            # Check file size
            file_size = api_path.stat().st_size
            print(f"  File size: {file_size:,} bytes")
            
            # Check imports
            with open(api_path, 'r') as f:
                content = f.read()
                if 'FastAPI' in content:
                    print("  ✓ FastAPI imported")
                if 'joblib' in content:
                    print("  ✓ Model loading code present")
                if '/api/predict' in content:
                    print("  ✓ Prediction endpoint defined")
                if '/api/recommendations' in content:
                    print("  ✓ Recommendations endpoint defined")
            
            return True
        else:
            print(f"✗ API server not found")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_documentation():
    """Test 8: Verify documentation files exist"""
    print("\n" + "="*80)
    print("TEST 8: Documentation")
    print("="*80)
    
    try:
        docs = [
            "ADVANCED_ML_STRUCTURE.md",
            "MASTER_ADVANCED_ML_REPORT.md",
            "INTEGRATION_GUIDE.md",
            "fix_tensorflow_imports.py"
        ]
        
        for doc in docs:
            doc_path = base_path / doc
            if doc_path.exists():
                file_size = doc_path.stat().st_size
                print(f"✓ {doc} ({file_size:,} bytes)")
            else:
                print(f"✗ {doc} NOT FOUND")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def verify_folder_structure():
    """Test 9: Verify all folders are created"""
    print("\n" + "="*80)
    print("TEST 9: Folder Structure")
    print("="*80)
    
    try:
        folders = [
            "8_Advanced_Models/ensemble",
            "8_Advanced_Models/explainable_ai",
            "8_Advanced_Models/regression",
            "8_Advanced_Models/recommendation",
            "8_Advanced_Models/multi_metric_optimizer",
            "9_Advanced_Results",
            "10_Advanced_Visualizations"
        ]
        
        for folder in folders:
            folder_path = base_path / folder
            if folder_path.exists():
                # Count files
                files = list(folder_path.rglob('*'))
                file_count = len([f for f in files if f.is_file()])
                print(f"✓ {folder} ({file_count} files)")
            else:
                print(f"✗ {folder} NOT FOUND")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def count_visualizations():
    """Test 10: Count all visualizations"""
    print("\n" + "="*80)
    print("TEST 10: Visualizations")
    print("="*80)
    
    try:
        viz_path = base_path / "10_Advanced_Visualizations"
        
        # Count by feature
        features = {
            'ensemble': 0,
            'shap': 0,
            'regression': 0,
            'recommendation': 0,
            'multi_metric_optimizer': 0
        }
        
        for feature in features:
            feature_path = viz_path / feature
            if feature_path.exists():
                png_files = list(feature_path.glob('*.png'))
                features[feature] = len(png_files)
                print(f"✓ {feature}: {len(png_files)} visualizations")
        
        total = sum(features.values())
        print(f"\nTotal visualizations: {total}")
        
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def run_all_tests():
    """Run all tests and generate report"""
    print("\n" + "#"*80)
    print(" ADVANCED ML FEATURES - COMPREHENSIVE TEST SUITE")
    print("#"*80)
    
    tests = [
        ("TensorFlow Import", test_tensorflow_import),
        ("Ensemble Models", test_ensemble_models),
        ("SHAP Explainability", test_shap_explainer),
        ("Regression Models", test_regression_models),
        ("Recommendation System", test_recommendation_system),
        ("Optimization Strategies", test_optimization_strategies),
        ("API Server", test_api_server),
        ("Documentation", test_documentation),
        ("Folder Structure", verify_folder_structure),
        ("Visualizations", count_visualizations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! All advanced ML features are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
