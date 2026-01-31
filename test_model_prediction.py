#!/usr/bin/env python3
"""
Test script to verify the ML server is using the correct model
and making proper predictions.
"""
import requests
import json

def test_server():
    """Test the ML server health and predictions"""
    server_url = "http://localhost:8000"
    
    # Test 1: Health check
    print("="*70)
    print("Testing ML Server Health...")
    print("="*70)
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        health = resp.json()
        print(f"Status: {health.get('status')}")
        print(f"Model Type: {health.get('model_type')}")
        print(f"Model Loaded: {health.get('model_loaded')}")
        print(f"Scaler Loaded: {health.get('scaler_loaded')}")
        print()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("\n⚠️  Make sure the ML server is running:")
        print("   python src/api/ml_server.py")
        return
    
    # Test 2: Root endpoint
    print("="*70)
    print("Testing Model Info...")
    print("="*70)
    try:
        resp = requests.get(f"{server_url}/", timeout=5)
        info = resp.json()
        print(f"Service: {info.get('service')}")
        print(f"Model: {info.get('model')}")
        print(f"Status: {info.get('status')}")
        print()
        
        # Check if using tertiles model
        model_name = info.get('model', '')
        if 'Tertile' in model_name or 'tertile' in model_name:
            print("✅ Server is using the TERTILE model (correct!)")
        elif 'K-means' in model_name or 'kmeans' in model_name:
            print("⚠️  Server is still using K-MEANS model (needs restart!)")
            print("   Please restart the ML server to load the tertiles model:")
            print("   1. Stop the current server (Ctrl+C)")
            print("   2. Run: python src/api/ml_server.py")
        else:
            print(f"⚠️  Unknown model type: {model_name}")
        print()
    except Exception as e:
        print(f"❌ Model info check failed: {e}")
        return
    
    print("="*70)
    print("Prediction test complete!")
    print("="*70)
    print("\nIf the server is using the wrong model, please restart it.")
    print("The tertiles model should correctly classify Google.com as 'Good'.")

if __name__ == "__main__":
    test_server()
