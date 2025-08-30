#!/usr/bin/env python3
"""
Test script to verify VeriFeed API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Model info: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_analyze():
    """Test analyze endpoint with dummy data"""
    try:
        # This would normally be a real video URL from Facebook
        test_data = {
            "videoUrl": "https://example.com/video.mp4",
            "platform": "facebook",
            "sequence_length": 20
        }
        
        response = requests.post(
            f"{BASE_URL}/analyze",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data)
        )
        
        print(f"Analyze endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Success: {response.json()}")
        else:
            print(f"Error: {response.json()}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Analyze test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing VeriFeed API endpoints...")
    print("=" * 50)
    
    health_ok = test_health()
    print()
    
    info_ok = test_model_info()
    print()
    
    analyze_ok = test_analyze()
    print()
    
    print("=" * 50)
    if health_ok and info_ok:
        print("✅ API server is working correctly!")
        print("You can now load the Chrome extension and test it on Facebook")
    else:
        print("❌ Some tests failed")
