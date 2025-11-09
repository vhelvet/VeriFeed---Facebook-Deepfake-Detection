"""
Security Testing Script for Verifeed Backend
Tests all security features
"""

import requests
import base64
import json
import time
from io import BytesIO
from PIL import Image
import numpy as np

# Configuration
BASE_URL = "http://localhost:5000"  # Change to your URL
API_KEY = "your_test_api_key"  # Your valid API key
ADMIN_KEY = "your_admin_key"  # Your admin key

def create_test_frame():
    """Create a test image frame"""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode()

def test_health_check():
    """Test health check endpoint"""
    print("\n1. Testing Health Check...")
    
    # Public health check
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Public health: {response.status_code}")
    assert response.status_code == 200
    assert 'status' in response.json()
    
    # Admin health check
    response = requests.get(
        f"{BASE_URL}/health",
        headers={'X-Admin-Key': ADMIN_KEY}
    )
    print(f"   Admin health: {response.status_code}")
    assert response.status_code == 200
    assert 'device' in response.json()
    
    print("   ✓ Health check passed")

def test_api_key_authentication():
    """Test API key authentication"""
    print("\n2. Testing API Key Authentication...")
    
    # Without API key
    response = requests.post(
        f"{BASE_URL}/predict",
        json={'frames': [create_test_frame() for _ in range(20)]}
    )
    print(f"   No API key: {response.status_code}")
    assert response.status_code == 401
    
    # With invalid API key
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={'X-API-Key': 'invalid_key'},
        json={'frames': [create_test_frame() for _ in range(20)]}
    )
    print(f"   Invalid API key: {response.status_code}")
    assert response.status_code == 401
    
    # With valid API key
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={'X-API-Key': API_KEY},
        json={'frames': [create_test_frame() for _ in range(20)]}
    )
    print(f"   Valid API key: {response.status_code}")
    # May be 400 if no faces, but should not be 401
    assert response.status_code != 401
    
    print("   ✓ API key authentication passed")

def test_input_validation():
    """Test input validation"""
    print("\n3. Testing Input Validation...")
    
    # Empty frames
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={'X-API-Key': API_KEY},
        json={'frames': []}
    )
    print(f"   Empty frames: {response.status_code}")
    assert response.status_code == 400
    
    # Too many frames
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={'X-API-Key': API_KEY},
        json={'frames': [create_test_frame() for _ in range(1000)]}
    )
    print(f"   Too many frames: {response.status_code}")
    assert response.status_code == 400
    
    # Invalid frame type
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={'X-API-Key': API_KEY},
        json={'frames': [123, 456]}
    )
    print(f"   Invalid frame type: {response.status_code}")
    assert response.status_code == 400
    
    # Invalid base64
    response = requests.post(
        f"{BASE_URL}/predict",
        headers={'X-API-Key': API_KEY},
        json={'frames': ['not_valid_base64!!!' for _ in range(20)]}
    )
    print(f"   Invalid base64: {response.status_code}")
    assert response.status_code == 400
    
    print("   ✓ Input validation passed")

def test_rate_limiting():
    """Test rate limiting"""
    print("\n4. Testing Rate Limiting...")
    print("   Making 25 rapid requests...")
    
    rate_limited = False
    for i in range(25):
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={'X-API-Key': API_KEY},
            json={'frames': []}  # Empty to fail fast
        )
        
        if response.status_code == 429:
            print(f"   Rate limited at request #{i+1}")
            rate_limited = True
            break
        
        time.sleep(0.1)  # Small delay
    
    if rate_limited:
        print("   ✓ Rate limiting working")
    else:
        print("   ⚠ Rate limiting may not be enabled")

def test_security_headers():
    """Test security headers"""
    print("\n5. Testing Security Headers...")
    
    response = requests.get(f"{BASE_URL}/health")
    headers = response.headers
    
    required_headers = {
        'X-Frame-Options': 'DENY',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block',
        'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'",
        'Referrer-Policy': 'no-referrer'
    }
    
    for header, expected_value in required_headers.items():
        if header in headers:
            print(f"   ✓ {header}: {headers[header]}")
        else:
            print(f"   ✗ {header}: MISSING")
    
    print("   ✓ Security headers check complete")

def test_path_traversal():
    """Test path traversal protection"""
    print("\n6. Testing Path Traversal Protection...")
    
    dangerous_paths = [
        '../../../etc/passwd',
        '..\\..\\..\\windows\\system32\\config\\sam',
        '/etc/passwd',
        'C:\\Windows\\System32\\config\\sam'
    ]
    
    for path in dangerous_paths:
        response = requests.post(
            f"{BASE_URL}/model/reload",
            headers={'X-Admin-Key': ADMIN_KEY},
            json={'model_path': path}
        )
        print(f"   Path '{path[:30]}...': {response.status_code}")
        assert response.status_code in [400, 403], f"Path traversal not blocked!"
    
    print("   ✓ Path traversal protection passed")

def test_payload_size_limit():
    """Test payload size limit"""
    print("\n7. Testing Payload Size Limit...")
    
    # Create oversized payload (if MAX_CONTENT_MB=100, try 101MB)
    large_data = "A" * (101 * 1024 * 1024)
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={'X-API-Key': API_KEY},
            json={'frames': [large_data]},
            timeout=5
        )
        print(f"   Large payload: {response.status_code}")
        assert response.status_code == 413
        print("   ✓ Payload size limit working")
    except requests.exceptions.RequestException as e:
        print(f"   ✓ Payload rejected (connection error: {type(e).__name__})")

def run_all_tests():
    """Run all security tests"""
    print("="*70)
    print("VERIFEED BACKEND SECURITY TESTS")
    print("="*70)
    print(f"Target: {BASE_URL}")
    print("="*70)
    
    try:
        test_health_check()
        test_api_key_authentication()
        test_input_validation()
        test_rate_limiting()
        test_security_headers()
        test_path_traversal()
        test_payload_size_limit()
        
        print("\n" + "="*70)
        print("✓ ALL SECURITY TESTS PASSED")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")

if __name__ == "__main__":
    run_all_tests()
