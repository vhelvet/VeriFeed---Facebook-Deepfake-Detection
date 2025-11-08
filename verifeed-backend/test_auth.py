"""
VeRiFeed Authentication Test Script
Tests all authentication endpoints and security features
"""

import requests
import json
import time

# Configuration - UPDATE THESE VALUES
API_URL = "http://localhost:5000"
API_KEY = "your-secure-api-key-here"  # Must match backend .env
ADMIN_KEY = "your-admin-key-here"     # Must match backend .env

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(name):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST: {name}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def print_success(message):
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    print(f"{RED}✗ {message}{RESET}")

def print_info(message):
    print(f"{YELLOW}ℹ {message}{RESET}")

def test_health_check():
    """Test 1: Health check endpoint (no auth required)"""
    print_test("Health Check (Public)")
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check successful")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Model loaded: {data.get('model_loaded')}")
            print_info(f"Device: {data.get('device')}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False

def test_token_generation():
    """Test 2: Generate JWT token"""
    print_test("Token Generation")
    
    try:
        # Test with correct API key
        response = requests.post(
            f"{API_URL}/auth/token",
            headers={"Content-Type": "application/json"},
            json={"api_key": API_KEY}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Token generated successfully")
            print_info(f"Token type: {data.get('type')}")
            print_info(f"Expires in: {data.get('expires_in')} seconds")
            print_info(f"Token (first 50 chars): {data.get('token')[:50]}...")
            return data.get('token')
        else:
            error_data = response.json()
            print_error(f"Token generation failed: {error_data.get('error')}")
            return None
            
    except Exception as e:
        print_error(f"Token generation error: {e}")
        return None

def test_invalid_token_generation():
    """Test 3: Try to generate token with wrong API key"""
    print_test("Invalid Token Generation (Should Fail)")
    
    try:
        response = requests.post(
            f"{API_URL}/auth/token",
            headers={"Content-Type": "application/json"},
            json={"api_key": "wrong-api-key"}
        )
        
        if response.status_code == 401:
            print_success("Correctly rejected invalid API key")
            return True
        else:
            print_error("Security issue: Invalid API key was accepted!")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def test_predict_no_auth():
    """Test 4: Try to access predict without authentication"""
    print_test("Predict Without Auth (Should Fail)")
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={"Content-Type": "application/json"},
            json={"frames": []}
        )
        
        if response.status_code == 401:
            print_success("Correctly rejected unauthorized request")
            return True
        else:
            print_error("Security issue: Unauthenticated request was accepted!")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def test_predict_with_token(token):
    """Test 5: Access predict with valid token"""
    print_test("Predict With Valid Token")
    
    if not token:
        print_error("No token available, skipping test")
        return False
    
    try:
        # Note: This will fail with "No frames provided" but that's OK
        # We're testing authentication, not the actual prediction
        response = requests.post(
            f"{API_URL}/predict",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            },
            json={"frames": []}
        )
        
        # We expect 400 (bad request - no frames) not 401 (unauthorized)
        if response.status_code == 400:
            error_data = response.json()
            if "frames" in error_data.get('error', '').lower():
                print_success("Authentication successful (endpoint accessible)")
                print_info("Got expected error about frames (authentication worked)")
                return True
        elif response.status_code == 401:
            print_error("Token was rejected (authentication failed)")
            return False
        else:
            print_info(f"Unexpected status code: {response.status_code}")
            print_info(f"Response: {response.json()}")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def test_model_info(token):
    """Test 6: Get model info with authentication"""
    print_test("Model Info (Authenticated)")
    
    if not token:
        print_error("No token available, skipping test")
        return False
    
    try:
        response = requests.get(
            f"{API_URL}/model/info",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Model info retrieved successfully")
            print_info(f"Model loaded: {data.get('loaded')}")
            print_info(f"Device: {data.get('device')}")
            print_info(f"Model file: {data.get('model_filename')}")
            return True
        else:
            print_error(f"Model info failed: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def test_model_reload_no_admin():
    """Test 7: Try to reload model without admin key"""
    print_test("Model Reload Without Admin Key (Should Fail)")
    
    try:
        response = requests.post(
            f"{API_URL}/model/reload",
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 403:
            print_success("Correctly rejected non-admin request")
            return True
        else:
            print_error("Security issue: Model reload accessible without admin key!")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def test_model_reload_with_admin():
    """Test 8: Reload model with admin key"""
    print_test("Model Reload With Admin Key")
    
    try:
        response = requests.post(
            f"{API_URL}/model/reload",
            headers={
                "Content-Type": "application/json",
                "X-Admin-Key": ADMIN_KEY
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Admin access granted")
            print_info(f"Success: {data.get('success')}")
            print_info(f"Message: {data.get('message')}")
            return True
        elif response.status_code == 403:
            print_error("Admin key was rejected")
            return False
        else:
            print_info(f"Unexpected response: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def test_api_key_authentication():
    """Test 9: Use API key directly (alternative to JWT)"""
    print_test("Direct API Key Authentication")
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": API_KEY
            },
            json={"frames": []}
        )
        
        # Should get 400 (no frames) not 401 (unauthorized)
        if response.status_code == 400:
            print_success("API key authentication successful")
            return True
        elif response.status_code == 401:
            print_error("API key was rejected")
            return False
        else:
            print_info(f"Unexpected status: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Test error: {e}")
        return False

def run_all_tests():
    """Run all authentication tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}VeRiFeed Authentication Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{YELLOW}Testing backend at: {API_URL}{RESET}\n")
    
    results = []
    token = None
    
    # Test 1: Public health check
    results.append(("Health Check", test_health_check()))
    time.sleep(0.5)
    
    # Test 2: Generate token
    token = test_token_generation()
    results.append(("Token Generation", token is not None))
    time.sleep(0.5)
    
    # Test 3: Invalid token generation
    results.append(("Invalid Token Rejection", test_invalid_token_generation()))
    time.sleep(0.5)
    
    # Test 4: Predict without auth
    results.append(("Unauthorized Access Blocked", test_predict_no_auth()))
    time.sleep(0.5)
    
    # Test 5: Predict with token
    results.append(("JWT Authentication", test_predict_with_token(token)))
    time.sleep(0.5)
    
    # Test 6: Model info
    results.append(("Model Info Access", test_model_info(token)))
    time.sleep(0.5)
    
    # Test 7: Model reload without admin
    results.append(("Admin Protection", test_model_reload_no_admin()))
    time.sleep(0.5)
    
    # Test 8: Model reload with admin
    results.append(("Admin Access", test_model_reload_with_admin()))
    time.sleep(0.5)
    
    # Test 9: API key auth
    results.append(("API Key Authentication", test_api_key_authentication()))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"{name:.<50} {status}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    
    if passed == total:
        print(f"{GREEN}✓ All tests passed! ({passed}/{total}){RESET}")
        print(f"{GREEN}Your authentication system is working correctly!{RESET}")
    else:
        print(f"{YELLOW}⚠ Some tests failed ({passed}/{total} passed){RESET}")
        print(f"{YELLOW}Please check the errors above.{RESET}")
    
    print(f"{BLUE}{'='*60}{RESET}\n")

if __name__ == "__main__":
    print(f"\n{YELLOW}⚠️  IMPORTANT: Make sure to update API_KEY and ADMIN_KEY at the top of this script!{RESET}")
    print(f"{YELLOW}⚠️  These must match the values in your backend .env file{RESET}\n")
    
    input("Press Enter to start tests...")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Tests interrupted by user{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Test suite error: {e}{RESET}\n")