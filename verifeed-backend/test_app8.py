"""
DIAGNOSTIC SCRIPT - Test your backend setup
Run this to identify what's wrong
"""

import os
import sys
import importlib.util

print("="*70)
print("VERIFEED BACKEND DIAGNOSTIC TEST")
print("="*70)

# 1. Check Python version
print(f"\n1. Python Version: {sys.version}")

# 2. Check current directory
print(f"\n2. Current Directory: {os.getcwd()}")
print(f"   Contents: {os.listdir('.')}")

# 3. Check if models directory exists
models_dir = os.path.join(os.getcwd(), 'models')
print(f"\n3. Models Directory: {models_dir}")
print(f"   Exists: {os.path.exists(models_dir)}")
if os.path.exists(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    print(f"   Model files found: {model_files}")
else:
    print("   ERROR: Models directory not found!")

# 4. Check required packages
print("\n4. Checking Required Packages:")
required_packages = {
    'flask': 'Flask',
    'flask_cors': 'Flask-CORS',
    'torch': 'PyTorch',
    'torchvision': 'torchvision',
    'cv2': 'opencv-python',
    'face_recognition': 'face_recognition',
    'PIL': 'Pillow',
    'numpy': 'numpy'
}

missing_packages = []
for package, name in required_packages.items():
    spec = importlib.util.find_spec(package)
    if spec is None:
        print(f"   ❌ {name} - NOT INSTALLED")
        missing_packages.append(name)
    else:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✓ {name} - {version}")
        except Exception as e:
            print(f"   ⚠ {name} - installed but error: {e}")

if missing_packages:
    print(f"\n   MISSING PACKAGES: {', '.join(missing_packages)}")
    print(f"   Install with: pip install {' '.join(missing_packages)}")

# 5. Check CUDA availability
try:
    import torch
    print(f"\n5. CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"\n5. Error checking CUDA: {e}")

# 6. Test Flask import
print("\n6. Testing Flask Import:")
try:
    from flask import Flask
    test_app = Flask(__name__)
    print("   ✓ Flask imports successfully")
except Exception as e:
    print(f"   ❌ Flask import failed: {e}")

# 7. Check port 5000
print("\n7. Checking Port 5000:")
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 5000))
if result == 0:
    print("   ⚠ Port 5000 is already in use!")
    print("   Another process is using this port. Kill it first.")
else:
    print("   ✓ Port 5000 is available")
sock.close()

# 8. Try to load model architecture
print("\n8. Testing Model Architecture:")
try:
    import torch.nn as nn
    from torchvision import models
    
    model = models.resnext50_32x4d(weights='IMAGENET1K_V2')
    print("   ✓ Model architecture can be created")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

if missing_packages:
    print("\n⚠ ACTION REQUIRED: Install missing packages")
    print(f"Run: pip install {' '.join(missing_packages)}")
elif not os.path.exists(models_dir):
    print("\n⚠ ACTION REQUIRED: Create models directory or adjust path")
else:
    print("\n✓ All checks passed! Your backend should work.")
    print("Next step: Run 'python app8.py' and check the output")