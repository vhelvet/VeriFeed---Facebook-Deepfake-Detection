#!/usr/bin/env python3
"""
Basic test script to verify the deepfake detection package functionality
"""

import sys
import os
import numpy as np

def test_model_creation():
    """Test if the model can be created successfully"""
    try:
        from model import Model
        model = Model(num_classes=2)
        print("✓ Model creation test passed")
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_data_loader_import():
    """Test if data loader classes can be imported"""
    try:
        from data_loader import VideoDataset, ValidationDataset, get_transforms
        print("✓ Data loader import test passed")
        return True
    except Exception as e:
        print(f"✗ Data loader import test failed: {e}")
        return False

def test_train_import():
    """Test if training functions can be imported"""
    try:
        from train import train_model, AverageMeter, calculate_accuracy
        print("✓ Training import test passed")
        return True
    except Exception as e:
        print(f"✗ Training import test failed: {e}")
        return False

def test_predict_import():
    """Test if prediction functions can be imported"""
    try:
        from predict import predict_video, predict_batch, generate_heatmap
        print("✓ Prediction import test passed")
        return True
    except Exception as e:
        print(f"✗ Prediction import test failed: {e}")
        return False

def test_dummy_prediction():
    """Test dummy prediction with random data"""
    try:
        import torch
        from model import Model
        
        # Create dummy input
        dummy_input = torch.randn(1, 20, 3, 112, 112)  # batch_size, seq_length, channels, height, width
        
        # Create model
        model = Model(num_classes=2)
        
        # Forward pass
        fmap, output = model(dummy_input)
        
        print(f"✓ Dummy prediction test passed")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Feature map shape: {fmap.shape}")
        print(f"  Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Dummy prediction test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running deepfake detection package tests...")
    print("=" * 50)
    
    tests = [
        test_model_creation,
        test_data_loader_import,
        test_train_import,
        test_predict_import,
        test_dummy_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All basic tests passed! The package structure is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
