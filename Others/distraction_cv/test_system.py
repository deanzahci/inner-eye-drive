#!/usr/bin/env python3
"""
Test script for Driver Distraction Detection System
Tests system components without requiring a model file.
Binary classification: Safe Driving vs Distracted
"""

import sys
import os
import numpy as np
import cv2
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (IP_CAMERA_URL, INPUT_SHAPE, BINARY_LABELS, CONFIDENCE_THRESHOLD,
                   ORIGINAL_CLASS_LABELS, CLASS_TO_BINARY_MAPPING)

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration file loading."""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import (IP_CAMERA_URL, INPUT_SHAPE, BINARY_LABELS, CONFIDENCE_THRESHOLD,
                          ORIGINAL_CLASS_LABELS, CLASS_TO_BINARY_MAPPING)
        print(f"✅ IP Camera URL: {IP_CAMERA_URL}")
        print(f"✅ Input Shape: {INPUT_SHAPE}")
        print(f"✅ Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"✅ Binary Labels: {len(BINARY_LABELS)} classes")
        
        for class_id, label in BINARY_LABELS.items():
            print(f"   {class_id}: {label}")
        
        print(f"✅ Original Labels: {len(ORIGINAL_CLASS_LABELS)} classes")
        print(f"✅ Binary Mapping: {len(CLASS_TO_BINARY_MAPPING)} mappings")
        
        return True
        
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

def test_camera_connection():
    """Test IP camera connection."""
    print("\n🧪 Testing IP camera connection...")
    
    try:
        cap = cv2.VideoCapture(IP_CAMERA_URL)
        
        if not cap.isOpened():
            print(f"❌ Could not connect to IP camera at {IP_CAMERA_URL}")
            print("   This is expected if IP Webcam is not running")
            return False
        
        print("✅ Successfully connected to IP camera")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret:
            print(f"✅ Successfully read frame: {frame.shape}")
        else:
            print("❌ Could not read frame from camera")
            return False
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_preprocessing():
    """Test image preprocessing functions."""
    print("\n🧪 Testing preprocessing...")
    
    try:
        # Create a dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"✅ Created dummy frame: {dummy_frame.shape}")
        
        # Test BGR to RGB conversion
        rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
        print(f"✅ BGR to RGB conversion: {rgb_frame.shape}")
        
        # Test resizing
        resized_frame = cv2.resize(rgb_frame, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
        print(f"✅ Resized frame: {resized_frame.shape}")
        
        # Test normalization
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        print(f"✅ Normalized frame: min={normalized_frame.min():.3f}, max={normalized_frame.max():.3f}")
        
        # Test batch dimension
        batch_frame = np.expand_dims(normalized_frame, axis=0)
        print(f"✅ Added batch dimension: {batch_frame.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_binary_mapping():
    """Test binary classification mapping."""
    print("\n🧪 Testing binary classification mapping...")
    
    try:
        # Test mapping from original classes to binary
        safe_count = 0
        distracted_count = 0
        
        for original_class, binary_class in CLASS_TO_BINARY_MAPPING.items():
            original_label = ORIGINAL_CLASS_LABELS[original_class]
            binary_label = BINARY_LABELS[binary_class]
            
            if binary_class == 0:
                safe_count += 1
            else:
                distracted_count += 1
            
            print(f"   c{original_class}: {original_label} -> {binary_label}")
        
        print(f"✅ Safe activities: {safe_count}")
        print(f"✅ Distracted activities: {distracted_count}")
        print(f"✅ Total mappings: {len(CLASS_TO_BINARY_MAPPING)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Binary mapping test failed: {e}")
        return False

def test_overlay_drawing():
    """Test overlay drawing functions."""
    print("\n🧪 Testing overlay drawing...")
    
    try:
        # Create a dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test drawing functions
        height, width = frame.shape[:2]
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (width - 10, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 140), (255, 255, 255), 2)
        
        # Draw binary classification text
        cv2.putText(frame, "Status: SAFE", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, "Binary: Safe Driving", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Confidence: 0.950", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw statistics
        cv2.putText(frame, "Safe: 85.0%", (width - 150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Distracted: 15.0%", (width - 150, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        print("✅ Overlay drawing test completed")
        return True
        
    except Exception as e:
        print(f"❌ Overlay drawing test failed: {e}")
        return False

def test_performance():
    """Test basic performance metrics."""
    print("\n🧪 Testing performance metrics...")
    
    try:
        start_time = time.time()
        frame_count = 0
        
        # Simulate processing 100 frames
        for i in range(100):
            # Simulate frame processing
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            rgb_frame = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(rgb_frame, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            batch_frame = np.expand_dims(normalized_frame, axis=0)
            
            frame_count += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        
        print(f"✅ Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"✅ Average FPS: {fps:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚗 Driver Distraction Detection System - Test Suite")
    print("🎯 Binary Classification: Safe Driving vs Distracted")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Binary Mapping", test_binary_mapping),
        ("Camera Connection", test_camera_connection),
        ("Preprocessing", test_preprocessing),
        ("Overlay Drawing", test_overlay_drawing),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n📋 Next steps:")
        print("   1. Download a pretrained model.h5 file")
        print("   2. Place it in the current directory")
        print("   3. Run: python run_distraction_detection.py")
        print("\n🎯 Binary Classification:")
        for class_id, label in BINARY_LABELS.items():
            print(f"   {class_id}: {label}")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check IP camera connection")
        print("   3. Verify configuration settings")

if __name__ == "__main__":
    main() 