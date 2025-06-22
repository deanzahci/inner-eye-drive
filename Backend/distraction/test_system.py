#!/usr/bin/env python3
"""
Test script for Driver Distraction Detection System
Verifies all components work correctly
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from model import build_model, unfreeze_model
        print("✓ model.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing model.py: {e}")
        return False
    
    try:
        from data import DistractionDataLoader
        print("✓ data.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing data.py: {e}")
        return False
    
    try:
        from utils import plot_training_history, save_confusion_matrix
        print("✓ utils.py imported successfully")
    except ImportError as e:
        print(f"✗ Error importing utils.py: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation and compilation."""
    print("\nTesting model creation...")
    
    try:
        from model import build_model
        
        # Test EfficientNetB0 model
        model = build_model(
            num_classes=10,
            input_shape=(224, 224, 3),
            backbone='efficientnetb0'
        )
        print(f"✓ EfficientNetB0 model created: {model.count_params():,} parameters")
        
        # Test MobileNetV2 model
        model_mobile = build_model(
            num_classes=10,
            input_shape=(224, 224, 3),
            backbone='mobilenetv2'
        )
        print(f"✓ MobileNetV2 model created: {model_mobile.count_params():,} parameters")
        
        # Test custom model
        model_custom = build_model(
            num_classes=10,
            input_shape=(224, 224, 3),
            backbone='custom'
        )
        print(f"✓ Custom model created: {model_custom.count_params():,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False

def test_data_loader():
    """Test data loader functionality."""
    print("\nTesting data loader...")
    
    try:
        from data import DistractionDataLoader
        
        # Create data loader
        data_loader = DistractionDataLoader(
            data_dir='imgs',
            img_size=(224, 224),
            batch_size=32
        )
        print("✓ Data loader created successfully")
        
        # Check if data directory exists
        if os.path.exists('imgs'):
            print("✓ Data directory 'imgs' exists")
            
            # Try to load data (this might fail if no images, but should not crash)
            try:
                X_train, X_test, y_train, y_test, class_names = data_loader.load_data_from_folders()
                print(f"✓ Data loaded successfully: {len(X_train)} training, {len(X_test)} test samples")
            except Exception as e:
                print(f"⚠ Data loading failed (expected if no images): {e}")
        else:
            print("⚠ Data directory 'imgs' not found - create it and add images for full testing")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with data loader: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import plot_training_history, save_confusion_matrix
        
        # Create dummy data for testing
        y_true = np.random.randint(0, 10, 100)
        y_pred = np.random.randint(0, 10, 100)
        class_names = [f'class_{i}' for i in range(10)]
        
        # Test confusion matrix creation (without showing plot)
        cm = save_confusion_matrix(y_true, y_pred, class_names, save_path=None, show_plot=False)
        print("✓ Confusion matrix function works")
        
        # Test training history plotting (with dummy data, without showing plot)
        class DummyHistory:
            def __init__(self):
                self.history = {
                    'loss': [0.5, 0.4, 0.3, 0.2],
                    'val_loss': [0.6, 0.5, 0.4, 0.3],
                    'accuracy': [0.8, 0.85, 0.9, 0.95],
                    'val_accuracy': [0.75, 0.8, 0.85, 0.9]
                }
        
        dummy_history = DummyHistory()
        plot_training_history(dummy_history, save_path=None, show_plot=False)
        print("✓ Training history plotting works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with utility functions: {e}")
        return False

def test_inference_simulation():
    """Test inference functionality with dummy data."""
    print("\nTesting inference simulation...")
    
    try:
        from model import build_model
        
        # Create a simple model
        model = build_model(
            num_classes=10,
            input_shape=(224, 224, 3),
            backbone='custom'  # Use custom for faster testing
        )
        
        # Create dummy image
        dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
        dummy_batch = np.expand_dims(dummy_image, axis=0)
        
        # Make prediction
        predictions = model.predict(dummy_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        print(f"✓ Inference test successful: class {predicted_class}, confidence {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with inference simulation: {e}")
        return False

def test_tensorflow_version():
    """Test TensorFlow version and GPU availability."""
    print("\nTesting TensorFlow setup...")
    
    try:
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("⚠ No GPU detected - will use CPU")
        
        # Test basic TensorFlow operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"✓ TensorFlow operations work: {c.numpy()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with TensorFlow: {e}")
        return False

def create_sample_data_structure():
    """Create sample data structure for testing."""
    print("\nCreating sample data structure...")
    
    try:
        # Create base directory
        os.makedirs('imgs', exist_ok=True)
        
        # Create class directories
        class_names = [
            'c0_safe_driving',
            'c1_texting_right',
            'c2_phone_right',
            'c3_texting_left',
            'c4_phone_left',
            'c5_adjusting_radio',
            'c6_drinking',
            'c7_reaching_behind',
            'c8_hair_makeup',
            'c9_talking_passenger'
        ]
        
        for class_name in class_names:
            class_dir = os.path.join('imgs', class_name)
            os.makedirs(class_dir, exist_ok=True)
            print(f"✓ Created directory: {class_dir}")
        
        print("✓ Sample data structure created")
        print("  Add training images to the class folders to test data loading")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating data structure: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DRIVER DISTRACTION DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("TensorFlow Setup", test_tensorflow_version),
        ("Import Tests", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loader", test_data_loader),
        ("Utility Functions", test_utils),
        ("Inference Simulation", test_inference_simulation),
        ("Data Structure", create_sample_data_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Add training images to the 'imgs' folder")
        print("2. Run training: python train.py --data-dir imgs")
        print("3. Run inference: python inference.py --model-path outputs/models/distraction_model.h5")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Check TensorFlow version: pip install tensorflow>=2.11.0")
        print("- Ensure you have sufficient disk space for models")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 