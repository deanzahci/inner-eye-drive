#!/usr/bin/env python3
"""
Driver Distraction Detection Runner
Connects to IP Webcam and runs real-time distraction detection.
Binary classification: Safe Driving vs Distracted
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distraction_detector import DriverDistractionDetector
from config import (IP_CAMERA_URL, MODEL_PATH, INPUT_SHAPE, BINARY_LABELS, 
                   CONFIDENCE_THRESHOLD, ORIGINAL_CLASS_LABELS, CLASS_TO_BINARY_MAPPING)

def main():
    """Main function to run the distraction detector."""
    print("🚗 Driver Distraction Detection System")
    print("=" * 50)
    
    print(f"📹 IP Camera URL: {IP_CAMERA_URL}")
    print(f"🤖 Model Path: {MODEL_PATH}")
    print(f"🎯 Input Shape: {INPUT_SHAPE}")
    print(f"📊 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print()
    
    print("🎯 Binary Classification:")
    for class_id, label in BINARY_LABELS.items():
        print(f"   {class_id}: {label}")
    print()
    
    print("📋 Original 10 Classes (mapped to binary):")
    for class_id, label in ORIGINAL_CLASS_LABELS.items():
        binary_class = CLASS_TO_BINARY_MAPPING[class_id]
        binary_label = BINARY_LABELS[binary_class]
        print(f"   c{class_id}: {label} -> {binary_label}")
    print()
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please ensure the model.h5 file exists in the current directory.")
        print("You can download a pretrained model from:")
        print("https://github.com/toshi-k/kaggle-distracted-driver-detection")
        return
    
    # Create detector instance
    detector = DriverDistractionDetector(
        model_path=MODEL_PATH,
        ip_camera_url=IP_CAMERA_URL,
        input_shape=INPUT_SHAPE,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Start detection
    try:
        detector.process_ip_camera_stream()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your IP camera connection and try again.")

if __name__ == "__main__":
    main() 