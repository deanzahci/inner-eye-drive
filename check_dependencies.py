#!/usr/bin/env python3
"""
Dependency checker for Inner Eye Drive Object Detection Service
Run this script to verify all required dependencies are available.
"""

import sys
import os
import importlib.util

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False

def main():
    print("🔍 Checking Inner Eye Drive Dependencies...")
    print("=" * 50)
    
    # Core Python packages
    print("\n📦 Core Python Packages:")
    core_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("asyncio", "AsyncIO"),
        ("json", "JSON"),
        ("websockets", "WebSockets"),
    ]
    
    core_ok = all(check_import(pkg, name) for pkg, name in core_packages)
    
    # Computer Vision packages
    print("\n🖼️  Computer Vision Packages:")
    cv_packages = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("scipy", "SciPy"),
        ("ultralytics", "YOLOv8"),
        ("shapely", "Shapely"),
    ]
    
    cv_ok = all(check_import(pkg, name) for pkg, name in cv_packages)
    
    # Ultra-Fast-Lane-Detection specific
    print("\n🛣️  Lane Detection Model:")
    ultra_fast_path = os.path.join(os.path.dirname(__file__), 
                                   'models', 'object_cv', 'Ultra-Fast-Lane-Detection')
    
    sys.path.append(ultra_fast_path)
    
    lane_packages = []
    try:
        from model.model import parsingNet
        print("✅ parsingNet model")
        lane_packages.append(True)
    except ImportError as e:
        print(f"❌ parsingNet model: {e}")
        lane_packages.append(False)
    
    try:
        from utils.config import Config
        print("✅ Config utils")
        lane_packages.append(True)
    except ImportError as e:
        print(f"❌ Config utils: {e}")
        lane_packages.append(False)
    
    lane_ok = all(lane_packages)
    
    # Required files
    print("\n📁 Required Files:")
    config_path = os.path.join(ultra_fast_path, 'configs', 'tusimple.py')
    model_path = os.path.join(ultra_fast_path, 'weights', 'tusimple_18.pth')
    yolo_path = os.path.join(ultra_fast_path, 'yolov8n.pt')
    
    files_ok = all([
        check_file_exists(config_path, "TuSimple Config"),
        check_file_exists(model_path, "Lane Detection Weights"),
        check_file_exists(yolo_path, "YOLO Model") or check_file_exists('yolov8n.pt', "YOLO Model (fallback)")
    ])
    
    # Camera check
    print("\n📹 Camera Check:")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera is available")
            cap.release()
            camera_ok = True
        else:
            print("⚠️  Camera not available (service will use fallback mode)")
            camera_ok = False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        camera_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DEPENDENCY SUMMARY:")
    print(f"   Core packages: {'✅ OK' if core_ok else '❌ MISSING'}")
    print(f"   CV packages: {'✅ OK' if cv_ok else '❌ MISSING'}")
    print(f"   Lane detection: {'✅ OK' if lane_ok else '❌ MISSING'}")
    print(f"   Required files: {'✅ OK' if files_ok else '❌ MISSING'}")
    print(f"   Camera: {'✅ OK' if camera_ok else '⚠️  UNAVAILABLE'}")
    
    if core_ok and cv_ok:
        if lane_ok and files_ok:
            print("\n🎉 ALL SYSTEMS GO! You can run the full service.")
        else:
            print("\n⚠️  PARTIAL FUNCTIONALITY: Service will run with YOLO-only mode.")
    else:
        print("\n❌ MISSING CRITICAL DEPENDENCIES: Please install missing packages.")
        print("\n💡 Installation suggestions:")
        if not core_ok:
            print("   pip install fastapi uvicorn websockets")
        if not cv_ok:
            print("   pip install opencv-python torch torchvision ultralytics shapely scipy pillow numpy")
    
    print("\n🚀 To start the service: python services/object_detection_cv.py")

if __name__ == "__main__":
    main()
