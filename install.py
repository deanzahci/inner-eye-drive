#!/usr/bin/env python3
"""
Installation script for Inner Eye Drive - Driver Drowsiness Detection
This script helps set up the environment and test the system.
"""

import subprocess
import sys
import os
import platform

def print_banner():
    """Print the project banner."""
    print("=" * 60)
    print("🚗 Inner Eye Drive - Driver Drowsiness Detection")
    print("   UC Berkeley AI Hackathon 2025")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing imports...")
    required_modules = [
        'cv2',
        'mediapipe',
        'numpy',
        'PIL',
        'flask',
        'flask_cors'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def test_camera():
    """Test if camera is accessible."""
    print("\n📹 Testing camera access...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is accessible")
                return True
            else:
                print("⚠️ Camera opened but couldn't read frame")
                return False
        else:
            print("❌ Could not open camera")
            return False
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
        return False

def run_quick_test():
    """Run a quick test of the drowsiness detector."""
    print("\n🚀 Running quick test...")
    try:
        # Change to Backend directory
        os.chdir('Backend')
        
        # Import and test the detector
        from model import DriverDrowsinessDetector
        
        # Create detector instance
        detector = DriverDrowsinessDetector()
        print("✅ Drowsiness detector created successfully")
        
        # Test with a simple image (create a test image)
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_drowsiness(test_image)
        
        print("✅ Drowsiness detection test completed")
        print(f"   Face detected: {result['face_detected']}")
        print(f"   Drowsy: {result['drowsy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        return False
    finally:
        # Change back to root directory
        os.chdir('..')

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the web interface:")
    print("   cd Backend")
    print("   python app.py")
    print("   Then open http://localhost:5000 in your browser")
    print()
    print("2. Or run the command line test:")
    print("   cd Backend")
    print("   python test_detector.py")
    print()
    print("3. For API usage, see the README.md file")
    print()
    print("🚗 Happy driving safely!")

def main():
    """Main installation function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Some modules failed to import. Please check the installation.")
        sys.exit(1)
    
    # Test camera (optional)
    camera_ok = test_camera()
    if not camera_ok:
        print("\n⚠️ Camera test failed. You may need to:")
        print("   - Grant camera permissions")
        print("   - Check if another application is using the camera")
        print("   - Install camera drivers")
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Quick test failed. Please check the error messages.")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 