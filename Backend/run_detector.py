#!/usr/bin/env python3
"""
Driver Drowsiness Detection Runner
Connects to IP Webcam and runs real-time drowsiness detection.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import DriverDrowsinessDetector

def main():
    """Main function to run the drowsiness detector."""
    print("🚗 Driver Drowsiness Detection System")
    print("=" * 50)
    
    # Configuration
    IP_CAMERA_URL = "http://10.56.19.74:8080/video"
    EAR_THRESHOLD = 0.25
    CONSECUTIVE_FRAMES = 50
    
    print(f"📹 IP Camera URL: {IP_CAMERA_URL}")
    print(f"👁️ EAR Threshold: {EAR_THRESHOLD}")
    print(f"⏱️ Consecutive Frames: {CONSECUTIVE_FRAMES}")
    print()
    
    # Create detector instance
    detector = DriverDrowsinessDetector(
        ip_camera_url=IP_CAMERA_URL,
        ear_threshold=EAR_THRESHOLD,
        consecutive_frames=CONSECUTIVE_FRAMES,
        enable_audio_alert=True,
        enable_logging=True
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