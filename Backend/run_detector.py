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
from config import IP_CAMERA_URL, EAR_THRESHOLD, CONSECUTIVE_FRAMES, ENABLE_AUDIO_ALERT, ENABLE_LOGGING

def main():
    """Main function to run the drowsiness detector."""
    print("🚗 Driver Drowsiness Detection System")
    print("=" * 50)
    
    print(f"📹 IP Camera URL: {IP_CAMERA_URL}")
    print(f"👁️ EAR Threshold: {EAR_THRESHOLD}")
    print(f"⏱️ Consecutive Frames: {CONSECUTIVE_FRAMES}")
    print()
    
    # Create detector instance
    detector = DriverDrowsinessDetector(
        ip_camera_url=IP_CAMERA_URL,
        ear_threshold=EAR_THRESHOLD,
        consecutive_frames=CONSECUTIVE_FRAMES,
        enable_audio_alert=ENABLE_AUDIO_ALERT,
        enable_logging=ENABLE_LOGGING
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