#!/usr/bin/env python3
"""
Driver Distraction Detection Runner
Connects to IP Webcam and runs real-time distraction detection.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distraction_detector import DriverDistractionDetector
from distraction_config import (
    IP_CAMERA_URL, YAW_THRESHOLD, PITCH_THRESHOLD, GAZE_THRESHOLD,
    VISIBILITY_TIMEOUT, DISTRACTION_PERSISTENCE, ENABLE_AUDIO_ALERT, ENABLE_LOGGING
)

def main():
    """Main function to run the distraction detector."""
    print("🚗 Driver Distraction Detection System")
    print("=" * 50)
    
    print(f"📹 IP Camera URL: {IP_CAMERA_URL}")
    print(f"🔄 Yaw Threshold: ±{YAW_THRESHOLD}°")
    print(f"📐 Pitch Threshold: ±{PITCH_THRESHOLD}°")
    print(f"👁️ Gaze Threshold: ±{GAZE_THRESHOLD}°")
    print(f"⏱️ Visibility Timeout: {VISIBILITY_TIMEOUT}s")
    print(f"🚨 Distraction Persistence: {DISTRACTION_PERSISTENCE}s")
    print()
    
    # Create detector instance
    detector = DriverDistractionDetector(
        ip_camera_url=IP_CAMERA_URL,
        yaw_threshold=YAW_THRESHOLD,
        pitch_threshold=PITCH_THRESHOLD,
        visibility_timeout=VISIBILITY_TIMEOUT,
        distraction_persistence=DISTRACTION_PERSISTENCE,
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