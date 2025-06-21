"""
Configuration file for Driver Distraction Detection System
"""

# IP Camera Configuration
IP_CAMERA_URL = "http://10.56.19.74:8080/video"

# Detection Parameters
YAW_THRESHOLD = 25.0  # Maximum allowed head yaw angle (left/right) in degrees
PITCH_THRESHOLD = 30.0  # Maximum allowed head pitch angle (up/down) in degrees
GAZE_THRESHOLD = 20.0  # Maximum allowed gaze angle in degrees
VISIBILITY_TIMEOUT = 3.0  # Time in seconds before considering driver not visible
DISTRACTION_PERSISTENCE = 2.0  # Time in seconds before triggering persistent distraction alert

# Alert Configuration
ENABLE_AUDIO_ALERT = True  # Enable audio beep alerts
ENABLE_LOGGING = True  # Enable event logging to file

# Performance Configuration
FPS_UPDATE_INTERVAL = 1.0  # How often to update FPS display (seconds)

# Display Configuration
SHOW_LANDMARKS = True  # Show facial landmarks and mesh
SHOW_HEAD_POSE_AXES = True  # Show head pose direction indicators

# Audio Alert Configuration (Windows only)
AUDIO_FREQUENCY = 800  # Frequency in Hz for audio alert
AUDIO_DURATION = 300  # Duration in milliseconds for audio alert

# Logging Configuration
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE = "distraction_events.log"  # Log file name

# MediaPipe Configuration
FACE_MESH_MAX_FACES = 1  # Maximum number of faces to detect
FACE_MESH_REFINE_LANDMARKS = True  # Enable refined landmarks
FACE_MESH_DETECTION_CONFIDENCE = 0.5  # Minimum detection confidence
FACE_MESH_TRACKING_CONFIDENCE = 0.5  # Minimum tracking confidence 