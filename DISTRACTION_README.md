# 🚗 Real-Time Driver Distraction Detection - IP Camera System

A comprehensive real-time driver distraction detection system that connects to Android IP Webcam for live video processing using MediaPipe Face Mesh for head pose estimation and gaze tracking.

## 🌟 Features

✅ **IP Camera Integration** - Connects to Android IP Webcam stream  
✅ **Head Pose Estimation** - Tracks yaw, pitch, and roll angles  
✅ **Gaze Direction Tracking** - Monitors eye movement and gaze direction  
✅ **Multi-modal Distraction Detection** - Head turns, tilts, gaze shifts, and visibility  
✅ **Real-time Processing** - Live video feed analysis  
✅ **MediaPipe Face Mesh** - Advanced 468-point facial landmark detection  
✅ **Configurable Thresholds** - Adjustable sensitivity for different scenarios  
✅ **Audio Alerts** - Beep sounds for distraction detection  
✅ **Event Logging** - Timestamped distraction events  
✅ **FPS Counter** - Real-time performance monitoring  
✅ **Visual Feedback** - Facial landmarks and head pose indicators  

## 🛠️ Technology Stack

- **Python 3.12** - Core programming language
- **MediaPipe** - Google's facial landmark detection and pose estimation
- **OpenCV** - Computer vision and video processing
- **IP Webcam** - Android app for video streaming
- **NumPy** - Numerical computations and 3D geometry
- **Logging** - Event tracking and debugging

## 📱 IP Webcam Setup

### 1. Install IP Webcam on Android
- Download "IP Webcam" from Google Play Store
- Install and open the app

### 2. Configure IP Webcam
- Connect your phone to the same WiFi network as your computer
- Open IP Webcam app
- Tap "Start server"
- Note the IP address shown (e.g., `http://10.56.19.74:8080`)

### 3. Access Video Stream
- The video stream URL will be: `http://[IP_ADDRESS]:8080/video`
- Test in browser: `http://10.56.19.74:8080/video`

## 🚀 Quick Start

### 1. Activate Virtual Environment
```bash
venv312\Scripts\activate
```

### 2. Run the Distraction Detection System
```bash
cd Backend
python run_distraction_detector.py
```

### 3. Test the System
- Look at the camera
- Try turning your head left/right
- Try looking up/down
- Try looking away from the camera
- Watch for distraction alerts

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Reset statistics |
| `l` | Toggle landmark display |

## ⚙️ Configuration

Edit `Backend/distraction_config.py` to customize parameters:

```python
# Detection sensitivity
YAW_THRESHOLD = 25.0  # Head turn threshold (degrees)
PITCH_THRESHOLD = 30.0  # Head tilt threshold (degrees)
GAZE_THRESHOLD = 20.0  # Gaze shift threshold (degrees)

# Timing parameters
VISIBILITY_TIMEOUT = 3.0  # Seconds before "not visible" alert
DISTRACTION_PERSISTENCE = 2.0  # Seconds before persistent alert

# IP Camera URL
IP_CAMERA_URL = "http://10.56.19.74:8080/video"
```

## 📊 How It Works

### 1. Video Capture
- Connects to IP Webcam stream via HTTP
- Processes frames in real-time
- Maintains optimal frame rate

### 2. Facial Landmark Detection
- MediaPipe Face Mesh detects 468 facial landmarks
- Provides 3D coordinates for precise tracking
- Enables head pose and gaze estimation

### 3. Head Pose Estimation
```
Yaw (Left/Right): Rotation around vertical axis
Pitch (Up/Down): Rotation around horizontal axis  
Roll (Tilt): Rotation around depth axis
```

**Key facial points used:**
- **Nose**: Reference point for head center
- **Left/Right Ears**: Calculate yaw angle
- **Left/Right Eyes**: Calculate pitch angle
- **Eye Corners**: Calculate roll angle

### 4. Gaze Direction Tracking
- Tracks iris position relative to eye corners
- Calculates horizontal and vertical gaze angles
- Detects when driver looks away from road

### 5. Distraction Detection Algorithm
- **Head Turn**: Yaw angle > 25° left/right
- **Head Tilt**: Pitch angle > 30° up/down
- **Gaze Shift**: Gaze angle > 20° from center
- **Not Visible**: No face detected for > 3 seconds

### 6. Alert System
- **Visual**: Red/orange text overlay with distraction type
- **Audio**: Beep sound for persistent distractions
- **Logging**: Timestamped events with duration and angles

## 📈 Performance Metrics

The system tracks:
- **FPS**: Real-time processing speed
- **Head Pose Angles**: Yaw, pitch, and roll in degrees
- **Gaze Angles**: Horizontal and vertical gaze direction
- **Distraction Duration**: Time spent in distracted state
- **Total Distractions**: Number of distraction events
- **Visibility Time**: Time since last face detection

## 🔧 Troubleshooting

### Connection Issues
```
❌ Error: Could not connect to IP camera
```
**Solutions:**
1. Check IP address in `distraction_config.py`
2. Ensure both devices on same network
3. Verify IP Webcam is running
4. Test URL in browser first

### Detection Issues
```
No face detected or inaccurate pose estimation
```
**Solutions:**
1. Improve lighting conditions
2. Ensure face is clearly visible
3. Adjust head pose thresholds
4. Check camera focus and positioning

### Performance Issues
```
Low FPS or lag
```
**Solutions:**
1. Reduce video resolution in IP Webcam
2. Lower MediaPipe confidence thresholds
3. Disable landmark display
4. Check network bandwidth

## 📝 Logging

Events are logged to `distraction_events.log`:
```
2024-01-15 14:30:25 - WARNING - DISTRACTION ALERT - Type: head_turn, Duration: 2.5s, Yaw: 28.5°, Pitch: 5.2°
2024-01-15 14:30:30 - WARNING - DISTRACTION ALERT - Type: not_visible, Duration: 3.2s, Yaw: 0.0°, Pitch: 0.0°
```

## 🎯 Use Cases

- **Personal Vehicles**: Real-time driver monitoring
- **Commercial Fleets**: Fleet safety management
- **Research**: Driver behavior studies
- **Education**: Driver safety training
- **Testing**: System validation and calibration

## 🔮 Advanced Features

### Custom Thresholds
```python
# Adjust for different individuals or scenarios
detector = DriverDistractionDetector(yaw_threshold=20.0)  # More sensitive
detector = DriverDistractionDetector(yaw_threshold=35.0)  # Less sensitive
```

### Multiple Camera Support
```python
# Switch between different IP cameras
detector = DriverDistractionDetector(ip_camera_url="http://192.168.1.100:8080/video")
```

### Custom Alert Sounds
```python
# Modify audio alert frequency and duration
AUDIO_FREQUENCY = 600  # Lower pitch
AUDIO_DURATION = 500  # Longer duration
```

## 📊 System Requirements

- **Python**: 3.12+
- **RAM**: 4GB+ recommended
- **Network**: Stable WiFi connection
- **Android**: IP Webcam app
- **Camera**: Front-facing phone camera

## 🚨 Safety Notes

- This system is for **assistance only**
- Do not rely solely on automated detection
- Always maintain proper driving attention
- System may have false positives/negatives
- Test thoroughly before production use

## 🔍 Technical Details

### Head Pose Calculation
```python
# Yaw calculation (left/right rotation)
ear_vector = right_ear - left_ear
yaw = math.degrees(math.atan2(ear_vector[0], abs(ear_vector[2])))

# Pitch calculation (up/down rotation)
eye_vector = (left_eye + right_eye) / 2 - nose
pitch = math.degrees(math.atan2(eye_vector[1], abs(eye_vector[2])))

# Roll calculation (head tilt)
eye_line = right_eye - left_eye
roll = math.degrees(math.atan2(eye_line[1], eye_line[0]))
```

### Gaze Direction Calculation
```python
# Gaze direction relative to eye centers
left_gaze = left_iris - left_corner
right_gaze = right_iris - right_corner

# Convert to angles
gaze_x = math.degrees(math.atan2(avg_gaze[0], 0.1))  # Horizontal
gaze_y = math.degrees(math.atan2(avg_gaze[1], 0.1))  # Vertical
```

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review log files
3. Verify network connectivity
4. Test with different lighting conditions

---

**Built for safer roads with real-time driver monitoring** 🚗👁️ 