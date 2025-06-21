# 🚗 Real-Time Driver Drowsiness Detection - IP Camera System

A comprehensive real-time driver drowsiness detection system that connects to Android IP Webcam for live video processing using MediaPipe Face Mesh and Eye Aspect Ratio (EAR) calculation.

## 🌟 Features

✅ **IP Camera Integration** - Connects to Android IP Webcam stream  
✅ **Real-time Processing** - Live video feed analysis  
✅ **MediaPipe Face Mesh** - Advanced 468-point facial landmark detection  
✅ **Eye Aspect Ratio (EAR)** - Scientific drowsiness measurement  
✅ **Multi-level Alerts** - Warning and critical alert levels  
✅ **Audio Alerts** - Beep sounds for drowsiness detection  
✅ **Event Logging** - Timestamped drowsiness events  
✅ **FPS Counter** - Real-time performance monitoring  
✅ **EAR Graph** - Visual EAR value tracking  
✅ **Configurable Parameters** - Easy threshold adjustment  

## 🛠️ Technology Stack

- **Python 3.12** - Core programming language
- **MediaPipe** - Google's facial landmark detection
- **OpenCV** - Computer vision and video processing
- **IP Webcam** - Android app for video streaming
- **NumPy** - Numerical computations
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

### 2. Run the Detection System
```bash
cd Backend
python run_detector.py
```

### 3. Test the System
- Look at the camera
- Try closing your eyes for a few seconds
- Watch for drowsiness alerts

## 🎮 Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Reset statistics |
| `s` | Toggle EAR graph display |

## ⚙️ Configuration

Edit `Backend/config.py` to customize parameters:

```python
# Detection sensitivity
EAR_THRESHOLD = 0.25  # Lower = more sensitive
CONSECUTIVE_FRAMES = 50  # Higher = less false positives

# IP Camera URL
IP_CAMERA_URL = "http://10.56.19.74:8080/video"

# Alerts
ENABLE_AUDIO_ALERT = True
ENABLE_LOGGING = True
```

## 📊 How It Works

### 1. Video Capture
- Connects to IP Webcam stream via HTTP
- Processes frames in real-time
- Maintains optimal frame rate

### 2. Facial Landmark Detection
- MediaPipe Face Mesh detects 468 facial landmarks
- Focuses on eye region landmarks (32 points per eye)
- Provides 3D coordinates for precise tracking

### 3. Eye Aspect Ratio (EAR) Calculation
```
EAR = (A + B) / (2.0 * C)
```
- **A, B**: Vertical distances between eye landmarks
- **C**: Horizontal distance between eye corners
- **Threshold**: 0.25 (configurable)

### 4. Drowsiness Detection Algorithm
- **Normal**: EAR > threshold (eyes open)
- **Warning**: EAR < threshold for 50 consecutive frames
- **Critical**: EAR < threshold for 100+ consecutive frames

### 5. Alert System
- **Visual**: Red/orange text overlay
- **Audio**: Beep sound (Windows)
- **Logging**: Timestamped events to file

## 📈 Performance Metrics

The system tracks:
- **FPS**: Real-time processing speed
- **EAR Values**: Left, right, and average eye ratios
- **Consecutive Closed Frames**: Current drowsiness streak
- **Total Blinks**: Complete eye closures
- **Drowsiness Alerts**: Warning/critical events

## 🔧 Troubleshooting

### Connection Issues
```
❌ Error: Could not connect to IP camera
```
**Solutions:**
1. Check IP address in `config.py`
2. Ensure both devices on same network
3. Verify IP Webcam is running
4. Test URL in browser first

### Performance Issues
```
Low FPS or lag
```
**Solutions:**
1. Reduce video resolution in IP Webcam
2. Lower MediaPipe confidence thresholds
3. Disable EAR graph display
4. Check network bandwidth

### Detection Issues
```
No face detected or inaccurate EAR
```
**Solutions:**
1. Improve lighting conditions
2. Adjust face position (front-facing)
3. Modify EAR threshold in config
4. Check camera focus

## 📝 Logging

Events are logged to `drowsiness_events.log`:
```
2024-01-15 14:30:25 - WARNING - DROWSINESS ALERT - EAR: 0.234, Level: warning, Consecutive Frames: 52
2024-01-15 14:30:30 - WARNING - DROWSINESS ALERT - EAR: 0.198, Level: critical, Consecutive Frames: 105
```

## 🎯 Use Cases

- **Personal Vehicles**: Real-time driver monitoring
- **Commercial Fleets**: Fleet safety management
- **Research**: Driver behavior studies
- **Education**: Driver safety training
- **Testing**: System validation and calibration

## 🔮 Advanced Features

### Custom EAR Thresholds
```python
# Adjust for different individuals
detector = DriverDrowsinessDetector(ear_threshold=0.21)  # More sensitive
detector = DriverDrowsinessDetector(ear_threshold=0.28)  # Less sensitive
```

### Multiple Camera Support
```python
# Switch between different IP cameras
detector = DriverDrowsinessDetector(ip_camera_url="http://192.168.1.100:8080/video")
```

### Custom Alert Sounds
```python
# Modify audio alert frequency and duration
AUDIO_FREQUENCY = 800  # Lower pitch
AUDIO_DURATION = 1000  # Longer duration
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

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review log files
3. Verify network connectivity
4. Test with different lighting conditions

---

**Built for safer roads with real-time driver monitoring** 🚗👁️ 