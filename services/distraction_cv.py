from fastapi import FastAPI, WebSocket
import asyncio
import json
import uvicorn
import cv2
import sys
import os

# Add the drowsiness_cv directory to the path
drowsiness_cv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'models', 'drowsiness_cv')
sys.path.append(drowsiness_cv_path)

from drowsinessCV import DriverDrowsinessDetector
from config import EAR_THRESHOLD, CONSECUTIVE_FRAMES, ENABLE_AUDIO_ALERT, ENABLE_LOGGING

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Global variables for drowsiness detection
detector = None
cap = None

@app.get("/")
async def root():
    return {"message": "Distraction CV Service - Drowsiness Detection"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Distraction CV Service",
        "detector_loaded": detector is not None,
        "camera_status": "available" if (cap is not None and cap.isOpened()) else "unavailable"
    }

@app.get("/status")
async def get_status():
    """Get service status with statistics"""
    if detector is not None:
        return {
            "service": "Distraction CV - Drowsiness Detection",
            "detector_loaded": True,
            "camera_available": cap is not None and cap.isOpened() if cap else False,
            "connections": len(manager.active_connections),
            "total_blinks": getattr(detector, 'total_blinks', 0),
            "drowsiness_alerts": getattr(detector, 'drowsiness_alerts', 0)
        }
    else:
        return {
            "service": "Distraction CV - Drowsiness Detection",
            "detector_loaded": False,
            "camera_available": False,
            "connections": len(manager.active_connections)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

async def process_drowsiness_detection():
    """Main drowsiness detection processing loop"""
    global detector, cap
    
    print("🚀 Starting drowsiness detection processing...")
    
    while True:
        try:
            # Check if camera is available
            if cap is None or not cap.isOpened():
                print("📹 Camera not available, using fallback mode...")
                await asyncio.sleep(5)  # Wait before retrying
                continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                await asyncio.sleep(0.1)
                continue

            # Perform drowsiness detection
            result = detector.detect_drowsiness(frame)
            
            # Prepare data to broadcast
            detection_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "face_detected": result.get('face_detected', False),
                "drowsy": result.get('drowsy', False),
                "alert_level": result.get('alert_level', 'normal'),
                "left_ear": result.get('left_ear', 0.0),
                "right_ear": result.get('right_ear', 0.0),
                "avg_ear": result.get('avg_ear', 0.0),
                "consecutive_closed_frames": result.get('consecutive_closed_frames', 0),
                "total_blinks": getattr(detector, 'total_blinks', 0),
                "drowsiness_alerts": getattr(detector, 'drowsiness_alerts', 0),
                "service": "drowsiness_detection"
            }

            # Broadcast to all connected clients
            await manager.broadcast(json.dumps(detection_data))
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in drowsiness detection processing: {e}")
            await asyncio.sleep(1)

def initialize_drowsiness_detector():
    """Initialize drowsiness detection system"""
    global detector, cap
    
    try:
        print("🔧 Initializing drowsiness detection system...")
        
        # Create detector instance with reduced audio alerts for API usage
        detector = DriverDrowsinessDetector(
            ip_camera_url="",  # We'll use direct camera access
            ear_threshold=EAR_THRESHOLD,
            consecutive_frames=CONSECUTIVE_FRAMES,
            enable_audio_alert=False,  # Disable audio alerts for API service
            enable_logging=ENABLE_LOGGING
        )
        
        # Initialize camera (try multiple sources)
        camera_sources = [0, 1, 2]  # Try different camera indices
        cap = None
        
        for source in camera_sources:
            print(f"   Trying camera source: {source}")
            test_cap = cv2.VideoCapture(source)
            if test_cap.isOpened():
                cap = test_cap
                print(f"✅ Camera initialized successfully on source {source}")
                break
            else:
                test_cap.release()
        
        if cap is None:
            print("⚠️  Warning: Could not open camera")
            print("🔄 Service will continue with fallback mode")
            return False
        
        print("✅ Drowsiness detection system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing drowsiness detector: {e}")
        return False

async def fallback_mode():
    """Fallback mode when detector fails to initialize"""
    counter = 0
    while True:
        counter += 1
        # Simulate drowsiness detection data
        simulated_drowsy = (counter % 30) > 25  # Simulate drowsiness every 30 cycles
        
        data = {
            "timestamp": asyncio.get_event_loop().time(),
            "face_detected": True,
            "drowsy": simulated_drowsy,
            "alert_level": "warning" if simulated_drowsy else "normal",
            "left_ear": 0.15 if simulated_drowsy else 0.3,
            "right_ear": 0.15 if simulated_drowsy else 0.3,
            "avg_ear": 0.15 if simulated_drowsy else 0.3,
            "consecutive_closed_frames": 25 if simulated_drowsy else 0,
            "total_blinks": counter // 10,
            "drowsiness_alerts": counter // 30,
            "service": "drowsiness_detection",
            "mode": "fallback"
        }
        await manager.broadcast(json.dumps(data))
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Initialize detector and start drowsiness detection processing"""
    if initialize_drowsiness_detector():
        asyncio.create_task(process_drowsiness_detection())
    else:
        print("Failed to initialize drowsiness detector, using fallback mode")
        asyncio.create_task(fallback_mode())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global cap
    if cap is not None:
        cap.release()
    print("🔧 Distraction CV Service shutdown complete")

if __name__ == "__main__":
    print("🚀 Starting Distraction CV Service (Drowsiness Detection) on port 8001")
    print("🔧 Initializing MediaPipe face detection and EAR calculation")
    uvicorn.run(app, host="127.0.0.1", port=8001)