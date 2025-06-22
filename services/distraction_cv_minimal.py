#!/usr/bin/env python3
"""
Minimal drowsiness detection service
This version provides basic simulation when full computer vision is not available
"""

from fastapi import FastAPI, WebSocket
import asyncio
import json
import uvicorn
import random
import math
import time

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

@app.get("/")
async def root():
    return {"message": "Distraction CV Service - Minimal Drowsiness Simulation"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

async def simulate_drowsiness_detection():
    """Simulate drowsiness detection with realistic patterns"""
    print("🎭 Starting drowsiness simulation mode...")
    
    counter = 0
    total_blinks = 0
    drowsiness_alerts = 0
    
    while True:
        counter += 1
        
        # Simulate natural blinking and occasional drowsiness
        base_ear = 0.3  # Normal open eye EAR
        
        # Add natural variation
        time_factor = time.time() / 10  # Slow variation
        ear_variation = 0.05 * math.sin(time_factor)
        
        # Simulate blinks (quick drops in EAR)
        if counter % 60 == 0:  # Blink every ~2 seconds at 30fps
            total_blinks += 1
            avg_ear = 0.15  # Closed eye EAR
            consecutive_frames = 2
            is_drowsy = False
            alert_level = "normal"
        
        # Simulate drowsiness episodes (longer periods of low EAR)
        elif counter % 300 < 50:  # Drowsy for 50 frames every 300 frames (~10 seconds)
            avg_ear = 0.18 + ear_variation
            consecutive_frames = (counter % 300) + 20
            is_drowsy = consecutive_frames > 20
            if consecutive_frames > 40:
                alert_level = "critical"
                if counter % 300 == 25:  # Count alert once per episode
                    drowsiness_alerts += 1
            else:
                alert_level = "warning"
        
        else:  # Normal alert state
            avg_ear = base_ear + ear_variation
            consecutive_frames = 0
            is_drowsy = False
            alert_level = "normal"
        
        # Generate detection data
        detection_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "face_detected": True,
            "drowsy": is_drowsy,
            "alert_level": alert_level,
            "left_ear": avg_ear + random.uniform(-0.02, 0.02),
            "right_ear": avg_ear + random.uniform(-0.02, 0.02),
            "avg_ear": avg_ear,
            "consecutive_closed_frames": consecutive_frames,
            "total_blinks": total_blinks,
            "drowsiness_alerts": drowsiness_alerts,
            "service": "drowsiness_detection",
            "mode": "simulation"
        }
        
        # Broadcast to all connected clients
        await manager.broadcast(json.dumps(detection_data))
        
        # Run at ~30 FPS
        await asyncio.sleep(0.033)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Distraction CV Service - Simulation Mode",
        "mode": "simulation"
    }

@app.get("/status")
async def get_status():
    return {
        "service": "Distraction CV - Drowsiness Detection",
        "detector_loaded": True,
        "camera_available": False,
        "connections": len(manager.active_connections),
        "mode": "simulation"
    }

@app.on_event("startup")
async def startup_event():
    print("🎭 Starting minimal drowsiness detection simulation...")
    asyncio.create_task(simulate_drowsiness_detection())

if __name__ == "__main__":
    print("🚀 Starting Distraction CV Service (Simulation Mode) on port 8001")
    print("🎭 This version provides realistic drowsiness detection simulation")
    uvicorn.run(app, host="127.0.0.1", port=8001)
