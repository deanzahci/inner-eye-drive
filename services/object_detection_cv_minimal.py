#!/usr/bin/env python3
"""
Minimal test version of object detection service
This version only uses YOLO and basic dependencies
"""

from fastapi import FastAPI, WebSocket
import asyncio
import json
import uvicorn

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
    return {"message": "Object Detection CV Service - Minimal Mode"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

async def minimal_cv_processing():
    """Minimal computer vision with just basic object detection"""
    try:
        import cv2
        from ultralytics import YOLO
        
        # Try to load YOLO
        yolo_model = YOLO('yolov8n.pt')  # This will download if not available
        cap = cv2.VideoCapture(0)
        
        print("✅ Minimal CV mode: YOLO + Camera ready")
        
        while True:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Basic YOLO detection
                    results = yolo_model(frame, conf=0.25)
                    
                    objects = []
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        
                        for box, cls, conf in zip(boxes, classes, confs):
                            objects.append({
                                "object": yolo_model.names[int(cls)],
                                "confidence": float(conf),
                                "bbox": box.tolist()
                            })
                    
                    data = {
                        "timestamp": asyncio.get_event_loop().time(),
                        "objects": objects,
                        "total_objects": len(objects),
                        "mode": "minimal_cv"
                    }
                    
                    await manager.broadcast(json.dumps(data))
            
            await asyncio.sleep(0.1)  # 10 FPS
            
    except Exception as e:
        print(f"Minimal CV failed: {e}, using simulation")
        await simulation_mode()

async def simulation_mode():
    """Pure simulation mode"""
    counter = 0
    while True:
        counter += 1
        data = {
            "timestamp": asyncio.get_event_loop().time(),
            "objects": [
                {"object": "car", "confidence": 0.85, "bbox": [100, 100, 200, 200]},
                {"object": "person", "confidence": 0.76, "bbox": [300, 150, 350, 300]}
            ],
            "total_objects": 2,
            "mode": "simulation"
        }
        await manager.broadcast(json.dumps(data))
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting minimal object detection service...")
    asyncio.create_task(minimal_cv_processing())

if __name__ == "__main__":
    print("🔧 Minimal Object Detection CV Service")
    uvicorn.run(app, host="127.0.0.1", port=8003)
