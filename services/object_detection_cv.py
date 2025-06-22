from fastapi import FastAPI, WebSocket
import asyncio
import json
import uvicorn
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import scipy.special
from ultralytics import YOLO
from shapely.geometry import Polygon, LineString, box as shapely_box
import sys
import os

# Add the Ultra-Fast-Lane-Detection directory to the path
ultra_fast_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'models', 'object_cv', 'Ultra-Fast-Lane-Detection')
sys.path.append(ultra_fast_path)

from model.model import parsingNet
from utils.config import Config

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

# === ZONE CLASSIFIER ===
def classify_object_by_edges(bbox, top, right, bottom, left):
    x1, y1, x2, y2 = bbox
    box_poly = shapely_box(x1, y1, x2, y2)
    t, r, b, l = top.intersects(box_poly), right.intersects(box_poly), bottom.intersects(box_poly), left.intersects(box_poly)
    count = sum([t, r, b, l])
    if count == 0: return 8
    if count == 4: return 3
    if count == 1:
        return 1 if r else 2 if l else 3
    if count == 2:
        if t and l: return 4
        if t and r: return 5
        if b and l: return 6
        if b and r: return 7
        return 3
    return 3

# === COMPUTER VISION SETUP ===
CONFIG_PATH = os.path.join(ultra_fast_path, 'configs', 'tusimple.py')
MODEL_PATH = os.path.join(ultra_fast_path, 'weights', 'tusimple_18.pth')
YOLO_MODEL_PATH = os.path.join(ultra_fast_path, 'yolov8n.pt')
USE_GPU = False

device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
cls_num_per_lane = 56
row_anchor = [64 + i * 4 for i in range(cls_num_per_lane)]

# Initialize models (will be done in startup)
cfg = None
net = None
yolo_model = None
img_transforms = None
cap = None

@app.get("/")
async def root():
    return {"message": "Object Detection CV Service"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

async def process_computer_vision():
    """Main computer vision processing loop with robust error handling"""
    global cfg, net, yolo_model, img_transforms, cap
    
    print("🚀 Starting computer vision processing loop...")
    
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

            frame_data = []
            lanes = []
            
            # Lane detection (if model is available)
            if net is not None and cfg is not None and img_transforms is not None:
                try:
                    # Lane detection processing
                    input_frame = cv2.resize(frame, (800, 288))
                    input_pil = Image.fromarray(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
                    tensor_img = img_transforms(input_pil).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out = net(tensor_img)

                    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
                    col_sample_w = col_sample[1] - col_sample[0]
                    out = out[0].cpu().numpy()[:, ::-1, :]
                    prob = scipy.special.softmax(out[:-1, :, :], axis=0)
                    idx = np.arange(cfg.griding_num) + 1
                    idx = idx.reshape(-1, 1, 1)
                    loc = np.sum(prob * idx, axis=0)
                    out_pos = np.argmax(out, axis=0)
                    loc[out_pos == cfg.griding_num] = 0
                    out = loc

                    # Extract lane points
                    for i in range(out.shape[1]):
                        pts = []
                        for k in range(out.shape[0]):
                            if out[k, i] > 0:
                                x = int(out[k, i] * col_sample_w * frame.shape[1] / 800) - 1
                                y = int(frame.shape[0] * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                                pts.append((x, y))
                        if len(pts) > 6:
                            lanes.append(pts)
                except Exception as e:
                    print(f"Lane detection error: {e}")
            
            # Object detection with YOLO
            if yolo_model is not None:
                try:
                    results = yolo_model(frame, conf=0.25)
                    if len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                        classes = results[0].boxes.cls.cpu().numpy().astype(int)
                        confs = results[0].boxes.conf.cpu().numpy()

                        # If we have lanes, do zone classification
                        if len(lanes) >= 2:
                            # Calculate midpoints between lanes
                            midpoints = []
                            for p1, p2 in zip(lanes[0], lanes[1]):
                                mx = int((p1[0] + p2[0]) / 2)
                                my = int((p1[1] + p2[1]) / 2)
                                midpoints.append((mx, my))

                            if len(midpoints) >= 2:
                                bottom = midpoints[0]
                                top = midpoints[-1]

                                width = 200
                                trapezoid = np.array([
                                    [top[0] - 60, top[1]],
                                    [top[0] + 60, top[1]],
                                    [bottom[0] + width, bottom[1]],
                                    [bottom[0] - width, bottom[1]]
                                ], np.int32)

                                # Create zone edges
                                top_edge = LineString([trapezoid[0], trapezoid[1]])
                                right_edge = LineString([trapezoid[1], trapezoid[2]])
                                bottom_edge = LineString([trapezoid[2], trapezoid[3]])
                                left_edge = LineString([trapezoid[3], trapezoid[0]])

                                for box, cls, conf in zip(boxes, classes, confs):
                                    zone = classify_object_by_edges(box, top_edge, right_edge, bottom_edge, left_edge)
                                    label = yolo_model.names[cls]
                                    frame_data.append({
                                        "object": label,
                                        "zone": zone,
                                        "confidence": float(conf),
                                        "bbox": box.tolist()
                                    })
                        else:
                            # No lanes detected, just report objects without zones
                            for box, cls, conf in zip(boxes, classes, confs):
                                label = yolo_model.names[cls]
                                frame_data.append({
                                    "object": label,
                                    "zone": 0,  # No zone info available
                                    "confidence": float(conf),
                                    "bbox": box.tolist()
                                })
                except Exception as e:
                    print(f"YOLO detection error: {e}")

            # Prepare data to broadcast
            detection_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "lanes_detected": len(lanes),
                "objects": frame_data,
                "total_objects": len(frame_data),
                "lane_model_active": net is not None,
                "yolo_model_active": yolo_model is not None
            }

            # Broadcast to all connected clients
            await manager.broadcast(json.dumps(detection_data))
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Error in computer vision processing: {e}")
            await asyncio.sleep(1)

def initialize_models():
    """Initialize computer vision models with robust error handling"""
    global cfg, net, yolo_model, img_transforms, cap
    
    print("🔧 Initializing computer vision models...")
    
    # Check if required files exist
    required_files = [CONFIG_PATH, MODEL_PATH]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("📋 Required files for lane detection:")
        print(f"   - Config: {CONFIG_PATH}")
        print(f"   - Model weights: {MODEL_PATH}")
        return False
    
    try:
        print("📦 Loading lane detection model...")
        
        # Import with error handling
        try:
            from model.model import parsingNet
            from utils.config import Config
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Make sure you're running from the correct directory or check sys.path")
            return False
        
        # Load lane detection model
        cfg = Config.fromfile(CONFIG_PATH)
        cfg.test_model = MODEL_PATH
        
        print(f"🏗️  Creating parsingNet with backbone: {cfg.backbone}")
        net = parsingNet(pretrained=False, backbone=cfg.backbone,
                        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4), use_aux=False)
        
        print("📥 Loading model weights...")
        state_dict = torch.load(cfg.test_model, map_location=device)
        if 'model' in state_dict:
            net.load_state_dict(state_dict['model'], strict=False)
        else:
            net.load_state_dict(state_dict, strict=False)
        net.to(device).eval()
        print("✅ Lane detection model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading lane detection model: {e}")
        print("🔄 Continuing with YOLO-only mode...")
        net = None
        cfg = None
    
    try:
        print("📦 Loading YOLO model...")
        # Try different YOLO model paths
        yolo_paths = [
            YOLO_MODEL_PATH,
            os.path.join(ultra_fast_path, 'yolov8n.pt'),
            'yolov8n.pt',  # Let ultralytics download it
            os.path.join(os.path.dirname(__file__), '..', 'models', 'object_cv', 'yolov8n.pt')
        ]
        
        yolo_model = None
        for yolo_path in yolo_paths:
            try:
                print(f"   Trying: {yolo_path}")
                yolo_model = YOLO(yolo_path)
                print(f"✅ YOLO model loaded from: {yolo_path}")
                break
            except Exception as e:
                print(f"   Failed: {e}")
                continue
        
        if yolo_model is None:
            print("❌ Could not load YOLO model from any path")
            return False
            
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return False
    
    try:
        print("🎯 Setting up image transforms...")
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        print("📹 Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  Warning: Could not open camera (webcam might be in use or unavailable)")
            print("🔄 Service will continue with fallback mode when camera is needed")
        else:
            print("✅ Camera initialized successfully!")
        
        model_status = "Lane + YOLO" if net is not None else "YOLO only"
        print(f"🎉 Computer vision models initialized successfully! Mode: {model_status}")
        return True
        
    except Exception as e:
        print(f"❌ Error in final initialization: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize models and start computer vision processing"""
    if initialize_models():
        asyncio.create_task(process_computer_vision())
    else:
        print("Failed to initialize models, using fallback mode")
        asyncio.create_task(fallback_mode())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global cap
    if cap is not None:
        cap.release()
    print("🔧 Object Detection CV Service shutdown complete")

@app.get("/status")
async def get_status():
    """Get service status"""
    return {
        "service": "Object Detection CV",
        "models_loaded": net is not None and yolo_model is not None,
        "camera_available": cap is not None and cap.isOpened() if cap else False,
        "connections": len(manager.active_connections)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Object Detection CV Service",
        "models_loaded": net is not None and yolo_model is not None,
        "camera_status": "available" if (cap is not None and cap.isOpened()) else "unavailable"
    }

async def fallback_mode():
    """Fallback mode when models fail to initialize"""
    counter = 0
    while True:
        counter += 1
        data = {
            "timestamp": asyncio.get_event_loop().time(),
            "lanes_detected": 2,
            "objects": [{"object": "car", "zone": counter % 8 + 1, "confidence": 0.85, "bbox": [100, 100, 200, 200]}],
            "total_objects": 1,
            "mode": "fallback"
        }
        await manager.broadcast(json.dumps(data))
        await asyncio.sleep(1)

if __name__ == "__main__":
    print("🚀 Starting Object Detection CV Service on port 8003")
    print("🔧 Initializing lane detection + object detection integration")
    uvicorn.run(app, host="127.0.0.1", port=8003)