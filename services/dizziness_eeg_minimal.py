#!/usr/bin/env python3
"""
Minimal EEG dizziness detection service
This version provides simulation when OpenBCI/PyTorch is not available
"""

from fastapi import FastAPI, WebSocket
import asyncio
import json
import uvicorn
import random
import math
import time
import numpy as np

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

DIZZINESS_LEVELS = {
    0: "High",
    1: "Moderate", 
    2: "Low",
    3: "None"
}

@app.get("/")
async def root():
    return {"message": "Dizziness EEG Service - Minimal Simulation"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

async def simulate_eeg_analysis():
    """Simulate EEG analysis with realistic dizziness patterns"""
    print("🧠 Starting EEG dizziness simulation...")
    
    counter = 0
    
    while True:
        counter += 1
        
        # Simulate natural patterns
        time_factor = time.time() / 20  # Slow changes
        
        # Base probabilities
        base_none = 0.6 + 0.3 * math.sin(time_factor)
        base_low = 0.2 + 0.1 * math.sin(time_factor + 1)
        base_moderate = 0.15 + 0.1 * math.sin(time_factor + 2)
        base_high = 0.05 + 0.05 * math.sin(time_factor + 3)
        
        # Normalize probabilities
        total = base_none + base_low + base_moderate + base_high
        probs = [base_high/total, base_moderate/total, base_low/total, base_none/total]
        
        # Add some random variation
        for i in range(len(probs)):
            probs[i] += random.uniform(-0.05, 0.05)
            probs[i] = max(0, min(1, probs[i]))
        
        # Renormalize
        total = sum(probs)
        probs = [p/total for p in probs]
        
        # Determine dominant state
        dominant_state = np.argmax(probs)
        confidence = probs[dominant_state]
        
        # Generate analysis data
        analysis_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "dizziness_probabilities": {
                "none": float(probs[3]),
                "low": float(probs[2]), 
                "moderate": float(probs[1]),
                "high": float(probs[0])
            },
            "dominant_state": int(dominant_state),
            "state_name": DIZZINESS_LEVELS[dominant_state],
            "confidence": float(confidence),
            "window_duration": 1.024,  # Simulated window duration
            "sample_count": 256,
            "channels": 8,
            "service": "dizziness_eeg",
            "mode": "simulation"
        }
        
        # Broadcast to all connected clients
        await manager.broadcast(json.dumps(analysis_data))
        
        # Update every 2 seconds (simulating processing time)
        await asyncio.sleep(2)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Dizziness EEG Service - Simulation Mode",
        "model_loaded": True,
        "eeg_acquisition": "Simulation",
        "mode": "simulation"
    }

@app.get("/status")
async def get_status():
    return {
        "service": "Dizziness EEG - Neural Network Analysis",
        "model_loaded": True,
        "eeg_source": "Simulation",
        "connections": len(manager.active_connections),
        "channels": 8,
        "sample_rate": 250,
        "window_size": 256,
        "mode": "simulation"
    }

@app.on_event("startup")
async def startup_event():
    print("🧠 Starting minimal EEG dizziness analysis simulation...")
    asyncio.create_task(simulate_eeg_analysis())

if __name__ == "__main__":
    print("🚀 Starting Dizziness EEG Service (Simulation Mode) on port 8002")
    print("🧠 This version provides realistic EEG dizziness detection simulation")
    uvicorn.run(app, host="127.0.0.1", port=8002)
