from fastapi import FastAPI, WebSocket
import asyncio
import json
import uvicorn
import numpy as np
import torch
import sys
import os
import threading
import queue
import time

# Add the dizziness_nn directory to the path
dizziness_nn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'models', 'dizziness_nn')
sys.path.append(dizziness_nn_path)

from config import PORT, CHANNEL_COUNT, SAMPLE_RATE, WINDOW_SIZE, OVERLAP, DIZZINESS_LEVELS

# Try to import OpenBCI (optional dependency)
try:
    from pyOpenBCI import OpenBCICyton
    OPENBCI_AVAILABLE = True
except ImportError:
    print("⚠️  pyOpenBCI not available. EEG simulation mode will be used.")
    OPENBCI_AVAILABLE = False

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

# Global variables for EEG acquisition and processing
eeg_buffer = []  # Will store tuples of (timestamp, data_array)
eeg_queue = queue.Queue(maxsize=50)
model = None
board = None
acquisition_thread = None
processing_thread = None

@app.get("/")
async def root():
    return {"message": "Dizziness EEG Service - Neural Network Analysis"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Dizziness EEG Service",
        "model_loaded": model is not None,
        "eeg_acquisition": "OpenBCI" if board is not None else "Simulation",
        "openbci_available": OPENBCI_AVAILABLE
    }

@app.get("/status")
async def get_status():
    """Get service status with detailed information"""
    return {
        "service": "Dizziness EEG - Neural Network Analysis",
        "model_loaded": model is not None,
        "eeg_source": "OpenBCI" if board is not None else "Simulation",
        "openbci_available": OPENBCI_AVAILABLE,
        "connections": len(manager.active_connections),
        "buffer_size": len(eeg_buffer),
        "queue_size": eeg_queue.qsize(),
        "channels": CHANNEL_COUNT,
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

def load_dizziness_model():
    """Load the trained dizziness detection model"""
    global model
    try:
        print("🧠 Loading dizziness detection neural network...")
        
        # Import the model class and evaluation function
        sys.path.append(dizziness_nn_path)
        from dizziness_eeg import SimpleMLP
        
        model_path = os.path.join(dizziness_nn_path, 'model_weights.pth')
        
        if not os.path.exists(model_path):
            print(f"❌ Model weights file not found: {model_path}")
            return False
        
        model = SimpleMLP()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        
        print("✅ Dizziness detection model loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading dizziness model: {e}")
        return False

def stream_callback(sample):
    """Callback for real OpenBCI data"""
    global eeg_buffer
    
    # Scale factor for OpenBCI Cyton
    SCALE_FACTOR = (4500000) / 24 / (2 ** 23 - 1)  # µV/count
    
    # Convert raw data to microvolts
    scaled_data = np.array(sample.channels_data[:CHANNEL_COUNT]) * SCALE_FACTOR
    timestamp = time.time()
    
    eeg_buffer.append((timestamp, scaled_data))
    
    # Keep buffer size manageable
    if len(eeg_buffer) > 1000:
        eeg_buffer = eeg_buffer[-500:]  # Keep last 500 samples

def generate_eeg_simulation():
    """Generate simulated EEG data"""
    global eeg_buffer
    
    print("🎭 Generating simulated EEG data...")
    
    start_time = time.time()
    sample_count = 0
    
    while True:
        # Generate realistic EEG-like data
        current_time = time.time()
        
        # Create multi-channel EEG simulation
        eeg_sample = np.zeros(CHANNEL_COUNT)
        
        for channel in range(CHANNEL_COUNT):
            # Base noise
            eeg_sample[channel] = np.random.normal(0, 10)  # µV
            
            # Add different frequency components
            # Alpha waves (8-13 Hz)
            alpha_freq = 8 + channel * 0.5
            eeg_sample[channel] += 15 * np.sin(2 * np.pi * alpha_freq * current_time)
            
            # Beta waves (13-30 Hz)
            beta_freq = 15 + channel * 2
            eeg_sample[channel] += 8 * np.sin(2 * np.pi * beta_freq * current_time)
            
            # Theta waves (4-8 Hz) - more prominent when "dizzy"
            theta_freq = 6 + channel * 0.3
            dizziness_factor = 1 + 0.5 * np.sin(0.1 * current_time)  # Slow variation
            eeg_sample[channel] += 12 * dizziness_factor * np.sin(2 * np.pi * theta_freq * current_time)
        
        eeg_buffer.append((current_time, eeg_sample))
        sample_count += 1
        
        # Keep buffer size manageable
        if len(eeg_buffer) > 1000:
            eeg_buffer = eeg_buffer[-500:]
        
        # Maintain sampling rate
        target_time = start_time + sample_count / SAMPLE_RATE
        sleep_time = target_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

def try_connect_openbci():
    """Try to connect to OpenBCI device"""
    global board
    
    if not OPENBCI_AVAILABLE:
        return False
    
    try:
        print(f"🔌 Attempting to connect to OpenBCI on port {PORT}...")
        board = OpenBCICyton(port=PORT, daisy=False)
        print("✅ OpenBCI Cyton connected successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to OpenBCI: {e}")
        return False

def epoch_and_analyze():
    """Process EEG buffer and run dizziness analysis"""
    global eeg_buffer, eeg_queue
    
    while True:
        try:
            if len(eeg_buffer) >= WINDOW_SIZE:
                # Extract the most recent window
                window_buffer = eeg_buffer[-WINDOW_SIZE:]
                
                # Separate timestamps and data
                timestamps = np.array([sample[0] for sample in window_buffer])
                window_data = np.array([sample[1] for sample in window_buffer])
                
                # Create epoched data structure
                epoched_data = {
                    'data': window_data.copy(),
                    'timestamps': timestamps.copy(),
                    'start_time': timestamps[0],
                    'end_time': timestamps[-1],
                    'duration': timestamps[-1] - timestamps[0],
                    'sample_count': len(timestamps)
                }
                
                # Send to processing queue
                try:
                    eeg_queue.put_nowait(epoched_data)
                except queue.Full:
                    print("⚠️  EEG processing queue full, dropping epoch")
                
                # Remove processed samples (with overlap)
                keep = int(WINDOW_SIZE * OVERLAP)
                eeg_buffer = eeg_buffer[WINDOW_SIZE - keep:]
            
            time.sleep(0.01)  # Small delay to prevent busy waiting
            
        except Exception as e:
            print(f"❌ Error in epoch processing: {e}")
            time.sleep(1)

async def process_eeg_analysis():
    """Main EEG analysis loop"""
    global model, eeg_queue
    
    print("🧠 Starting EEG analysis processing...")
    
    while True:
        try:
            # Get epoched data from queue
            try:
                epoched_data = eeg_queue.get(timeout=1.0)
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
            
            if model is None:
                print("⚠️  Model not loaded, skipping analysis")
                await asyncio.sleep(1)
                continue
            
            # Prepare data for the model (256 samples, 8 channels)
            window_data = epoched_data['data']  # Shape: (256, 8)
            
            # Run dizziness evaluation using the model directly
            # Apply FFT and prepare input for the neural network
            fft_data = np.abs(np.fft.fft(window_data, axis=0)[1:33])  # First 32 frequency bins
            model_input = torch.from_numpy(fft_data.flatten()).float()
            
            with torch.no_grad():
                dizziness_probs = model(model_input).tolist()
            
            # Determine dominant state
            dominant_state = np.argmax(dizziness_probs)
            confidence = dizziness_probs[dominant_state]
            
            # Prepare analysis results
            analysis_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "dizziness_probabilities": {
                    "none": float(dizziness_probs[3]),
                    "low": float(dizziness_probs[2]), 
                    "moderate": float(dizziness_probs[1]),
                    "high": float(dizziness_probs[0])
                },
                "dominant_state": int(dominant_state),
                "state_name": DIZZINESS_LEVELS[dominant_state],
                "confidence": float(confidence),
                "window_duration": epoched_data['duration'],
                "sample_count": epoched_data['sample_count'],
                "channels": CHANNEL_COUNT,
                "service": "dizziness_eeg"
            }
            
            # Broadcast to all connected clients
            await manager.broadcast(json.dumps(analysis_data))
            
        except Exception as e:
            print(f"❌ Error in EEG analysis: {e}")
            await asyncio.sleep(1)

def initialize_eeg_system():
    """Initialize EEG acquisition and processing system"""
    global acquisition_thread, processing_thread
    
    print("🔧 Initializing EEG acquisition system...")
    
    # Load the dizziness detection model
    if not load_dizziness_model():
        return False
    
    # Start epoch processing thread
    processing_thread = threading.Thread(target=epoch_and_analyze, daemon=True)
    processing_thread.start()
    print("✅ EEG processing thread started")
    
    # Try to connect to OpenBCI
    if try_connect_openbci():
        # Real OpenBCI acquisition
        try:
            print("🚀 Starting OpenBCI data stream...")
            acquisition_thread = threading.Thread(
                target=lambda: board.start_stream(stream_callback), 
                daemon=True
            )
            acquisition_thread.start()
            print("✅ OpenBCI acquisition started")
        except Exception as e:
            print(f"❌ Error starting OpenBCI stream: {e}")
            board = None
    
    if board is None:
        # Fallback to simulation
        print("🎭 Starting EEG simulation...")
        acquisition_thread = threading.Thread(target=generate_eeg_simulation, daemon=True)
        acquisition_thread.start()
        print("✅ EEG simulation started")
    
    return True

async def fallback_mode():
    """Simple fallback mode when EEG system fails"""
    counter = 0
    while True:
        counter += 1
        
        # Simple simulation
        simulated_state = counter % 4
        confidence = 0.6 + 0.4 * np.sin(counter * 0.1)
        
        data = {
            "timestamp": asyncio.get_event_loop().time(),
            "dizziness_probabilities": {
                "none": 0.7 if simulated_state == 3 else 0.1,
                "low": 0.7 if simulated_state == 2 else 0.1,
                "moderate": 0.7 if simulated_state == 1 else 0.1,
                "high": 0.7 if simulated_state == 0 else 0.1
            },
            "dominant_state": simulated_state,
            "state_name": DIZZINESS_LEVELS[simulated_state],
            "confidence": confidence,
            "service": "dizziness_eeg",
            "mode": "fallback"
        }
        
        await manager.broadcast(json.dumps(data))
        await asyncio.sleep(2)

@app.on_event("startup")
async def startup_event():
    """Initialize EEG system and start processing"""
    if initialize_eeg_system():
        asyncio.create_task(process_eeg_analysis())
    else:
        print("Failed to initialize EEG system, using fallback mode")
        asyncio.create_task(fallback_mode())

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global board
    if board is not None:
        try:
            board.stop_stream()
            board.disconnect()
        except:
            pass
    print("🔧 Dizziness EEG Service shutdown complete")

if __name__ == "__main__":
    print("🚀 Starting Dizziness EEG Service on port 8002")
    print("🧠 Initializing neural network-based dizziness detection")
    print("📡 EEG acquisition: OpenBCI Cyton or simulation mode")
    uvicorn.run(app, host="127.0.0.1", port=8002)