# NeuroDrive

**UC Berkeley AI Hackathon 2025 Project**

A real-time driver monitoring system that integrates computer vision and EEG analysis to detect distraction, drowsiness, and potential hazards for enhanced driving safety.

## 🏆 Project Vision

We win these - Building the future of intelligent driver assistance systems.

## 🏗️ Architecture

### Microservices Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Distraction CV │    │  Dizziness EEG  │    │ Object Detection│
│    Port 8001    │    │    Port 8002    │    │    Port 8003    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │    Main GUI     │
                    │   (Tkinter)     │
                    └─────────────────┘
```

### Core Services

1. **Distraction CV Service** (`services/distraction_cv.py`)
   - Real-time eye tracking and attention detection
   - Head pose estimation
   - WebSocket streaming on port 8001

2. **Dizziness EEG Service** (`services/dizziness_eeg.py`)
   - EEG signal processing for drowsiness detection
   - Brain wave analysis (Alpha, Beta, Theta)
   - WebSocket streaming on port 8002

3. **Object Detection Service** (`services/object_detection_cv.py`)
   - Real-time road hazard detection
   - Vehicle and pedestrian identification
   - WebSocket streaming on port 8003

4. **Main Integration** (`main.py`)
   - Tkinter-based GUI for unified monitoring
   - Real-time data aggregation from all services
   - Central control and visualization

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd inner-eye-drive
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the System

1. **Start all services** (in separate terminals):
   ```bash
   # Terminal 1 - Distraction CV
   python3 services/distraction_cv.py
   
   # Terminal 2 - Dizziness EEG  
   python3 services/dizziness_eeg.py
   
   # Terminal 3 - Object Detection
   python3 services/object_detection_cv.py
   ```

2. **Launch the main application**:
   ```bash
   # Terminal 4 - Main GUI
   python3 main.py
   ```

3. **Test with multi-client** (optional):
   ```bash
   # View all service outputs simultaneously
   python3 multi_client.py
   ```

## 🔧 API Endpoints

### Service Health Checks
- **Distraction CV**: `GET http://127.0.0.1:8001/`
- **Dizziness EEG**: `GET http://127.0.0.1:8002/`
- **Object Detection**: `GET http://127.0.0.1:8003/`

### WebSocket Endpoints
- **Distraction CV**: `ws://127.0.0.1:8001/ws`
- **Dizziness EEG**: `ws://127.0.0.1:8002/ws`
- **Object Detection**: `ws://127.0.0.1:8003/ws`

## 📁 Project Structure

```
inner-eye-drive/
├── main.py                     # Main GUI application
├── multi_client.py            # Multi-service test client
├── client.py                  # Single service test client
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── services/
│   ├── distraction_cv.py     # Eye tracking & attention detection
│   ├── dizziness_eeg.py      # EEG drowsiness detection
│   └── object_detection_cv.py # Road hazard detection
└── models/                    # ML models (future implementation)
```

## 🛠️ Development Status

### ✅ Completed
- [x] Microservices architecture setup
- [x] FastAPI + WebSocket infrastructure
- [x] Real-time communication between services
- [x] Basic Tkinter GUI framework
- [x] Multi-service client testing

### 🚧 In Progress
- [ ] Computer vision algorithms implementation
- [ ] EEG signal processing integration
- [ ] Machine learning model integration
- [ ] Advanced GUI with real-time visualizations

### 📋 Planned Features
- [ ] Real-time driver alerting system
- [ ] Data logging and analytics
- [ ] Mobile app integration
- [ ] Cloud-based monitoring dashboard

## 🤝 Contributing

This project was developed during the UC Berkeley AI Hackathon 2025. 

## 📄 License

[Add your license information here]

---

**Built with ❤️ at UC Berkeley AI Hackathon 2025**