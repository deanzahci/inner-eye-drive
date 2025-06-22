#!/usr/bin/env python3
"""
Test script for Drowsiness Detection API Integration
"""

import asyncio
import websockets
import json
import requests
import time

async def test_websocket():
    """Test WebSocket connection to drowsiness detection service"""
    uri = "ws://127.0.0.1:8001/ws"
    
    print("🔌 Testing WebSocket connection...")
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Drowsiness Detection WebSocket")
            
            # Send hello message
            await websocket.send("hello")
            
            # Listen for a few messages
            for i in range(10):
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    print(f"📊 Detection Data {i+1}:")
                    print(f"   Face Detected: {data.get('face_detected', 'N/A')}")
                    print(f"   Drowsy: {data.get('drowsy', 'N/A')}")
                    print(f"   Alert Level: {data.get('alert_level', 'N/A')}")
                    print(f"   Avg EAR: {data.get('avg_ear', 0):.3f}")
                    print(f"   Total Blinks: {data.get('total_blinks', 'N/A')}")
                    print()
                    
                except asyncio.TimeoutError:
                    print("⏰ Timeout waiting for message")
                    break
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON: {message}")
                    
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")

def test_http_endpoints():
    """Test HTTP endpoints"""
    base_url = "http://127.0.0.1:8001"
    
    print("🌐 Testing HTTP endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health endpoint: {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test status endpoint
    try:
        response = requests.get(f"{base_url}/status")
        print(f"✅ Status endpoint: {response.json()}")
    except Exception as e:
        print(f"❌ Status endpoint failed: {e}")

def main():
    print("🧪 Drowsiness Detection API Test")
    print("=" * 50)
    print("Make sure the service is running: python services/distraction_cv.py")
    print()
    
    # Test HTTP endpoints first
    test_http_endpoints()
    
    print("\n" + "=" * 50)
    
    # Test WebSocket
    asyncio.run(test_websocket())
    
    print("🏁 Test complete!")

if __name__ == "__main__":
    main()
