import asyncio
import websockets
import json

async def connect_to_service(service_name, port):
    uri = f"ws://127.0.0.1:{port}/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[{service_name}] Connected to service on port {port}")
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print(f"[{service_name}] Received: {data}")
                except websockets.exceptions.ConnectionClosed:
                    print(f"[{service_name}] Connection closed")
                    break
                except json.JSONDecodeError:
                    print(f"[{service_name}] Invalid JSON: {message}")
    except Exception as e:
        print(f"[{service_name}] Connection failed: {e}")

async def main():
    services = [
        ("DISTRACTION_CV", 8001),
        ("DIZZINESS_EEG", 8002),
        ("OBJECT_DETECTION", 8003)
    ]
    
    tasks = []
    for service_name, port in services:
        task = asyncio.create_task(connect_to_service(service_name, port))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

if __name__ == "__main__"っっっっっっっd
    asyncio.run(main())
