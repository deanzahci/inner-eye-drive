import asyncio
import websockets
import json
import threading

# Global variable to control the running state
running = False
loop = None

async def connect_to_service(service_name, port):
    global running
    uri = f"ws://127.0.0.1:{port}/ws"
    print(f"[{service_name}] Attempting to connect to {uri}")
    
    try:
        async with websockets.connect(uri, ping_timeout=20, ping_interval=20) as websocket:
            print(f"[{service_name}] Successfully connected to service on port {port}")
            
            # Send initial message to establish connection (needed for some services)
            await websocket.send("hello")
            
            while running:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    print(f"[{service_name}] Received: {data}")
                except asyncio.TimeoutError:
                    # Timeout allows us to check the running flag periodically
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print(f"[{service_name}] Connection closed")
                    break
                except json.JSONDecodeError:
                    print(f"[{service_name}] Invalid JSON: {message}")
    except ConnectionRefusedError:
        print(f"[{service_name}] Connection refused - service not running on port {port}")
    except Exception as e:
        print(f"[{service_name}] Connection failed: {e}")
    
    print(f"[{service_name}] Service connection ended")

async def main():
    global running
    services = [
        ("DISTRACTION_CV", 8001),
        ("DIZZINESS_EEG", 8002),
        ("OBJECT_DETECTION", 8003)
    ]
    
    tasks = []
    for service_name, port in services:
        task = asyncio.create_task(connect_to_service(service_name, port))
        tasks.append(task)
    
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Client tasks cancelled")

def start():
    global running, loop
    running = True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Client interrupted")
    finally:
        loop.close()

def stop():
    global running, loop
    running = False
    if loop and not loop.is_closed():
        # Schedule the loop to stop
        loop.call_soon_threadsafe(loop.stop)
    print("Client stopped")