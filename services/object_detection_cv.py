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
counter = 0

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

async def send_counter():
    global counter
    while True:
        counter += 1
        data = {"counter": counter}
        await manager.broadcast(json.dumps(data))
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(send_counter())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)