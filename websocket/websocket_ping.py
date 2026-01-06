from fastapi import WebSocket
import asyncio

async def websocket_keepalive(websocket: WebSocket, interval: int = 20):
    while True:
        try:
            await websocket.send_ping()

            await asyncio.sleep(interval)
        except Exception as e:
            print(f"WebSocket keepalive failed: {e}")
            break