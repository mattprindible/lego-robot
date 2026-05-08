#!/usr/bin/env python3
"""
SensorHub — WebSocket server that bridges the iPhone and the Jetson.

The iPhone app connects here and streams JPEG camera frames.
The server sends drive commands back to the robot via the iPhone's BLE bridge.
"""
import asyncio
import json
import socket

import websockets
from websockets.server import WebSocketServerProtocol

PORT = 8765


class SensorHub:
    def __init__(self, port: int = PORT):
        self.port = port
        self.latest_frame: bytes | None = None
        self._frame_event = asyncio.Event()
        self._client: WebSocketServerProtocol | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def next_frame(self) -> bytes:
        """Await the next camera frame. Returns JPEG bytes."""
        self._frame_event.clear()
        await self._frame_event.wait()
        return self.latest_frame  # type: ignore

    async def send_hub_cmd(self, command: str) -> None:
        """Send a drive command to the robot via the iPhone."""
        if self._client is None:
            raise RuntimeError("iPhone not connected")
        await self._client.send(json.dumps({"type": "hub_cmd", "command": command}))

    async def _handle(self, websocket: WebSocketServerProtocol):
        self._loop = asyncio.get_running_loop()
        self._client = websocket
        print(f"  iPhone connected from {websocket.remote_address[0]}")
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    self.latest_frame = message
                    self._frame_event.set()
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._client = None
            print("  iPhone disconnected")

    async def serve_forever(self):
        async with websockets.serve(self._handle, "0.0.0.0", self.port):
            await asyncio.Future()


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
