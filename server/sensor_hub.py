#!/usr/bin/env python3
"""
SensorHub — WebSocket server that bridges the iPhone and the Jetson.

The iPhone app connects here and streams:
  - JPEG camera frames  (type: "frame" or raw binary)
  - ARKit pose          (bundled in "frame" messages)
  - Hub BLE state/lines (type: "hub_state" / "hub_line")

The hub accepts commands:
  hub.hub_cmd("drive:200:0")   # fire-and-forget from any thread
  hub.hub_connect()            # tell iPhone to connect to LEGO hub
  hub.hub_disconnect()         # tell iPhone to disconnect
"""
import asyncio
import base64
import json
import socket
import logging
from collections import deque

import websockets
from websockets.server import WebSocketServerProtocol

PORT = 8765


class SensorHub:
    def __init__(self, port: int = PORT, save_frames_to: str | None = None):
        self.port = port
        self.save_frames_to = save_frames_to
        self.latest_frame: bytes | None = None
        self.latest_pose: dict | None = None
        self.latest_intrinsics: dict | None = None  # {fx, fy, cx, cy, img_w, img_h}
        self.frame_count: int = 0
        self.latest_hub_line: str | None = None
        self.hub_lines: deque[str] = deque(maxlen=200)
        self.hub_state: str = "Disconnected"
        self._frame_event = asyncio.Event()
        self._hub_line_event = asyncio.Event()
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

    async def next_hub_line(self) -> str:
        """Await the next line of output from the hub."""
        self._hub_line_event.clear()
        await self._hub_line_event.wait()
        return self.latest_hub_line  # type: ignore

    async def send_hub_cmd(self, command: str) -> None:
        """Async: send a drive command to the robot via the iPhone."""
        if self._client is None:
            raise RuntimeError("iPhone not connected")
        await self._client.send(json.dumps({"type": "hub_cmd", "command": command}))

    def hub_cmd(self, command: str) -> None:
        """Sync fire-and-forget hub command. Safe to call from any thread."""
        if self._loop is None or self._client is None:
            raise RuntimeError("iPhone not connected")
        asyncio.run_coroutine_threadsafe(self.send_hub_cmd(command), self._loop)

    async def _send_json(self, msg: dict) -> None:
        if self._client is None:
            raise RuntimeError("iPhone not connected")
        await self._client.send(json.dumps(msg))

    def hub_connect(self) -> None:
        if self._loop is None or self._client is None:
            raise RuntimeError("iPhone not connected")
        asyncio.run_coroutine_threadsafe(self._send_json({"type": "hub_connect"}), self._loop)

    def hub_disconnect(self) -> None:
        if self._loop is None or self._client is None:
            raise RuntimeError("iPhone not connected")
        asyncio.run_coroutine_threadsafe(self._send_json({"type": "hub_disconnect"}), self._loop)

    def wait_hub_state(self, target: str, timeout: float = 15.0) -> bool:
        """Block until hub_state matches target. Returns True on success."""
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.hub_state == target:
                return True
            time.sleep(0.1)
        return False

    async def _handle(self, websocket: WebSocketServerProtocol):
        self._loop = asyncio.get_running_loop()
        self._client = websocket
        print(f"  iPhone connected from {websocket.remote_address[0]}")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    self._dispatch_json(message)
                elif isinstance(message, bytes):
                    self.latest_frame = message
                    self.frame_count += 1
                    self._frame_event.set()
                    if self.save_frames_to:
                        with open(self.save_frames_to, "wb") as f:
                            f.write(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._client = None
            print("  iPhone disconnected")

    def _dispatch_json(self, raw: str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        if msg.get("type") == "hub_line":
            line = msg.get("line", "")
            self.latest_hub_line = line
            self.hub_lines.append(line)
            self._hub_line_event.set()
            return

        if msg.get("type") == "hub_state":
            self.hub_state = msg.get("state", "Unknown")
            return

        if msg.get("type") == "frame":
            self.latest_frame = base64.b64decode(msg["jpeg"])
            self.latest_pose = {
                "position":  msg.get("position"),
                "rotation":  msg.get("rotation"),
                "tracking":  msg.get("tracking"),
                "timestamp": msg.get("timestamp"),
            }
            if "intrinsics" in msg and "image_size" in msg:
                fx, fy, cx, cy = msg["intrinsics"]
                w, h = msg["image_size"]
                self.latest_intrinsics = {
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy, "img_w": w, "img_h": h
                }
            self.frame_count += 1
            self._frame_event.set()
            if self.save_frames_to:
                with open(self.save_frames_to, "wb") as f:
                    f.write(self.latest_frame)

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
