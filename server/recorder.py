"""
NoMaD recorder + web controller.

  WebSocket  ws://0.0.0.0:8765  — phone (device) and browser (UI) clients
  HTTP       http://0.0.0.0:8080 — web controller UI

Open http://localhost:8080 in your browser after starting this script.

Saved trajectories go to:
  data/traj_NNNN/
    0.jpg, 1.jpg, …
    traj_data.pkl  →  {"position": [[x,y],…], "yaw": [rad,…]}

Coordinate convention:
  position[:,0] = forward  (-ARKit Z)
  position[:,1] = left     (-ARKit X)
  yaw = rotation around ARKit world-Y (CCW positive)
"""

import asyncio
import base64
import io
import json
import math
import pickle
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import websockets
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

WS_PORT   = 8765
HTTP_PORT = 8080
OUTPUT_DIR = Path(__file__).parent.parent / "data"
MIN_FRAMES = 8

# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NoMaD Controller</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #111; color: #ddd;
    font-family: 'SF Mono', 'Fira Mono', monospace;
    font-size: 13px;
    height: 100vh; overflow: hidden;
  }
  #app {
    display: flex; height: 100vh; gap: 10px; padding: 10px;
  }

  /* ── Left: feed panel ── */
  #feed-panel {
    flex: 3; display: flex; flex-direction: column; gap: 6px; min-width: 0;
  }
  #feed-wrap {
    flex: 1; background: #000; border-radius: 8px; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    position: relative;
  }
  #feed {
    max-width: 100%; max-height: 100%; object-fit: contain; display: block;
  }
  #rec-overlay {
    position: absolute; top: 10px; left: 10px;
    background: rgba(0,0,0,.55); border-radius: 5px;
    padding: 4px 8px; font-size: 12px; display: none;
  }
  #rec-overlay.active { display: block; color: #ff5555; }
  #feed-status {
    padding: 4px 8px; font-size: 11px; color: #555;
    background: #1a1a1a; border-radius: 6px; text-align: center;
  }

  /* ── Right: control panel ── */
  #ctrl-panel {
    flex: 2; display: flex; flex-direction: column; gap: 14px;
    background: #1c1c1c; border-radius: 8px; padding: 14px;
    overflow-y: auto;
  }
  h2 { font-size: 14px; color: #eee; letter-spacing: .05em; }
  hr { border: none; border-top: 1px solid #2a2a2a; }

  /* D-pad */
  #dpad {
    display: grid;
    grid-template: 52px 52px 52px / 52px 52px 52px;
    gap: 5px; align-self: center;
  }
  .dp {
    background: #2a2a2a; border: none; border-radius: 7px;
    color: #ccc; font-size: 18px; cursor: pointer;
    transition: background .07s;
    display: flex; align-items: center; justify-content: center;
  }
  .dp:active, .dp.on { background: #4a4a4a; color: #fff; }
  .dp-blank { background: transparent; pointer-events: none; }

  /* Sliders */
  .row { display: flex; align-items: center; gap: 8px; }
  .row label { width: 56px; color: #888; }
  .row input[type=range] { flex: 1; accent-color: #4a9eff; }
  .row .val { width: 64px; text-align: right; color: #ccc; }

  /* Record button */
  #rec-btn {
    padding: 10px; border: none; border-radius: 7px;
    font-size: 13px; font-family: inherit; cursor: pointer;
    background: #1e4d1e; color: #6fcf6f; letter-spacing: .04em;
    transition: background .15s;
  }
  #rec-btn.active { background: #4d1e1e; color: #ff7070; }

  /* Status */
  .stat { display: flex; justify-content: space-between; color: #555; }
  .stat span { color: #bbb; }
  .dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: #444; margin-right: 5px; vertical-align: middle;
  }
  .dot.green { background: #4caf50; }
  .dot.orange { background: #ff9800; }
  .dot.red { background: #f44336; }
</style>
</head>
<body>
<div id="app">

  <!-- Feed -->
  <div id="feed-panel">
    <div id="feed-wrap">
      <img id="feed" src="" alt="">
      <div id="rec-overlay">⏺ REC &nbsp;<span id="frame-ctr">0</span> frames</div>
    </div>
    <div id="feed-status">No feed — start camera on phone</div>
  </div>

  <!-- Controls -->
  <div id="ctrl-panel">
    <h2>NoMaD Controller</h2>
    <hr>

    <!-- D-pad -->
    <div id="dpad">
      <div class="dp-blank"></div>
      <button class="dp" id="dp-fwd"  data-key="w">▲</button>
      <div class="dp-blank"></div>
      <button class="dp" id="dp-left" data-key="a">◀</button>
      <div class="dp-blank"></div>
      <button class="dp" id="dp-right"data-key="d">▶</button>
      <div class="dp-blank"></div>
      <button class="dp" id="dp-back" data-key="s">▼</button>
      <div class="dp-blank"></div>
    </div>
    <div style="text-align:center;color:#444;font-size:11px">WASD or arrow keys</div>

    <hr>

    <!-- Sliders -->
    <div class="row">
      <label>Speed</label>
      <input type="range" id="spd" min="50" max="300" value="150">
      <span class="val" id="spd-v">150 mm/s</span>
    </div>
    <div class="row">
      <label>Turn</label>
      <input type="range" id="trn" min="5" max="40" value="20">
      <span class="val" id="trn-v">20 °/s</span>
    </div>

    <hr>

    <!-- Record -->
    <button id="rec-btn">⏺&nbsp; Record &nbsp;[R]</button>

    <hr>

    <!-- Status -->
    <div class="stat">WebSocket <span id="ws-st"><span class="dot"></span>—</span></div>
    <div class="stat">Hub &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span id="hub-st">—</span></div>
    <div class="stat">Trajectories <span id="traj-n">0</span></div>
    <div class="stat">This traj <span id="traj-f">—</span></div>
  </div>
</div>

<script>
const feed      = document.getElementById('feed');
const feedStatus= document.getElementById('feed-status');
const recOverlay= document.getElementById('rec-overlay');
const frameCtr  = document.getElementById('frame-ctr');
const recBtn    = document.getElementById('rec-btn');
const spdSlider = document.getElementById('spd');
const trnSlider = document.getElementById('trn');
const spdVal    = document.getElementById('spd-v');
const trnVal    = document.getElementById('trn-v');
const wsSt      = document.getElementById('ws-st');
const hubSt     = document.getElementById('hub-st');
const trajN     = document.getElementById('traj-n');
const trajF     = document.getElementById('traj-f');

let ws = null, recording = false, recFrames = 0;
const held = new Set();
let lastCmd = null;

// ── Sliders ───────────────────────────────────────────────────────────────────
spdSlider.oninput = () => spdVal.textContent = spdSlider.value + ' mm/s';
trnSlider.oninput = () => trnVal.textContent = trnSlider.value + ' °/s';

// ── WebSocket ─────────────────────────────────────────────────────────────────
function mkDot(cls) { return `<span class="dot ${cls}"></span>`; }

function connect() {
  ws = new WebSocket(`ws://${location.hostname}:8765`);
  wsSt.innerHTML = mkDot('orange') + 'Connecting…';

  ws.onopen = () => {
    ws.send(JSON.stringify({ type: 'ui_register' }));
    wsSt.innerHTML = mkDot('green') + 'Connected';
  };
  ws.onclose = () => {
    wsSt.innerHTML = mkDot('red') + 'Disconnected';
    setTimeout(connect, 2000);
  };
  ws.onerror = () => wsSt.innerHTML = mkDot('red') + 'Error';

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'frame') {
      feed.src = 'data:image/jpeg;base64,' + msg.jpeg;
      feedStatus.textContent = `Frame ${msg.frame_idx ?? '?'}`;
      if (recording) { recFrames++; frameCtr.textContent = recFrames; }
    } else if (msg.type === 'hub_state') {
      hubSt.textContent = msg.state;
    } else if (msg.type === 'rec_saved') {
      trajN.textContent = msg.traj_count;
      trajF.textContent = msg.frames + ' frames saved';
    }
  };
}
connect();

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

// ── Drive loop ────────────────────────────────────────────────────────────────
function currentCmd() {
  const spd = parseInt(spdSlider.value);
  const trn = parseInt(trnSlider.value);
  const fwd   = held.has('w') || held.has('arrowup');
  const back  = held.has('s') || held.has('arrowdown');
  const left  = held.has('a') || held.has('arrowleft');
  const right = held.has('d') || held.has('arrowright');
  const v = (fwd ? spd : 0) - (back ? spd : 0);
  const w = (right ? trn : 0) - (left ? trn : 0);
  return (v || w) ? `drive:${v}:${w}` : 'stop';
}

setInterval(() => {
  const cmd = currentCmd();
  if (cmd !== lastCmd) { send({ type: 'drive_cmd', command: cmd }); lastCmd = cmd; }
}, 80);

// ── Keyboard ──────────────────────────────────────────────────────────────────
const driveKeys = new Set(['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright']);

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (driveKeys.has(k)) { e.preventDefault(); held.add(k); syncDpad(); }
  if (k === 'r') toggleRec();
});
document.addEventListener('keyup', e => {
  held.delete(e.key.toLowerCase()); syncDpad();
});

// ── D-pad buttons ─────────────────────────────────────────────────────────────
document.querySelectorAll('.dp[data-key]').forEach(btn => {
  const k = btn.dataset.key;
  const add = e => { e.preventDefault(); held.add(k); syncDpad(); };
  const del = e => { e.preventDefault(); held.delete(k); syncDpad(); };
  btn.addEventListener('mousedown', add);
  btn.addEventListener('touchstart', add, { passive: false });
  ['mouseup','mouseleave'].forEach(ev => btn.addEventListener(ev, del));
  btn.addEventListener('touchend', del, { passive: false });
});

function syncDpad() {
  document.getElementById('dp-fwd') .classList.toggle('on', held.has('w') || held.has('arrowup'));
  document.getElementById('dp-back').classList.toggle('on', held.has('s') || held.has('arrowdown'));
  document.getElementById('dp-left').classList.toggle('on', held.has('a') || held.has('arrowleft'));
  document.getElementById('dp-right').classList.toggle('on',held.has('d') || held.has('arrowright'));
}

// ── Recording ─────────────────────────────────────────────────────────────────
function toggleRec() {
  recording = !recording;
  if (recording) {
    recFrames = 0; frameCtr.textContent = '0';
    send({ type: 'record_start' });
    recBtn.textContent = '⏹  Stop recording  [R]';
    recBtn.classList.add('active');
    recOverlay.classList.add('active');
    trajF.textContent = 'recording…';
  } else {
    send({ type: 'record_stop' });
    recBtn.textContent = '⏺  Record  [R]';
    recBtn.classList.remove('active');
    recOverlay.classList.remove('active');
  }
}
recBtn.addEventListener('click', toggleRec);
</script>
</body>
</html>
"""

# ── Coordinate helpers ────────────────────────────────────────────────────────

def arkit_to_2d(position, rotation):
    px, _py, pz = position
    x_fwd  = -pz
    y_left = -px
    qx, qy, qz, qw = rotation
    yaw = math.atan2(2*(qw*qy + qz*qx), 1 - 2*(qy**2 + qz**2))
    return x_fwd, y_left, yaw

# ── Trajectory recorder ───────────────────────────────────────────────────────

class TrajectoryRecorder:
    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.frames:  list[bytes] = []
        self.poses:   list[tuple] = []
        self.traj_idx = len(sorted(OUTPUT_DIR.glob("traj_*")))

    def start(self):
        self.recording = True
        self.frames, self.poses = [], []
        print(f"\n[rec] ● Recording trajectory {self.traj_idx:04d} …")

    def stop(self) -> dict | None:
        if not self.recording:
            return None
        self.recording = False
        return self._save()

    def add(self, jpeg: bytes, position: list, rotation: list):
        if self.recording:
            self.frames.append(jpeg)
            self.poses.append((position, rotation))

    def _save(self) -> dict:
        n = len(self.frames)
        if n < MIN_FRAMES:
            print(f"[rec] ✗ Only {n} frames — discarded (need ≥ {MIN_FRAMES}).")
            return {"ok": False, "frames": n}

        traj_dir = OUTPUT_DIR / f"traj_{self.traj_idx:04d}"
        traj_dir.mkdir(parents=True, exist_ok=True)

        for i, jpeg in enumerate(self.frames):
            Image.open(io.BytesIO(jpeg)).save(traj_dir / f"{i}.jpg")

        positions, yaws = [], []
        for pos, rot in self.poses:
            x, y, yaw = arkit_to_2d(pos, rot)
            positions.append([x, y])
            yaws.append(yaw)

        traj_data = {
            "position": np.array(positions, dtype=np.float32),
            "yaw":      np.array(yaws,      dtype=np.float32),
        }
        with open(traj_dir / "traj_data.pkl", "wb") as f:
            pickle.dump(traj_data, f)

        print(f"[rec] ✓ Saved {n} frames → {traj_dir}")
        self.traj_idx += 1
        self.frames, self.poses = [], []
        return {"ok": True, "frames": n, "traj_count": self.traj_idx}

# ── WebSocket routing ─────────────────────────────────────────────────────────

recorder      = TrajectoryRecorder()
ui_clients:    set = set()
device_clients: set = set()

async def relay(clients: set, msg: dict):
    if not clients:
        return
    text = json.dumps(msg)
    await asyncio.gather(*(c.send(text) for c in clients), return_exceptions=True)

async def handle(websocket):
    client_type = None
    try:
        async for raw in websocket:
            msg  = json.loads(raw)
            kind = msg.get("type", "")

            # Identify on first message
            if client_type is None:
                if kind == "ui_register":
                    client_type = "ui"
                    ui_clients.add(websocket)
                    print(f"[ws] Browser UI connected  ({len(ui_clients)} ui, {len(device_clients)} device)")
                else:
                    client_type = "device"
                    device_clients.add(websocket)
                    print(f"[ws] Device connected  ({len(ui_clients)} ui, {len(device_clients)} device)")

            # ── Device → server / UI ──────────────────────────────────────────
            if client_type == "device":
                if kind == "frame":
                    recorder.add(
                        base64.b64decode(msg["jpeg"]),
                        msg["position"],
                        msg["rotation"],
                    )
                    # Forward frame to all UI clients (strip heavy jpeg for perf? no — keep it)
                    await relay(ui_clients, msg)

                elif kind == "hub_line":
                    print(f"[hub] {msg.get('line', '')}")

                elif kind == "hub_state":
                    await relay(ui_clients, msg)

            # ── UI → server / device ──────────────────────────────────────────
            elif client_type == "ui":
                if kind == "drive_cmd":
                    cmd = msg.get("command", "stop")
                    await relay(device_clients, {"type": "hub_cmd", "command": cmd})

                elif kind == "record_start":
                    recorder.start()

                elif kind == "record_stop":
                    result = recorder.stop()
                    if result:
                        await relay(ui_clients, {"type": "rec_saved", **result})

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ui_clients.discard(websocket)
        device_clients.discard(websocket)
        if client_type:
            print(f"[ws] {client_type} disconnected  ({len(ui_clients)} ui, {len(device_clients)} device)")

# ── HTTP server (serves the UI page) ─────────────────────────────────────────

class UIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def log_message(self, *_):
        pass   # silence request logs

def run_http():
    server = HTTPServer(("0.0.0.0", HTTP_PORT), UIHandler)
    server.serve_forever()

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    import socket
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "unknown"

    threading.Thread(target=run_http, daemon=True).start()

    print(f"\nNoMaD recorder")
    print(f"  Web UI:    http://localhost:{HTTP_PORT}")
    print(f"  WebSocket: ws://0.0.0.0:{WS_PORT}")
    print(f"  Phone URL: ws://{local_ip}:{WS_PORT}")
    print(f"  Output:    {OUTPUT_DIR.resolve()}\n")

    async with websockets.serve(handle, "0.0.0.0", WS_PORT):
        await asyncio.Future()   # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
