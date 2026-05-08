# lego-robot

LEGO Mindstorms robot with vision-guided navigation using Berkeley's NoMaD model.

**The stack:** iPhone 13 mini (camera + BLE relay) → Jetson Orin Nano (NoMaD inference on GPU) → LEGO Inventor Hub (Pybricks motor control)

**Model weights + training code:** lives separately in `~/visualnav-transformer` on the Jetson.
Set `VISUALNAV` env var to override the default location.

---

## Directory structure

```
ios/        iPhone app (Xcode) — ARKit camera, WebSocket client, BLE hub bridge
hub/        Pybricks MicroPython — motor control, obstacle safety, telemetry
server/     Python inference server and utilities (runs on Jetson)
  nomad_server.py   Main inference server (explore + navigate modes)
  sensor_hub.py     WebSocket bridge — SensorHub class
  recorder.py       Training data collection with browser UI
  infer_nomad.py    Standalone inference smoke test
config/     robot.yaml
data/       Collected trajectories (traj_NNNN/) — gitignored, local only
```

---

## Launch sequence

```bash
# On Jetson (nomad venv):
source ~/venvs/nomad/bin/activate
cd ~/lego-robot/server
python nomad_server.py          # explore mode
# or:
python nomad_server.py --mode navigate --goal /path/to/goal.jpg
```

Then open the iPhone app and connect to `ws://192.168.0.77:8765`.

## Collecting training data

```bash
cd ~/lego-robot/server
python recorder.py
# then open http://192.168.0.77:8080 in a browser
```

Saves to `data/traj_NNNN/` — JPEG frames + `traj_data.pkl` with ARKit position and yaw.

## Deploy hub code

```bash
cd hub
python deploy.py    # disconnects hub → uploads main.py → reconnects → Ready
```

---

## Architecture

Two layers are running:

```
Layer 2 — Geometry (Python / Jetson)
  NoMaD inference → waypoint → drive:speed:turn_rate
  Runs the control loop. Handles coordinates, distances, headings.
  Never makes semantic judgments.

Layer 1 — Motor control (Hub / Pybricks MicroPython)
  drive.drive(speed, turn_rate) — continuous, runs until superseded
  Handles IMU heading correction, obstacle speed scaling, safety events.
  Never makes navigation decisions.
```

Interfaces:
- Python → Hub: `drive:200:0` / `turn:90` / `stop`
- Hub → Python: `event|safety_stop|148` / `telem|ds:...|obs:...|hdg:...`

A semantic layer (Claude picks exploration targets, decides when goal is reached) is the intended next step — not yet wired in.

---

## visualnav-transformer setup (Jetson)

Clone from https://github.com/robodhruv/visualnav-transformer to `~/visualnav-transformer`. After a fresh clone:

```bash
# 1. Model weights directory (nomad.pth is not in the repo)
mkdir -p ~/visualnav-transformer/deployment/model_weights
# copy nomad.pth here

# 2. Remove a dead import — torchvision isn't installed in the nomad venv
sed -i '/^import torchvision$/d' ~/visualnav-transformer/train/vint_train/models/nomad/nomad_vint.py
```

---

## Hub continuous drive model

The hub runs a continuous drive loop — no command queue.

- `drive:speed:turn_rate` — sets target speed/turn; `drive_updater` re-applies every 100ms
- `turn:deg` — blocking gyro-corrected turn; fires `event|done` when complete
- `stop` / `brake` — immediate stop
- `shutdown` — stops program

`drive_updater` handles:
- Obstacle speed scaling: linear ramp from full speed at 500mm to 0 at 150mm
- Calls `drive.stop()` (not `drive.drive(0,0)`) when scale == 0 — prevents lurching when obstacle clears
- IMU heading correction: holds straight line when `commanded_turn_rate < 5`

After `safety_stop`, the hub zeros `commanded_speed` and sets `drive_active = False`.
The robot will NOT self-resume when the obstacle clears — that's a Python-layer decision.

---

## Hub events

| Event | Meaning |
|-------|---------|
| `event\|ready` | Hub program started |
| `event\|done\|<cmd>` | Command acknowledged |
| `event\|obstacle_near\|<mm>` | Obstacle within 300mm |
| `event\|safety_stop\|<mm>` | Hard stop at 150mm |
| `event\|safety_clear` | Obstacle gone, >= 200mm clear |
| `event\|stall\|spd:...\|tr:...` | Commanded motion but not moving for 400ms |
| `event\|surface_change\|p:\|r:` | Pitch/roll crossed 15° (fires once on entry) |
| `event\|tipping\|p:\|r:` | Pitch/roll crossed 35° |
| `event\|impact\|lat:<mg>` | Lateral spike > 3000mg while stopped |
| `event\|heading_drift\|tr:` | Heading error > 10° for 1s while driving straight |
| `event\|battery_low\|<mv>` | Battery below 7200mV (fires once) |

Note: `impact` can fire spuriously immediately after `safety_stop` deceleration.

---

## Telemetry (200ms interval)

```
telem|ds:0|spd:0|ang:0|tr:0|hdg:0|obs:2000|spd_scale:100|safe:1|p:0|r:0|ax:0|ay:0|az:0|bat:7800
```

| Field | Meaning |
|-------|---------|
| `ds` | Distance travelled (mm, DriveBase wheel odometry) |
| `spd` | Current speed (mm/s) |
| `ang` | DriveBase accumulated heading (unreliable — inflated by gyro corrections) |
| `tr` | Turn rate (deg/s) |
| `hdg` | IMU gyro heading — ground truth for rotation |
| `obs` | Ultrasonic obstacle distance (mm, 2000 = clear) |
| `spd_scale` | Current speed scale % (100 = full, 0 = stopped by obstacle) |
| `safe` | 1 = safe to drive, 0 = safety_stop active |
| `p` / `r` | Pitch / roll (deg) |
| `ax/ay/az` | IMU acceleration (mg) |
| `bat` | Battery voltage (mV) |

Use `hdg` for heading ground truth. `ang` overcounts due to wheel slip.

---

## Camera

- **Current:** ARKit `ARWorldTrackingConfiguration` → main wide camera (67° FOV, 152mm height)
- **Ideal for NoMaD:** ultra-wide (108° FOV, 160mm height) — requires switching iOS to `AVCaptureSession`
- Server center-crops any resolution to square → 96×96 for NoMaD, so the mismatch is tolerated for now

---

## Known good things (hard-won)

- `CameraStreamer` must be `@State` on the `App` struct, not on a `View`
- `start()` called from `CameraStreamer.init()`, not `.onAppear`
- `ping()` must guard against nil `wsTask` — otherwise `withCheckedContinuation` hangs
- Default server URL hardcoded to `ws://192.168.0.77:8765` — UserDefaults clears on reinstall
- Use `hdg` (IMU gyro) for heading, not `ang` (DriveBase odometry overcounts after gyro turns)
- `impact` events fire spuriously ~1s after `safety_stop` — ignore them in that window
