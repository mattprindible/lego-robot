"""
Spatial map: records (position, description, image) entries as the robot moves.
Persists to map.json. All logic runs in Python; ARKit pose comes from the hub server.

Usage:
    from server import start_background
    from map import SpatialMap

    hub = start_background()
    m = SpatialMap(hub)
    m.start()          # background thread: watches pose, records on movement
    m.describe_now()   # force a vision snapshot right now
    m.entries          # list of recorded entries
    m.save()           # write map.json
"""

import json
import math
import time
import threading
from pathlib import Path
from datetime import datetime


MAP_FILE = Path(__file__).parent / "map.json"
MOVE_THRESHOLD_M = 0.15      # record when robot moves >15cm from last sample
TURN_THRESHOLD_DEG = 20      # or turns >20°
MIN_INTERVAL_S = 2.0         # but no faster than every 2s
GOOD_TRACKING = {"normal"}   # only record when ARKit confidence is high


# ── quaternion helpers ────────────────────────────────────────────────────────

def quat_to_yaw(q):
    """Extract yaw (heading) in degrees from a quaternion [x, y, z, w]."""
    x, y, z, w = q
    siny = 2 * (w * y + z * x)
    cosy = 1 - 2 * (y * y + z * z)
    return math.degrees(math.atan2(siny, cosy))


def pos_dist(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def angle_diff(a, b):
    d = (b - a + 180) % 360 - 180
    return abs(d)


# ── SpatialMap ────────────────────────────────────────────────────────────────

class SpatialMap:
    def __init__(self, hub, map_file=MAP_FILE):
        self.hub = hub
        self.map_file = Path(map_file)
        self.entries: list[dict] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_pos = None
        self._last_yaw = None
        self._last_record_time = 0

        if self.map_file.exists():
            self._load()

    # MARK: - Public API

    def start(self):
        """Start background thread that watches pose and records on movement."""
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="spatial-map")
        self._thread.start()
        print(f"SpatialMap started ({len(self.entries)} existing entries)")

    def stop(self):
        self._running = False

    def describe_now(self, note: str = "") -> dict | None:
        """Force a vision snapshot at the current position. Returns the entry."""
        pose = self.hub.latest_pose
        frame = self.hub.latest_frame
        if pose is None or frame is None:
            print("No pose/frame available yet.")
            return None
        if pose["tracking"] not in GOOD_TRACKING:
            print(f"Skipping — tracking is '{pose['tracking']}'")
            return None
        return self._record(pose, frame, note=note)

    def save(self):
        with self._lock:
            data = [self._serializable(e) for e in self.entries]
        self.map_file.write_text(json.dumps(data, indent=2))
        print(f"Saved {len(self.entries)} entries to {self.map_file}")

    def summary(self):
        with self._lock:
            n = len(self.entries)
        if n == 0:
            print("Map is empty.")
            return
        print(f"{n} entries:")
        for i, e in enumerate(self.entries):
            p = e["position"]
            print(f"  [{i}] ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})  "
                  f"tracking={e['tracking']}  "
                  f"{e['timestamp'][:19]}  "
                  f"{e.get('description', '')[:60]}")

    # MARK: - Background watch loop

    def _watch_loop(self):
        while self._running:
            time.sleep(0.2)
            pose = self.hub.latest_pose
            frame = self.hub.latest_frame
            if pose is None or frame is None:
                continue
            if pose["tracking"] not in GOOD_TRACKING:
                continue
            if time.time() - self._last_record_time < MIN_INTERVAL_S:
                continue

            p = pose["position"]
            yaw = quat_to_yaw(pose["rotation"])

            moved = (self._last_pos is None or
                     pos_dist(self._last_pos, p) > MOVE_THRESHOLD_M)
            turned = (self._last_yaw is None or
                      angle_diff(self._last_yaw, yaw) > TURN_THRESHOLD_DEG)

            if moved or turned:
                self._record(pose, frame)

    # MARK: - Recording

    def _record(self, pose, frame, note: str = "") -> dict:
        from vision import describe as vision_describe
        pos = pose["position"]
        print(f"  Recording at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) — describing...", end=" ", flush=True)

        try:
            description = vision_describe("Describe what you see briefly (1-2 sentences).", hub=self.hub)
        except Exception as e:
            description = f"[vision error: {e}]"

        entry = {
            "position": pos,
            "rotation": pose["rotation"],
            "tracking": pose["tracking"],
            "timestamp": datetime.utcnow().isoformat(),
            "description": description,
            "note": note,
            "_frame_bytes": frame,  # kept in memory only, not serialized
        }

        print(description[:80])

        with self._lock:
            self.entries.append(entry)
            self._last_pos = pos
            self._last_yaw = quat_to_yaw(pose["rotation"])
            self._last_record_time = time.time()

        self.save()
        return entry

    # MARK: - Persistence

    def _serializable(self, entry: dict) -> dict:
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    def _load(self):
        try:
            data = json.loads(self.map_file.read_text())
            self.entries = data
            print(f"Loaded {len(self.entries)} entries from {self.map_file}")
        except Exception as e:
            print(f"Could not load map: {e}")
