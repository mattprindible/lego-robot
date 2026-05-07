"""
Obstacle test: drive forward, stop on safety_stop, exit on safety_clear.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import start_background
from deploy import deploy
from hub_utils import wait_for_connection
import time

hub = start_background()
wait_for_connection(hub)
deploy(hub)

if hub.hub_state != "Ready":
    print(f"Hub not ready (state={hub.hub_state}), aborting.")
    raise SystemExit

print()
print("=== Driving forward. Place an obstacle in front. ===")
print()

hub.hub_cmd("drive:200:0")

seen = set()
t0 = time.time()
deadline = time.time() + 60
while time.time() < deadline:
    for line in list(hub.hub_lines):
        if line not in seen:
            seen.add(line)
            if line.startswith("event|") and not line.startswith("event|done"):
                elapsed = int(time.time() - t0)
                print(f"  [{elapsed:3d}s] {line}")
                if line.startswith("event|safety_clear"):
                    hub.hub_cmd("stop")
                    print()
                    print("=== Obstacle cleared, stopped. ===")
                    raise SystemExit
    time.sleep(0.02)

hub.hub_cmd("stop")
print()
print("=== Timeout. ===")
