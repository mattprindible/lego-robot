import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Structured event test.

1. Deploys hub code, waits for Ready
2. Drives forward
3. Watches and prints all events for 90 seconds

Physical test sequence while it's running:
  - Place object in front → obstacle_near → safety_stop
  - Remove object        → safety_clear
  - Block wheels         → stall
  - Tip robot backwards  → surface_change / tipping
  - Pick it up and drop  → impact
"""

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
print("=== Driving. Do the test sequence now. ===")
print("  1. Place object in front")
print("  2. Remove object when stopped")
print("  3. Block wheels from the side")
print("  4. Tip robot backwards")
print("  5. Pick it up and drop it")
print()

hub.hub_cmd("drive:200:0")

seen = set()
t0 = time.time()
deadline = time.time() + 90
while time.time() < deadline:
    for line in list(hub.hub_lines):
        if line not in seen:
            seen.add(line)
            if line.startswith("event|") and not line.startswith("event|done"):
                elapsed = int(time.time() - t0)
                print(f"  [{elapsed:3d}s] {line}")
    time.sleep(0.02)

hub.hub_cmd("stop")
print()
print("=== Test complete ===")
