import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Square test: drive 4 sides, return to origin, measure drift.

Reports VIO displacement from start at each corner and on return,
giving a picture of how error accumulates around the loop.

Usage:
    from server import start_background
    hub = start_background()
    # wait for connection + deploy...

    from square_test import run
    run(hub)                  # default 400mm sides
    run(hub, side_mm=250)     # smaller square
"""

import time
from hub_utils import pos, dist, straight, turn


def run(hub, side_mm=400):
    print(f"\n=== Square test: {side_mm}mm sides ===\n")

    deadline = time.time() + 10
    while pos(hub) is None and time.time() < deadline:
        time.sleep(0.2)

    start = pos(hub)
    if start is None:
        print("No ARKit pose available.")
        return None

    print(f"Start:    ({start[0]:+.3f}, {start[1]:+.3f}, {start[2]:+.3f})")
    print()

    corners = []
    for i in range(4):
        print(f"Side {i+1}: straight {side_mm}mm...", end=" ", flush=True)
        straight(hub, side_mm, timeout=30)
        p = pos(hub)
        drift = dist(start, p) * 1000
        print(f"pos ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  "
              f"dist from start: {drift:.0f}mm")
        corners.append(p)

        if i < 3:
            print(f"         turn 90°...", end=" ", flush=True)
            turn(hub, 90)
            p = pos(hub)
            print(f"pos ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    final = pos(hub)
    drift_mm = dist(start, final) * 1000

    print()
    print(f"Start:    ({start[0]:+.3f}, {start[1]:+.3f}, {start[2]:+.3f})")
    print(f"End:      ({final[0]:+.3f}, {final[1]:+.3f}, {final[2]:+.3f})")
    print(f"Drift:    {drift_mm:.1f}mm  ({100*drift_mm/(4*side_mm):.1f}% of perimeter)")

    return {"start": start, "end": final, "drift_mm": drift_mm, "corners": corners}
