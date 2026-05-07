import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
VIO validation: drive a known distance, compare ARKit pose vs hub odometry.

Usage:
    from server import start_background
    hub = start_background()
    # wait for connection + deploy...

    from vio_test import run
    run(hub)
    run(hub, distance_mm=1000)
"""

import time
from hub_utils import pos, dist, latest_telem, straight


def run(hub, distance_mm=500):
    print(f"\n=== VIO test: straight {distance_mm}mm ===\n")

    time.sleep(0.5)
    pose_before = hub.latest_pose
    t0 = latest_telem(hub)
    p0 = pos(hub)

    if pose_before is None or p0 is None:
        print("No ARKit pose available.")
        return None

    tracking_before = pose_before["tracking"]
    ds_before = t0.get("ds", 0) if t0 else 0

    print(f"Start position:  x={p0[0]:.3f}  y={p0[1]:.3f}  z={p0[2]:.3f}")
    print(f"Tracking:        {tracking_before}")
    print(f"Hub odometry:    ds={ds_before:.0f} mm")
    print()

    print(f"Driving {distance_mm}mm ...")
    actual_mm = straight(hub, distance_mm)

    t1 = latest_telem(hub)
    pose_after = hub.latest_pose
    p1 = pos(hub)

    tracking_after = pose_after["tracking"] if pose_after else "unknown"
    ds_after = t1.get("ds", 0) if t1 else 0
    hub_mm = ds_after - ds_before
    arkit_mm = dist(p0, p1) * 1000 if p1 else actual_mm

    print(f"End position:    x={p1[0]:.3f}  y={p1[1]:.3f}  z={p1[2]:.3f}")
    print(f"Tracking:        {tracking_after}")
    print()
    print(f"Expected:        {distance_mm} mm")
    print(f"Hub odometry:    {hub_mm:.0f} mm")
    print(f"ARKit measured:  {arkit_mm:.1f} mm")
    err = abs(arkit_mm - distance_mm)
    print(f"ARKit error:     {err:.1f} mm  ({100*err/distance_mm:.1f}%)")

    events = [l for l in hub.hub_lines if l.startswith("event")]
    print(f"\nHub events:      {events}")

    return {
        "arkit_mm": arkit_mm,
        "hub_mm": hub_mm,
        "expected_mm": distance_mm,
        "tracking_before": tracking_before,
        "tracking_after": tracking_after,
    }
