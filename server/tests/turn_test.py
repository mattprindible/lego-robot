import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Turn accuracy test: compare commanded turn vs three independent heading sources.

  DriveBase ang  — wheel-encoder odometry (telem: ang)
  IMU hdg        — hub gyro, independent of wheels (telem: hdg)
  ARKit yaw      — iPhone camera + IMU, independent of both

Usage:
    from server import start_background
    hub = start_background()
    # wait for connection + deploy...

    from turn_test import run
    run(hub)                          # default test sequence
    run(hub, turns=[90, 90, 90])      # repeat the same turn
    run(hub, turns=[45, 90, 135])     # ramp up
"""

import time
import math
from hub_utils import turn, fresh_telem, arkit_yaw, angle_diff


def run(hub, turns=None):
    """
    Issue each turn in `turns`, measure actual heading change from all three
    sources, print a comparison table.

    Default sequence: +45  +90  +135  −45  −90  −135
    """
    if turns is None:
        turns = [45, 90, 135, -45, -90, -135]

    print("\n=== Turn accuracy test ===")
    print(f"  {'cmd':>6}  {'DriveBase':>10}  {'IMU hdg':>10}  {'ARKit':>10}  "
          f"{'DB err':>8}  {'IMU err':>8}  {'ARKit err':>8}")
    print("  " + "─" * 76)

    results = []

    for commanded in turns:
        time.sleep(0.4)

        t0 = fresh_telem(hub)
        y0 = arkit_yaw(hub)

        if t0 is None:
            print(f"  {commanded:>+5}°  — no telemetry, skipping")
            continue

        ang0 = t0.get("ang", 0.0)
        hdg0 = t0.get("hdg")

        turn(hub, commanded, timeout=15)

        time.sleep(0.4)

        t1 = fresh_telem(hub)
        y1 = arkit_yaw(hub)

        if t1 is None:
            print(f"  {commanded:>+5}°  — no telemetry after turn")
            continue

        ang1 = t1.get("ang", 0.0)
        hdg1 = t1.get("hdg")

        db_actual  = ang1 - ang0
        imu_actual = angle_diff(hdg0, hdg1) if hdg0 is not None and hdg1 is not None else None
        ark_actual = angle_diff(y0, y1)     if y0  is not None and y1  is not None else None

        db_err  = db_actual - commanded
        imu_err = (imu_actual - commanded) if imu_actual is not None else None
        ark_err = (ark_actual - commanded) if ark_actual is not None else None

        def _fmt(v): return f"{v:>+8.1f}°" if v is not None else "      N/A"

        print(f"  {commanded:>+5}°"
              f"  {db_actual:>+10.1f}°"
              f"  {_fmt(imu_actual)}"
              f"  {_fmt(ark_actual)}"
              f"  {db_err:>+7.1f}°"
              f"  {_fmt(imu_err)}"
              f"  {_fmt(ark_err)}")

        results.append({
            "commanded": commanded,
            "drivebase": db_actual,
            "imu":       imu_actual,
            "arkit":     ark_actual,
            "db_err":    db_err,
            "imu_err":   imu_err,
            "arkit_err": ark_err,
        })

    if len(results) >= 2:
        print("\n── Summary ──")
        for key, label in [("db_err", "DriveBase"), ("imu_err", "IMU hdg "), ("arkit_err", "ARKit   ")]:
            vals = [r[key] for r in results if r[key] is not None]
            if vals:
                mean = sum(vals) / len(vals)
                rms  = math.sqrt(sum(v**2 for v in vals) / len(vals))
                print(f"  {label}:  mean err = {mean:>+6.1f}°   RMS = {rms:>5.1f}°")

    return results
