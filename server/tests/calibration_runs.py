import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Calibration runs: three targeted tests to isolate error sources.

  A. out_and_back  — pure straight, no turns. How accurate is linear travel?
  B. heading_loop  — 4 × 90° turns, no driving. Do they compose back to start?
  C. small_square  — 250mm sides. Combined accuracy at smaller scale.

Usage:
    from server import start_background
    hub = start_background()
    # wait for connection + deploy...

    from calibration_runs import run_all
    run_all(hub)

    # or individually:
    from calibration_runs import out_and_back, heading_loop, small_square
"""

import time
from hub_utils import pos, dist, latest_telem, straight, turn, angle_diff


# ── Run A: Out-and-back ───────────────────────────────────────────────────────

def out_and_back(hub, distance_mm=400):
    """
    Drive forward distance_mm, then backward distance_mm.
    Net displacement should be ~0. Isolates linear accuracy from turns.
    """
    print(f"\n── A: Out-and-back ({distance_mm}mm) ──\n")

    time.sleep(0.5)
    p0 = pos(hub)
    t0 = latest_telem(hub)

    if p0 is None:
        print("No ARKit pose. Skipping.")
        return None

    ds0 = t0.get("ds", 0) if t0 else 0
    print(f"  Start:      VIO ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})")

    print(f"  → {distance_mm}mm...", end=" ", flush=True)
    straight(hub, distance_mm, timeout=30)
    p1 = pos(hub)
    t1 = latest_telem(hub)
    ds1 = t1.get("ds", 0) if t1 else 0
    vio_fwd = dist(p0, p1) * 1000
    hub_fwd = abs(ds1 - ds0)
    print(f"VIO {vio_fwd:.1f}mm   hub ds {hub_fwd:.0f}mm   (VIO/cmd = {100*vio_fwd/distance_mm:.1f}%)")

    print(f"  ← {distance_mm}mm...", end=" ", flush=True)
    straight(hub, -distance_mm, timeout=30)
    p2 = pos(hub)
    t2 = latest_telem(hub)
    ds2 = t2.get("ds", 0) if t2 else 0
    vio_back = dist(p1, p2) * 1000
    hub_back = abs(ds2 - ds1)
    print(f"VIO {vio_back:.1f}mm   hub ds {hub_back:.0f}mm   (VIO/cmd = {100*vio_back/distance_mm:.1f}%)")

    net_mm = dist(p0, p2) * 1000
    print(f"\n  Net displacement: {net_mm:.1f}mm  (should be ~0)")
    print(f"  VIO fwd vs back:  {vio_fwd:.1f}mm vs {vio_back:.1f}mm  "
          f"(asymmetry = {abs(vio_fwd - vio_back):.1f}mm)")

    return {"vio_fwd": vio_fwd, "vio_back": vio_back, "net_mm": net_mm,
            "hub_fwd": hub_fwd, "hub_back": hub_back}


# ── Run B: Heading loop ───────────────────────────────────────────────────────

def heading_loop(hub, n_turns=4, angle=90):
    """
    Execute (n_turns × angle)° of total rotation with no driving.
    Reports IMU heading after each turn.
    Final heading should match start (net rotation = n_turns × angle, mod 360).
    """
    total = n_turns * angle
    print(f"\n── B: Heading loop ({n_turns} × {angle:+d}° = {total}° total) ──\n")

    time.sleep(0.5)
    t0 = latest_telem(hub)
    if t0 is None:
        print("No telemetry. Skipping.")
        return None

    hdg0 = t0.get("hdg")
    if hdg0 is None:
        print("No IMU heading in telemetry. Skipping.")
        return None

    print(f"  Start heading: {hdg0:.2f}°")
    headings = [hdg0]

    for i in range(n_turns):
        print(f"  Turn {i+1}: {angle:+d}°...", end=" ", flush=True)
        turn(hub, angle, timeout=15)
        t = latest_telem(hub)
        hdg = t.get("hdg") if t else None
        if hdg is None:
            print("no heading")
            continue
        delta      = angle_diff(headings[-1], hdg)
        cumulative = angle_diff(hdg0, hdg)
        print(f"hdg {hdg:.2f}°   Δ {delta:+.2f}°   cumulative {cumulative:+.2f}°")
        headings.append(hdg)

    final_hdg = headings[-1]
    expected  = total % 360
    if expected > 180:
        expected -= 360
    net_err = angle_diff(hdg0, final_hdg) - expected

    print(f"\n  Start heading:  {hdg0:.2f}°")
    print(f"  Final heading:  {final_hdg:.2f}°")
    print(f"  Heading error:  {net_err:+.2f}°")

    return {"headings": headings, "net_error": net_err}


# ── Run C: Small square ───────────────────────────────────────────────────────

def small_square(hub, side_mm=250):
    """
    Drive a square with shorter sides, logging VIO and IMU at each corner.
    """
    print(f"\n── C: Small square ({side_mm}mm sides) ──\n")

    time.sleep(0.5)
    p0 = pos(hub)
    t0 = latest_telem(hub)

    if p0 is None:
        print("No ARKit pose. Skipping.")
        return None

    hdg0 = t0.get("hdg") if t0 else None
    print(f"  Start: VIO ({p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f})  IMU hdg {hdg0:.2f}°")

    corners = []
    for i in range(4):
        print(f"  Side {i+1}: {side_mm}mm...", end=" ", flush=True)
        straight(hub, side_mm, timeout=20)
        p = pos(hub)
        t = latest_telem(hub)
        hdg = t.get("hdg") if t else None
        leg_dist   = dist(corners[-1][0] if corners else p0, p) * 1000
        from_start = dist(p0, p) * 1000
        print(f"leg {leg_dist:.0f}mm   dist from start {from_start:.0f}mm   hdg {hdg:.2f}°")
        corners.append((p, hdg))

        if i < 3:
            print(f"         turn 90°...", end=" ", flush=True)
            turn(hub, 90)
            p = pos(hub)
            t = latest_telem(hub)
            hdg = t.get("hdg") if t else None
            heading_change = angle_diff(corners[-1][1], hdg)
            print(f"Δhdg {heading_change:+.1f}°   "
                  f"cumulative from start {angle_diff(hdg0, hdg):+.1f}°")

    final = pos(hub)
    t_final = latest_telem(hub)
    hdg_final = t_final.get("hdg") if t_final else None
    drift_mm  = dist(p0, final) * 1000
    hdg_err   = angle_diff(hdg0, hdg_final) if hdg0 and hdg_final else None

    print(f"\n  Drift:          {drift_mm:.1f}mm")
    if hdg_err is not None:
        print(f"  Heading error:  {hdg_err:+.1f}°  (should be ~0°)")
    print(f"  (% of perim):   {100*drift_mm/(4*side_mm):.1f}%")

    return {"drift_mm": drift_mm, "heading_err": hdg_err, "corners": corners}


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all(hub):
    print("\n══════════════════════════════════════════")
    print("  Calibration runs A + B + C")
    print("══════════════════════════════════════════")

    a = out_and_back(hub)
    time.sleep(1)
    b = heading_loop(hub)
    time.sleep(1)
    c = small_square(hub)

    print("\n══════════════════════════════════════════")
    print("  Summary")
    print("══════════════════════════════════════════")
    if a:
        print(f"  A. Linear net displacement:  {a['net_mm']:.1f}mm")
        print(f"     VIO fwd/back symmetry:     {abs(a['vio_fwd']-a['vio_back']):.1f}mm asymmetry")
    if b:
        print(f"  B. Heading loop error:        {b['net_error']:+.2f}° after 360°")
    if c:
        print(f"  C. Small square drift:        {c['drift_mm']:.1f}mm  ({100*c['drift_mm']/1000:.1f}% of perim)")
        if c["heading_err"] is not None:
            print(f"     Heading error after square: {c['heading_err']:+.1f}°")

    return {"A": a, "B": b, "C": c}
