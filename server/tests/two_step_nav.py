"""
Two-step navigation test.

Locate an orange marker at ~1000mm. Drive 500mm toward it, re-locate,
drive another 500mm. Report where VIO says we ended up.

Usage:
    python tests/two_step_nav.py
"""

import sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import start_background
from deploy import deploy
from hub_utils import wait_for_connection, straight, turn, pos, dist, latest_telem
from tests.unproject_test import locate_marker, unproject_floor, _jpeg_size

CAMERA_HEIGHT_MM = 165
STEP_MM          = 500


def locate_and_unproject(hub) -> tuple[float, float] | None:
    """Locate the orange marker and return (x_mm, z_mm) floor position."""
    frame = hub.latest_frame
    if not frame:
        print("  No frame.")
        return None

    image_w, image_h = _jpeg_size(frame)
    t    = latest_telem(hub)
    intr = hub.latest_intrinsics or {}
    pitch_deg = t.get("p", 0.0) if t else 0.0

    loc = locate_marker(hub, color="orange")
    if not loc:
        print("  Marker not found.")
        return None

    px, py = loc["px"], loc["py"]
    conf   = loc.get("confidence", "?")
    print(f"  Marker: px={px:.3f}  py={py:.3f}  confidence={conf}")

    if py < 0.5:
        print(f"  WARNING: py={py:.3f} — likely a false positive, skipping.")
        return None

    result = unproject_floor(
        px, py, CAMERA_HEIGHT_MM, image_w, image_h, pitch_deg,
        fx=intr.get("fx"), fy=intr.get("fy"),
        cx=intr.get("cx"), cy=intr.get("cy"),
        landscape_w=intr.get("img_w"), landscape_h=intr.get("img_h"),
    )
    if result is None:
        print("  Point above horizon.")
        return None

    x_mm, z_mm = result
    print(f"  Unprojected: z={z_mm:.0f}mm forward  x={x_mm:+.0f}mm lateral  "
          f"(total {math.hypot(x_mm, z_mm):.0f}mm)")
    return x_mm, z_mm


def run(hub=None):
    if hub is None:
        hub = start_background()
        wait_for_connection(hub)
        print("Waiting for first frame...", end=" ", flush=True)
        for _ in range(150):
            if hub.latest_frame:
                break
            time.sleep(0.2)
        print("got it." if hub.latest_frame else "timed out.")
        print("Deploying to hub...")
        deploy(hub)
        print("Hub ready.")

    p_start = pos(hub)
    if p_start is None:
        print("No VIO pose — is ARKit tracking?")
        return

    print(f"\nStart: VIO ({p_start[0]:+.3f}, {p_start[1]:+.3f}, {p_start[2]:+.3f})\n")

    total_driven = 0

    for step in range(1, 3):
        print(f"── Step {step}/2 ──")

        result = locate_and_unproject(hub)
        if result is None:
            print("Skipping drive — no valid marker fix.")
        else:
            x_mm, z_mm = result
            # Turn to center the marker laterally
            if abs(x_mm) > 20 and z_mm > 0:
                turn_deg = round(math.degrees(math.atan2(x_mm, z_mm)))
                turn_deg = max(-30, min(30, turn_deg))
                if abs(turn_deg) >= 3:
                    print(f"  Turning {turn_deg:+d}° to center marker...")
                    turn(hub, turn_deg, timeout=8)
                    time.sleep(0.3)

        print(f"  Driving {STEP_MM}mm...", end=" ", flush=True)
        moved = straight(hub, STEP_MM, timeout=20)
        total_driven += moved
        print(f"moved {moved:.0f}mm (VIO)")
        print()
        time.sleep(0.5)

    p_end = pos(hub)
    total_vio = dist(p_start, p_end) * 1000 if p_end else None

    print("── Result ──")
    print(f"  Steps driven:       2 × {STEP_MM}mm = {2*STEP_MM}mm commanded")
    print(f"  VIO per-step total: {total_driven:.0f}mm")
    if total_vio is not None:
        print(f"  VIO displacement:   {total_vio:.0f}mm  (straight-line from start)")
    print(f"  Target was:         ~1000mm")
    if total_vio is not None:
        err = total_vio - 1000
        print(f"  Error vs target:    {err:+.0f}mm  ({100*err/1000:+.1f}%)")

    print(f"\nEnd: VIO ({p_end[0]:+.3f}, {p_end[1]:+.3f}, {p_end[2]:+.3f})" if p_end else "")


if __name__ == "__main__":
    run()
