import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
D: Floor unprojection calibration.

Tests how accurately a 2D image pixel maps to a real-world floor position
using the fixed ground-plane assumption (rigid chassis, known camera height).

The key question: at what distance does the prediction error become intolerable?
Run geometry_only first to tune CAMERA_HEIGHT_MM without moving the robot, then
drive_to_marker to validate the full end-to-end path.

SETUP: Place a bright marker (tape X, coloured card) at a measured distance on
flat floor, directly in front of the robot. The marker should be clearly visible
in the camera frame.

Usage:
    from server import start_background
    hub = start_background()
    # wait_for_connection(hub), deploy(hub)...

    from tests.unproject_test import geometry_only, drive_to_marker, run_all
    geometry_only(hub, [500, 1000, 1500, 2000])
    drive_to_marker(hub, 1000)
    run_all(hub)
"""

import io
import json
import math
import struct
import time
import base64
from pathlib import Path

import anthropic
from hub_utils import straight, pos, dist, latest_telem

_key = Path(__file__).resolve().parent.parent / "key.txt"
_client = anthropic.Anthropic(api_key=_key.read_text().strip())

# ── Config ────────────────────────────────────────────────────────────────────

HFOV_DEG          = 70      # iPhone horizontal FOV (matches navigate.py)
CAMERA_HEIGHT_MM  = 165     # wide-angle (bottom) lens, measured 6.5" from floor

MARKER_PROMPT = """\
There is a single {color} tape marker on the FLOOR in front of the camera. \
The camera is mounted low on a robot, so the floor occupies the bottom half \
of the image. The marker will appear somewhere in the lower portion of the frame \
(py > 0.5). Ignore any {color} objects on walls, furniture, or in the background.

Find the floor marker and return its centre position as normalised image coordinates.

px: 0.0 = left edge, 1.0 = right edge
py: 0.0 = top edge,  1.0 = bottom edge (floor is here)

Respond with ONLY valid JSON (no markdown):
{{"px": 0.XX, "py": 0.XX, "confidence": "high|medium|low"}}"""


# ── Math ──────────────────────────────────────────────────────────────────────

def unproject_floor(
    px_norm: float,
    py_norm: float,
    camera_height_mm: float,
    image_w: int,
    image_h: int,
    pitch_deg: float = 0.0,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    landscape_w: float | None = None,
    landscape_h: float | None = None,
) -> tuple[float, float] | None:
    """
    Map a normalised image point to (x_mm, z_mm) on the floor plane.

    The frame arrives portrait-rotated (.oriented(.right)). ARKit intrinsics are
    in landscape/native sensor coords. We handle the axis swap here.

    Prefer passing fx/fy/cx/cy/landscape_w/landscape_h from hub.latest_intrinsics.
    Falls back to HFOV estimate if intrinsics are unavailable.

    x_mm = right/left offset from robot centre line (positive = right).
    z_mm = forward distance from camera.
    Returns None if the point is above the horizon (no floor intersection).
    """
    if fx is not None:
        # ARKit intrinsics are in landscape coords (landscape_w × landscape_h).
        # After .oriented(.right), landscape-X → portrait-Y, landscape-Y → portrait-X.
        # So for a portrait pixel (px_norm, py_norm):
        #   portrait px (horizontal) ← landscape Y axis  → use fy, cy
        #   portrait py (vertical)   ← landscape X axis  → use fx, cx
        lw = landscape_w or image_h   # landscape width  = portrait height
        lh = landscape_h or image_w   # landscape height = portrait width
        # Portrait pixel positions in landscape sensor coords:
        lx = py_norm * lw   # portrait-Y maps to landscape-X
        ly = px_norm * lh   # portrait-X maps to landscape-Y
        ray_x = (ly - cy) / fy   # horizontal in robot frame (left/right)
        ray_y = -(lx - cx) / fx  # vertical in robot frame (up/down, neg = down)
    else:
        # Fallback: estimate from HFOV. Long side carries HFOV_DEG.
        f_est = (max(image_w, image_h) / 2) / math.tan(math.radians(HFOV_DEG / 2))
        ray_x = (px_norm * image_w - image_w / 2) / f_est
        ray_y = -(py_norm * image_h - image_h / 2) / f_est

    ray_z = 1.0

    # Apply camera pitch (positive = tilted down)
    if pitch_deg != 0.0:
        p = math.radians(pitch_deg)
        ray_y, ray_z = (
            ray_y * math.cos(p) - ray_z * math.sin(p),
            ray_y * math.sin(p) + ray_z * math.cos(p),
        )

    # Floor intersection: camera at y = h, floor at y = 0
    # Parametric: y(t) = h + t * ray_y = 0  →  t = -h / ray_y
    if ray_y >= 0:
        return None  # ray points up or horizontal — no floor hit

    t = -camera_height_mm / ray_y
    return ray_x * t, ray_z * t   # (x_mm, z_mm)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _jpeg_size(frame: bytes) -> tuple[int, int]:
    """Return (width, height) by parsing the JPEG SOF marker."""
    data = io.BytesIO(frame)
    data.read(2)  # SOI FF D8
    while True:
        marker, = struct.unpack(">H", data.read(2))
        size,   = struct.unpack(">H", data.read(2))
        if marker in (0xFFC0, 0xFFC2):
            data.read(1)          # sample precision
            h, w = struct.unpack(">HH", data.read(4))
            return w, h
        data.seek(size - 2, 1)   # skip this segment


def _b64(frame: bytes) -> str:
    return base64.standard_b64encode(frame).decode()


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def locate_marker(hub, color: str = "red") -> dict | None:
    """
    Ask Haiku to locate the marker in the current frame.
    Returns {"px": 0..1, "py": 0..1, "confidence": str} or None.
    """
    frame = hub.latest_frame
    if frame is None:
        print("  No frame available.")
        return None

    msg = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=64,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(frame)}},
            {"type": "text",  "text": MARKER_PROMPT.format(color=color)},
        ]}],
    )
    return _parse_json(msg.content[0].text)


# ── Run D1: Geometry only (no driving) ───────────────────────────────────────

def geometry_only(
    hub,
    distances_mm: list[int] = (500, 1000, 1500, 2000),
    camera_height_mm: float = CAMERA_HEIGHT_MM,
    color: str = "red",
) -> list[dict]:
    """
    For each distance: ask user to place marker, locate it, unproject, compare.
    No driving — pure geometry check to calibrate camera_height_mm.
    """
    print(f"\n── D1: Geometry-only unprojection  (camera_height={camera_height_mm}mm) ──\n")
    results = []

    for actual_mm in distances_mm:
        input(f"  Place {color} marker at {actual_mm}mm and press Enter...")

        frame = hub.latest_frame
        if frame is None:
            print("  No frame — skipping.")
            continue
        image_w, image_h = _jpeg_size(frame)

        t = latest_telem(hub)
        pitch_deg = t.get("p", 0.0) if t else 0.0

        intr = hub.latest_intrinsics or {}
        if intr:
            print(f"  Intrinsics: fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
                  f"cx={intr['cx']:.1f} cy={intr['cy']:.1f} "
                  f"(landscape {intr['img_w']:.0f}×{intr['img_h']:.0f})")
        else:
            print("  No intrinsics — falling back to HFOV estimate.")

        print(f"  Locating marker...", end=" ", flush=True)
        loc = locate_marker(hub, color)
        if loc is None:
            print("locate failed — skipping.")
            continue

        px, py   = loc["px"], loc["py"]
        conf     = loc.get("confidence", "?")
        print(f"px={px:.3f}  py={py:.3f}  confidence={conf}")

        result = unproject_floor(
            px, py, camera_height_mm, image_w, image_h, pitch_deg,
            fx=intr.get("fx"), fy=intr.get("fy"),
            cx=intr.get("cx"), cy=intr.get("cy"),
            landscape_w=intr.get("img_w"), landscape_h=intr.get("img_h"),
        )
        if result is None:
            print(f"  Point above horizon — skipping.")
            continue

        x_mm, z_mm = result
        predicted_mm = math.hypot(x_mm, z_mm)
        error_mm     = predicted_mm - actual_mm
        error_pct    = 100 * error_mm / actual_mm

        print(f"  Actual:    {actual_mm}mm")
        print(f"  Predicted: {predicted_mm:.0f}mm  (x={x_mm:+.0f}  z={z_mm:.0f})")
        print(f"  Error:     {error_mm:+.0f}mm  ({error_pct:+.1f}%)\n")

        results.append({
            "actual_mm": actual_mm,
            "predicted_mm": predicted_mm,
            "x_mm": x_mm,
            "z_mm": z_mm,
            "error_mm": error_mm,
            "error_pct": error_pct,
            "px": px,
            "py": py,
            "pitch_deg": pitch_deg,
            "confidence": conf,
        })

    if results:
        print("  ── Summary ──")
        for r in results:
            bar = "█" * int(abs(r["error_pct"]) / 2)
            print(f"  {r['actual_mm']:>5}mm  →  {r['predicted_mm']:>6.0f}mm  "
                  f"err {r['error_mm']:>+6.0f}mm ({r['error_pct']:>+5.1f}%)  {bar}")

        # Hint: if errors are consistently one sign, suggest a height adjustment
        avg_err_pct = sum(r["error_pct"] for r in results) / len(results)
        if abs(avg_err_pct) > 5:
            suggested_h = camera_height_mm * (1 - avg_err_pct / 100)
            print(f"\n  Hint: consistent {avg_err_pct:+.1f}% bias → "
                  f"try camera_height_mm={suggested_h:.0f}mm")

    return results


# ── Run D2: Drive to marker ───────────────────────────────────────────────────

def drive_to_marker(
    hub,
    distance_mm: int = 1000,
    camera_height_mm: float = CAMERA_HEIGHT_MM,
    color: str = "red",
) -> dict | None:
    """
    Full end-to-end test: locate marker, unproject to waypoint, drive there,
    measure actual displacement with VIO.
    """
    print(f"\n── D2: Drive-to-marker  (target={distance_mm}mm) ──\n")

    input(f"  Place {color} marker at {distance_mm}mm and press Enter...")

    frame = hub.latest_frame
    if frame is None:
        print("  No frame.")
        return None
    image_w, image_h = _jpeg_size(frame)

    t = latest_telem(hub)
    pitch_deg = t.get("p", 0.0) if t else 0.0

    intr = hub.latest_intrinsics or {}

    print(f"  Locating marker...", end=" ", flush=True)
    loc = locate_marker(hub, color)
    if loc is None:
        print("failed.")
        return None

    px, py = loc["px"], loc["py"]
    conf   = loc.get("confidence", "?")
    print(f"px={px:.3f}  py={py:.3f}  confidence={conf}")

    result = unproject_floor(
        px, py, camera_height_mm, image_w, image_h, pitch_deg,
        fx=intr.get("fx"), fy=intr.get("fy"),
        cx=intr.get("cx"), cy=intr.get("cy"),
        landscape_w=intr.get("img_w"), landscape_h=intr.get("img_h"),
    )
    if result is None:
        print("  Point above horizon — aborting.")
        return None

    x_mm, z_mm    = result
    predicted_mm  = math.hypot(x_mm, z_mm)
    print(f"  Predicted floor position: z={z_mm:.0f}mm forward  x={x_mm:+.0f}mm lateral")
    print(f"  Predicted total distance: {predicted_mm:.0f}mm  (actual: {distance_mm}mm)")

    # Drive the predicted forward distance (straight line — lateral offset ignored for now)
    p0 = pos(hub)
    if p0 is None:
        print("  No VIO pose — cannot measure.")
        return None

    print(f"  Driving {z_mm:.0f}mm...", end=" ", flush=True)
    moved_mm = straight(hub, z_mm, timeout=30)

    p1 = pos(hub)
    vio_mm = dist(p0, p1) * 1000 if p1 else None

    print(f"hub reported {moved_mm:.0f}mm", end="")
    if vio_mm is not None:
        print(f"  VIO {vio_mm:.0f}mm", end="")
    print()

    error_predicted = predicted_mm - distance_mm
    error_vio       = (vio_mm - distance_mm) if vio_mm else None

    print(f"\n  Geometry error (predicted vs actual): {error_predicted:+.0f}mm "
          f"({100*error_predicted/distance_mm:+.1f}%)")
    if error_vio is not None:
        print(f"  Navigation error (VIO vs actual):     {error_vio:+.0f}mm "
              f"({100*error_vio/distance_mm:+.1f}%)")

    return {
        "distance_mm": distance_mm,
        "predicted_mm": predicted_mm,
        "z_mm": z_mm,
        "x_mm": x_mm,
        "moved_hub_mm": moved_mm,
        "moved_vio_mm": vio_mm,
        "error_predicted_mm": error_predicted,
        "error_vio_mm": error_vio,
    }


# ── Run all ───────────────────────────────────────────────────────────────────

def run_all(hub, camera_height_mm: float = CAMERA_HEIGHT_MM) -> dict:
    """
    Full calibration sweep: geometry survey then a 1000mm drive validation.
    """
    print("\n══════════════════════════════════════════")
    print("  Unprojection calibration  D1 + D2")
    print("══════════════════════════════════════════")

    d1 = geometry_only(hub, [500, 1000, 1500, 2000], camera_height_mm)
    time.sleep(1)

    input("\n  Reposition robot to clear space for driving, then press Enter...")
    d2 = drive_to_marker(hub, 1000, camera_height_mm)

    print("\n══════════════════════════════════════════")
    print("  Summary")
    print("══════════════════════════════════════════")
    if d1:
        print("  D1. Geometry errors:")
        for r in d1:
            print(f"      {r['actual_mm']:>5}mm: {r['error_mm']:>+5.0f}mm ({r['error_pct']:>+5.1f}%)")
    if d2:
        print(f"  D2. Drive to 1000mm:")
        print(f"      Geometry: {d2['error_predicted_mm']:>+5.0f}mm")
        if d2["error_vio_mm"] is not None:
            print(f"      VIO nav:  {d2['error_vio_mm']:>+5.0f}mm")

    return {"D1": d1, "D2": d2}
