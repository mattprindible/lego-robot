"""
Shared utilities for hub test scripts.

Typical usage:

    from server import start_background
    from deploy import deploy
    from hub_utils import wait_for_connection, straight, turn

    hub = start_background()
    wait_for_connection(hub)
    deploy(hub)

    turn(hub, 90)
    straight(hub, 500)
"""

import math
import time


# ── Connection ────────────────────────────────────────────────────────────────

def wait_for_connection(hub, timeout=90, prompt="Waiting for iPhone..."):
    """Block until hub.connected is True. Raises RuntimeError on timeout."""
    print(prompt)
    deadline = time.time() + timeout
    while not hub.connected and time.time() < deadline:
        time.sleep(0.2)
    if not hub.connected:
        raise RuntimeError(f"iPhone did not connect within {timeout}s")
    print(f"iPhone connected. hub_state={hub.hub_state}")
    return hub


# ── Event waiting ─────────────────────────────────────────────────────────────

def wait_done(hub, timeout=15):
    """Block until event|done arrives. Returns True on success, False on timeout."""
    deadline = time.time() + timeout
    seen = set(hub.hub_lines)
    while time.time() < deadline:
        for line in list(hub.hub_lines):
            if line not in seen and line.startswith("event|done"):
                return True
            seen.add(line)
        time.sleep(0.05)
    return False


# ── Telemetry ─────────────────────────────────────────────────────────────────

def parse_telem(line):
    """Parse a telem| line into a dict. Returns None if not a telem line."""
    if not line or not line.startswith("telem|"):
        return None
    result = {}
    for seg in line.split("|")[1:]:
        if ":" in seg:
            k, v = seg.split(":", 1)
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v
    return result


def latest_telem(hub):
    """Most recent parsed telem from hub_lines. Returns dict or None."""
    for line in reversed(list(hub.hub_lines)):
        t = parse_telem(line)
        if t is not None:
            return t
    return None


def fresh_telem(hub, timeout=1.0):
    """Wait for a telem line newer than the current one. Falls back to latest."""
    deadline = time.time() + timeout
    old = hub.latest_hub_line
    while time.time() < deadline:
        line = hub.latest_hub_line
        if line and line != old and line.startswith("telem|"):
            return parse_telem(line)
        time.sleep(0.02)
    return latest_telem(hub)


# ── Pose / geometry ───────────────────────────────────────────────────────────

def pos(hub):
    """Current VIO position as (x, y, z) tuple, or None."""
    p = hub.latest_pose
    return tuple(p["position"]) if p else None


def dist(p1, p2):
    """3D Euclidean distance between two (x, y, z) tuples, in metres."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def arkit_yaw(hub):
    """
    Heading of the camera's forward direction projected onto the horizontal plane.

    The naive Euler yaw formula breaks when the phone is lying flat on the robot
    (world Y ≈ phone Z → gimbal lock). Instead we rotate the camera's forward
    vector [0,0,-1] by the quaternion and take atan2 of its horizontal projection.

    Returns degrees, or None if no pose available.
    """
    pose = hub.latest_pose
    if pose is None or pose.get("rotation") is None:
        return None
    x, y, z, w = pose["rotation"]
    fx = 2 * (x * z + w * y)
    fz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.atan2(fx, -fz))


def angle_diff(a, b):
    """Signed difference b − a, wrapped to (−180, 180]."""
    return (b - a + 180) % 360 - 180


# ── Motor commands ────────────────────────────────────────────────────────────

def turn(hub, deg, timeout=15):
    """
    Turn deg degrees (positive = clockwise). Blocks until event|done or timeout.
    Returns True on success.
    """
    hub.hub_lines.clear()
    hub.hub_cmd(f"turn:{deg}")
    ok = wait_done(hub, timeout=timeout)
    if not ok:
        print(f"  WARNING: timeout waiting for turn:{deg}")
    time.sleep(0.3)
    return ok


def straight(hub, mm, speed=300, timeout=30):
    """
    Drive mm millimetres using continuous drive + VIO distance check.
    Positive = forward, negative = backward.

    Blocks until the target distance is covered, a safety_stop fires,
    or timeout. Returns actual mm travelled (from VIO) or 0 if no pose.
    """
    if mm == 0:
        return 0.0

    p0 = pos(hub)
    direction = 1 if mm >= 0 else -1
    target_mm = abs(mm)

    hub.hub_cmd(f"drive:{direction * speed}:0")

    deadline = time.time() + timeout
    last_progress_t = time.time()
    last_moved_mm = 0.0

    while time.time() < deadline:
        time.sleep(0.05)

        # Check for safety stop — hub already stopped motors
        for line in list(hub.hub_lines):
            if line.startswith("event|safety_stop"):
                hub.hub_cmd("stop")
                break
        else:
            p1 = pos(hub)
            if p0 and p1:
                moved_mm = dist(p0, p1) * 1000
                if moved_mm >= target_mm:
                    break
                if moved_mm > last_moved_mm + 5:
                    last_progress_t = time.time()
                    last_moved_mm = moved_mm
                elif time.time() - last_progress_t > 2.0:
                    # No progress for 2s — assume safety_stop or stall
                    break
            continue
        break  # safety_stop fired

    hub.hub_cmd("stop")
    time.sleep(0.3)

    p_final = pos(hub)
    return dist(p0, p_final) * 1000 if (p0 and p_final) else 0.0
