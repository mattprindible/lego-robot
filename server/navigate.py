"""
Vision-based navigation with visual servoing.

Claude picks a target ONCE. A servo loop then drives toward it:
  locate target in frame → turn to center it → drive a step → VIO check → repeat

Usage:
    from server import start_background
    hub = start_background()
    # wait for connection...

    from navigate import navigate
    navigate(hub)                    # full pipeline
    navigate(hub, max_steps=5)       # limit steps

    # or separately:
    from navigate import select_target, servo_to_target
    target = select_target(hub)
    servo_to_target(hub, target["target"])
"""

import json
import math
import time
import base64
from pathlib import Path
import anthropic
from hub_utils import straight, turn, pos, dist

_key = Path(__file__).parent / "key.txt"
_client = anthropic.Anthropic(api_key=_key.read_text().strip())

# ── Config ────────────────────────────────────────────────────────────────────

HFOV_DEG   = 70     # iPhone 13 mini horizontal FOV (approx)
STEP_MM    = 250    # drive this far each servo step
MAX_TURN   = 40     # cap turn magnitude
CENTER_TOL = 0.08   # image_x within ±0.08 of center → already centered
MIN_MOVE_M = 0.02   # if VIO shows <2cm movement after a drive → stuck

# ── Prompts ───────────────────────────────────────────────────────────────────

SELECT_PROMPT = """You are navigating a robot through a room. The camera is at robot height (low to ground).

Pick ONE navigable target: an open space, gap, doorway, or area the robot can drive into.
Avoid furniture legs, walls, and dead ends.

Respond with ONLY valid JSON (no markdown):
{
  "target": "<specific description, e.g. 'the gap between the dresser and the wall'>",
  "reasoning": "<one sentence: why this is a good target>"
}"""

LOCATE_PROMPT = """The robot is navigating toward: "{target}"

Look at this camera frame and find that target.

Respond with ONLY valid JSON (no markdown):
{
  "status": "<visible|arrived|not_visible>",
  "image_x": <0.0–1.0 where target appears, 0=far left, 0.5=center, 1.0=far right, or null if not visible>,
  "notes": "<optional one-line observation>"
}

Use "arrived" if the target is very close and fills most of the frame.
Use "not_visible" if you cannot locate the target at all."""


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Core functions ────────────────────────────────────────────────────────────

def select_target(hub) -> dict | None:
    """Claude picks a navigable target from the current frame. Returns {target, reasoning}."""
    frame = hub.latest_frame
    if frame is None:
        print("No frame available.")
        return None

    print("Selecting target...")
    msg = _client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(frame)}},
            {"type": "text", "text": SELECT_PROMPT},
        ]}],
    )
    result = _parse_json(msg.content[0].text)
    if result:
        print(f"Target: {result['target']}")
        print(f"Why:    {result['reasoning']}")
    return result


def locate_target(hub, target: str) -> dict | None:
    """Haiku locates the target in the current frame. Returns {status, image_x, notes}."""
    frame = hub.latest_frame
    if frame is None:
        return None

    msg = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(frame)}},
            {"type": "text", "text": LOCATE_PROMPT.replace("{target}", target)},
        ]}],
    )
    return _parse_json(msg.content[0].text)


def servo_to_target(hub, target: str, max_steps: int = 8) -> dict:
    """
    Visual servo loop: locate → center → drive → VIO check → repeat.
    Returns summary dict with steps taken, final pose, outcome.
    """
    print(f"\n── Servoing toward: '{target}' ──\n")
    steps = 0
    outcome = "max_steps"

    for step in range(max_steps):
        steps = step + 1
        print(f"Step {steps}/{max_steps}", end="  ")

        loc = locate_target(hub, target)
        if loc is None:
            print("locate failed — stopping.")
            outcome = "locate_error"
            break

        status   = loc.get("status", "not_visible")
        image_x  = loc.get("image_x")
        notes    = loc.get("notes", "")
        print(f"status={status}", end="")
        if image_x is not None:
            print(f"  image_x={image_x:.2f}", end="")
        if notes:
            print(f"  ({notes})", end="")
        print()

        if status == "arrived":
            print("Arrived!")
            outcome = "arrived"
            break

        if status == "not_visible" or image_x is None:
            # Rotate slowly to search
            print("  Target not visible — searching (rotating 25°)...")
            turn(hub, 25, timeout=8)
            continue

        # ── Turn to center target ─────────────────────────────────────────────
        turn_deg = (image_x - 0.5) * HFOV_DEG
        turn_deg = max(-MAX_TURN, min(MAX_TURN, turn_deg))
        turn_deg = round(turn_deg)

        if abs(turn_deg) >= 5:
            print(f"  Turning {turn_deg:+d}°...")
            turn(hub, turn_deg, timeout=8)

        # ── Drive a step ──────────────────────────────────────────────────────
        time.sleep(0.5)  # let ARKit settle after turn
        pos_before = pos(hub)
        print(f"  Driving {STEP_MM}mm...", end=" ")
        moved_mm = straight(hub, STEP_MM, timeout=15)

        # ── VIO stuck check ───────────────────────────────────────────────────
        pos_after = pos(hub)
        if pos_before and pos_after:
            origin_dist = dist((0, 0, 0), pos_after)
            if origin_dist < 0.05:
                print(f"moved {moved_mm:.0f}mm (VIO — skipping stuck check, ARKit still initializing)")
            elif moved_mm < MIN_MOVE_M * 1000:
                print(f"moved {moved_mm:.0f}mm (VIO) — looks stuck, stopping.")
                outcome = "stuck"
                break
            else:
                print(f"moved {moved_mm:.0f}mm (VIO)")
        print()

    final_pose = hub.latest_pose
    p = pos(hub)
    print(f"\nOutcome: {outcome}  |  steps: {steps}")
    if p:
        print(f"Final position: x={p[0]:.3f}  y={p[1]:.3f}  z={p[2]:.3f}")
    if final_pose:
        print(f"Tracking: {final_pose['tracking']}")

    return {"outcome": outcome, "steps": steps, "final_pose": final_pose}


def navigate(hub, max_steps: int = 8) -> dict | None:
    """Full pipeline: select target, then servo toward it."""
    target_info = select_target(hub)
    if target_info is None:
        return None
    return servo_to_target(hub, target_info["target"], max_steps=max_steps)
