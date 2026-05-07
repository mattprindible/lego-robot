"""
Claude vision calls against live iPhone frames.

Usage (after start_background()):
    from vision import describe
    print(describe())
    print(describe("What's the most salient object?"))
    print(describe("Given the camera is at position {pos}, what direction am I facing?"))
"""

import base64
from pathlib import Path
import anthropic

_key_file = Path(__file__).parent / "key.txt"
_client = anthropic.Anthropic(api_key=_key_file.read_text().strip())


def describe(prompt: str = "Briefly describe what you see.", hub=None) -> str:
    """
    Send the latest frame (+ pose if available) to Claude and return its response.
    Pass your own hub instance, or it imports the module-level one automatically.
    """
    if hub is None:
        from server import hub as _hub
        hub = _hub

    frame = hub.latest_frame
    if not frame:
        return "(no frame yet)"

    b64 = base64.standard_b64encode(frame).decode()

    # Append spatial context if we have a pose
    full_prompt = prompt
    pose = hub.latest_pose
    if pose and pose.get("tracking") == "normal":
        x, y, z = pose["position"]
        full_prompt += (
            f"\n\nSpatial context from ARKit: "
            f"camera position x={x:.2f} y={y:.2f} z={z:.2f} metres "
            f"(relative to where AR tracking started)."
        )

    msg = _client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                },
                {"type": "text", "text": full_prompt},
            ],
        }],
    )
    return msg.content[0].text
