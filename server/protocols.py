"""
Layer interface contracts for the robot navigation system.

Three layers, two boundaries:

  Physical  ──[PhysicalSensorProtocol]──▶  Spatial
  Spatial   ──[SpatialNavProtocol]──▶      Semantic

Code in a higher layer should only touch the layer below it through
these protocols. That constraint is what makes parallel agent development
safe: an agent working on the semantic layer cannot break the physical
layer because it never calls into it directly.

A Python Protocol is structural: any class that has the right methods and
properties satisfies the protocol automatically, no explicit inheritance
needed. You can verify compliance at runtime with isinstance() if you
add @runtime_checkable.
"""

from typing import Protocol, TypedDict, Literal
from dataclasses import dataclass


# ── Shared data shapes ────────────────────────────────────────────────────────

class Pose(TypedDict):
    position: list[float]                        # [x, y, z] ARKit metres
    rotation: list[float]                        # [x, y, z, w] quaternion
    tracking: str                                # "normal" | "limited" | "notAvailable"
    timestamp: float | None

class Intrinsics(TypedDict):
    fx: float
    fy: float
    cx: float
    cy: float
    img_w: int
    img_h: int

class Telemetry(TypedDict, total=False):
    ds: float    # distance sensor mm (forward ultrasonic)
    hdg: float   # gyro heading degrees
    obs: float   # obstacle distance mm


# ── Boundary 1: Physical → Spatial ───────────────────────────────────────────

class PhysicalSensorProtocol(Protocol):
    """
    What the spatial layer needs from the physical layer.

    SensorHub (server.py) satisfies this already. If you ever want to swap
    in a simulator or a replay harness, implement these same properties and
    methods and the rest of the stack won't notice.
    """

    @property
    def latest_frame(self) -> bytes | None: ...

    @property
    def latest_pose(self) -> Pose | None: ...

    @property
    def latest_intrinsics(self) -> Intrinsics | None: ...

    @property
    def connected(self) -> bool: ...

    @property
    def hub_state(self) -> str: ...

    def hub_cmd(self, command: str) -> None:
        """Fire-and-forget command to the hub (e.g. 'stop', 'turn:90')."""
        ...

    def wait_hub_state(self, target: str, timeout: float = 15.0) -> bool:
        """Block until hub_state matches target. Returns True on success."""
        ...


# ── Boundary 2: Spatial → Semantic ───────────────────────────────────────────

@dataclass
class FloorPoint:
    """A point on the floor in robot-relative coordinates."""
    x_mm: float    # positive = right of robot
    z_mm: float    # positive = forward from robot


class SpatialNavProtocol(Protocol):
    """
    What the semantic layer needs from the spatial layer.

    This wraps hub_utils.py primitives and unproject_floor into a single
    object. The semantic layer calls this; it never imports hub_utils or
    touches SensorHub directly.

    Concrete implementation: Navigator (to be written) wraps a
    PhysicalSensorProtocol and delegates to the existing hub_utils functions.
    """

    def turn(self, deg: float) -> bool:
        """
        Turn deg degrees (positive = clockwise). Blocking.
        Returns True if event|done received, False on timeout.
        """
        ...

    def straight(self, mm: float) -> float:
        """
        Drive mm forward (negative = reverse). Blocking.
        Returns actual mm moved according to VIO odometry.
        """
        ...

    def pose(self) -> Pose | None:
        """Current ARKit pose, or None if unavailable."""
        ...

    def heading_deg(self) -> float | None:
        """Current yaw projected onto the horizontal plane, in degrees."""
        ...

    def unproject(self, px_norm: float, py_norm: float) -> FloorPoint | None:
        """
        Convert a portrait-image pixel (each axis 0.0–1.0) to a floor point
        in robot-relative mm. Returns None if intrinsics or pose unavailable.

        Pitch correction is applied automatically from the hub's latest
        telemetry heading — callers don't need to pass it.
        """
        ...


# ── Semantic layer data shapes ────────────────────────────────────────────────

class TargetInfo(TypedDict):
    target: str       # natural-language description of the navigable target
    reasoning: str    # one sentence: why this target was chosen

class LocationResult(TypedDict):
    status: Literal["visible", "arrived", "not_visible"]
    px: float | None    # 0.0–1.0 horizontal position in image (was image_x)
    py: float | None    # 0.0–1.0 vertical position in image (NEW — needed for unproject)
    confidence: Literal["high", "medium", "low"] | None
    notes: str

GoalOutcome = Literal["done", "continue", "stuck"]


class SemanticLayerProtocol(Protocol):
    """
    What the navigation loop needs from the semantic layer.

    Implementations can use Claude, a fine-tuned VLM, YOLO, or pure
    heuristics — anything that produces the right shapes. The servo loop
    in navigate.py should call only these methods.
    """

    def select_target(self, frame: bytes) -> TargetInfo | None:
        """
        Pick one navigable target from a single JPEG frame.
        Called once per goal — latency of several seconds is acceptable.
        Returns None if no suitable target exists.
        """
        ...

    def locate_target(self, frame: bytes, target: str) -> LocationResult | None:
        """
        Locate the target in a JPEG frame.
        Called every servo step — should complete in <2s.

        Returns BOTH px (horizontal) AND py (vertical) so the caller can
        pass the result to SpatialNavProtocol.unproject() for a floor point.
        (Current navigate.py only uses px; py enables Phase 4 geometry.)
        """
        ...

    def assess_goal(self, frame: bytes, goal: str) -> GoalOutcome:
        """
        Assess whether the overall navigation goal has been achieved.
        Called after each drive step.

        Returns:
          "done"     — goal is complete, stop navigating
          "continue" — keep going
          "stuck"    — robot cannot make progress, abort
        """
        ...


# ── Spatial map contract ──────────────────────────────────────────────────────

class MapNode(TypedDict):
    node_id: str
    position: list[float]    # [x, y, z] ARKit metres
    description: str
    timestamp: str           # ISO-8601

class SpatialMapProtocol(Protocol):
    """
    Interface for the spatial graph.

    Current SpatialMap (map.py) is a flat list of entries — it satisfies
    add_node, find_nearby, all_nodes, and save today. Phase 5 adds edges
    and re-visit detection on top of the same interface; callers don't change.
    """

    def add_node(self, pose: Pose, description: str) -> MapNode:
        """Record a new observation node. Returns the created node."""
        ...

    def find_nearby(
        self, position: list[float], radius_m: float
    ) -> list[MapNode]:
        """Return all nodes within radius_m of position, nearest first."""
        ...

    def all_nodes(self) -> list[MapNode]:
        """All recorded nodes in insertion order."""
        ...

    def save(self) -> None:
        """Persist current state to disk."""
        ...
