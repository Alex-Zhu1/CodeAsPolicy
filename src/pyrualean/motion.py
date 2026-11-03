"""Units and limits for time-parameterized joint commands (no simulator imports)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


# Planning choices for this simulator, not hardware ratings or force limits.
PHASES = {
    "transit": dict(velocity=3.0, acceleration=18.0, jerk=180.0),
    "approach": dict(velocity=1.0, acceleration=6.0, jerk=90.0),
    "carry": dict(velocity=1.8, acceleration=10.0, jerk=120.0),
    "precision": dict(velocity=0.45, acceleration=3.0, jerk=45.0),
}


@dataclass(frozen=True)
class NativeMotion:
    """Use the simulator's original waypoint interpolation, without spline limits."""

    def payload(self):
        return {"mode": "native"}


@dataclass(frozen=True)
class MotionProfile:
    """Joint-command limits in rad/s, rad/s² and rad/s³, for every moving joint.

    These bound the planned position/velocity targets, not measured motion or
    contact force. Defaults are planning choices, not hardware ratings. Requires
    a backend that advertises timed substeps; never estimates timing from RPCs.
    """

    velocity: float = 1.0
    acceleration: float = 4.0
    jerk: float = 40.0
    phase: str | None = None

    def __post_init__(self):
        if self.phase is not None and self.phase not in PHASES:
            raise ValueError(f"Unknown motion phase: {self.phase!r}")
        for name in ("velocity", "acceleration", "jerk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

    def payload(self):
        values = asdict(self)
        if self.phase is None:
            values.pop("phase")
        return values


@dataclass(frozen=True)
class MotionWindow:
    """Slower joint limits near one end of the nominal TCP path, in one C2 move.

    distance_m is path travel, not sensed clearance. Crossing segments use the
    near limits in full. For dual motion, travel sums the larger arm displacement
    per segment; this shared clock does not locate either arm's contact point.
    """

    far: MotionProfile
    near: MotionProfile
    distance_m: float = 0.02
    at: str = "end"

    def __post_init__(self):
        if not isinstance(self.far, MotionProfile) or not isinstance(self.near, MotionProfile):
            raise ValueError("far and near must be flat motion profiles")
        if (type(self.distance_m) not in (int, float) or not math.isfinite(self.distance_m)
                or not 0 < self.distance_m <= 1):
            raise ValueError("distance_m must be finite, positive and at most 1 metre")
        if self.at not in ("start", "end"):
            raise ValueError("at must be 'start' or 'end'")
        if any(getattr(self.near, k) > getattr(self.far, k)
               for k in ("velocity", "acceleration", "jerk")):
            raise ValueError("near limits must not exceed far limits")

    def payload(self):
        return {"window": dict(far=self.far.payload(), near=self.near.payload(),
                               distance_m=self.distance_m, at=self.at)}


def resolve_motion(value):
    if value is None or isinstance(value, (MotionProfile, MotionWindow, NativeMotion)):
        return value
    if value == "native":
        return NativeMotion()
    if isinstance(value, str) and value in PHASES:
        return MotionProfile(**PHASES[value], phase=value)
    raise ValueError("motion must be 'native', a phase, robo.motion_profile() or robo.motion_window()")


def motion_from_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("motion payload must be an object")
    try:
        if "window" in payload:
            if set(payload) != {"window"}:
                raise ValueError("window cannot be combined with flat motion limits")
            window = dict(payload["window"])
            window["far"] = MotionProfile(**window["far"])
            window["near"] = MotionProfile(**window["near"])
            return MotionWindow(**window)
        return MotionProfile(**payload)
    except (TypeError, KeyError) as exc:
        raise ValueError("invalid motion profile/window") from exc


PROFILE_SCHEMA = {"type": "object", "properties": {
    **{key: {"type": "number", "exclusiveMinimum": 0}
       for key in ("velocity", "acceleration", "jerk")},
    "phase": {"type": "string", "enum": list(PHASES)},
}, "additionalProperties": False}
MOTION_SCHEMA = {"oneOf": [{"type": "object", "properties": {
    "mode": {"const": "native"},
}, "required": ["mode"], "additionalProperties": False}, PROFILE_SCHEMA, {
    "type": "object", "properties": {"window": {
        "type": "object", "properties": {
            "far": PROFILE_SCHEMA, "near": PROFILE_SCHEMA,
            "distance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "at": {"type": "string", "enum": ["start", "end"]},
        }, "required": ["far", "near", "distance_m", "at"], "additionalProperties": False,
    }}, "required": ["window"], "additionalProperties": False,
}]}
