# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Distil CodeAsPolicy episodes into PEAR Visual-Delta multi-turn training data.

The online layer (``robo.nudge`` / ``robo.visual_align_and_insert``) asks a small VLM for
relative end-effector deltas.  This module produces the data to fine-tune such a model:

* :func:`coalesce` groups recorded native steps into PEAR "move until at rest" macro deltas,
  splitting at gripper switches, contact/stall stops, phase boundaries and displacement caps,
  and dropping zero-motion hold noise.
* :func:`extract_episode` reads a finished episode directory (``result.json``, ``card.json``,
  ``calls.jsonl``, ``steps/NNNN/observation.json`` + camera PNGs) and builds SFT conversations
  or RL prompt/solution samples, optionally restricted to last-mile contact phases.
* :class:`VisualDeltaEnvRecorder` wraps a live env (``step`` / ``get_obs`` / ``get_status``)
  and flushes the same JSONL when a trial ends.
* ``python -m codeaspolicy.visual_delta_distill extract|stats`` is the command line.

Protocol v1 conventions (shared with :mod:`codeaspolicy.visual_delta`):

* ``x, y, z`` deltas are centimetres of the TCP grasp point (link6 origin plus
  ``TCP_OFFSET`` = 0.145 m along link6 +x), ``next - current``.  ``frame="tcp"`` expresses
  the world delta in the current link6 axes (+x pointing, +y finger closing, +z normal).
* ``roll, pitch, yaw`` deltas are radians: the wrapped difference of
  ``robodojo_kinematics.angles()`` (pitch, yaw, roll in degrees); an executor applies
  ``target = current + delta`` through ``kin.orientation(pitch, yaw, roll)``.
* Gripper action ``width`` is a switch: ``+1`` open, ``-1`` close, ``0`` keep.  RoboDojo's
  internal codes are ``OPEN = -1`` / ``CLOSE = 1`` and its ``ee_joint_state`` command is
  ``1.0`` open / ``0.0`` closed.  Observed width is the jaw span in cm,
  ``round((0.0023 + 0.1091 * opening) * 100, 1)``.

Everything here is pure Python (numpy and Pillow are used only when present), so the
pipeline runs in lightweight data-processing environments.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import struct
import sys
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL = "pear.visual_delta.v1"

#: Grasp point ahead of link6 along its pointing axis, metres (``robodojo.TCP_OFFSET``).
TCP_OFFSET = 0.145
#: Jaw span at a measured opening: GRIP_ZERO_M + GRIP_SPAN_M * opening (``robodojo_tools``).
GRIP_ZERO_M = 0.0023
GRIP_SPAN_M = 0.1091

#: RoboDojo's gripper codes (``robo.OPEN`` / ``robo.CLOSE``) and ee command levels.
ROBODOJO_OPEN = -1
ROBODOJO_CLOSE = 1
ROBODOJO_KEEP = 0
EE_OPEN_CMD = 1.0
EE_CLOSE_CMD = 0.0

#: PEAR gripper action switch values.
PEAR_OPEN = 1
PEAR_CLOSE = -1
PEAR_KEEP = 0

ARMS = ("left", "right")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
WRIST_CAMERA = {"left": "cam_left_wrist", "right": "cam_right_wrist"}
AXES = ("x", "y", "z", "roll", "pitch", "yaw")

#: Phases (or tool names) that count as last-mile contact / alignment work.
CONTACT_PHASES = frozenset(
    {
        "align",
        "contact",
        "nudge",
        "visual_align_and_insert",
        "grasp_step",
        "probe",
        "insert",
        "press",
        "close",
        "close_gripper",
    }
)


# ---------------------------------------------------------------------------
# gripper conventions
# ---------------------------------------------------------------------------


def width_cm(opening: float | None) -> float | None:
    """Measured jaw span in centimetres for a gripper opening reading."""

    if opening is None:
        return None
    return round((GRIP_ZERO_M + GRIP_SPAN_M * float(opening)) * 100.0, 1)


def pear_gripper_to_robodojo(value: float) -> int:
    """PEAR switch (``>0`` open, ``<0`` close, ``0`` keep) to RoboDojo ``OPEN=-1`` / ``CLOSE=1``."""

    value = float(value)
    if value > 0:
        return ROBODOJO_OPEN
    if value < 0:
        return ROBODOJO_CLOSE
    return ROBODOJO_KEEP


def robodojo_gripper_to_pear(code: float) -> int:
    """RoboDojo ``OPEN=-1`` / ``CLOSE=1`` / ``KEEP=0`` to the PEAR switch."""

    code = float(code)
    if code < 0:
        return PEAR_OPEN
    if code > 0:
        return PEAR_CLOSE
    return PEAR_KEEP


def ee_command_to_pear(command: float | None, previous: float | None = None) -> int:
    """An ``ee_joint_state`` command level (1.0 open, 0.0 closed) as a PEAR switch.

    With ``previous`` given, an unchanged level is ``0`` (keep): PEAR actions are switches,
    the level itself is state.
    """

    if command is None:
        return PEAR_KEEP
    command = float(command)
    if previous is not None and abs(command - float(previous)) < 0.5:
        return PEAR_KEEP
    return PEAR_OPEN if command >= 0.5 else PEAR_CLOSE


def pear_to_ee_command(value: float, current: float | None = None) -> float | None:
    """A PEAR switch as the ee command level to send (``None`` / ``current`` on keep)."""

    code = pear_gripper_to_robodojo(value)
    if code == ROBODOJO_OPEN:
        return EE_OPEN_CMD
    if code == ROBODOJO_CLOSE:
        return EE_CLOSE_CMD
    return current


# ---------------------------------------------------------------------------
# rotation math (pure Python mirror of robodojo_kinematics)
# ---------------------------------------------------------------------------

Matrix = list[list[float]]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(a: Matrix) -> Matrix:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _matvec(a: Matrix, v: Sequence[float]) -> list[float]:
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _rx(a: float) -> Matrix:
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _ry(a: float) -> Matrix:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rz(a: float) -> Matrix:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def wrap_deg(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def rot_from_quat(q: Sequence[float]) -> Matrix:
    """``[w, x, y, z]`` to a rotation matrix (``kin.rot_from_quat``)."""

    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def orientation(pitch: float, yaw: float, roll: float) -> Matrix:
    """Gripper rotation for (pitch, yaw, roll) degrees: ``Rz(90 + yaw) Ry(pitch) Rx(roll)``."""

    return _matmul(
        _matmul(_rz(math.radians(90.0 + yaw)), _ry(math.radians(pitch))), _rx(math.radians(roll))
    )


_VERTICAL = math.sin(math.radians(1.0))


def angles(rot: Matrix) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees of a gripper rotation (``kin.angles``)."""

    point = [rot[0][0], rot[1][0], rot[2][0]]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < _VERTICAL:
        jaw = [rot[0][1], rot[1][1], rot[2][1]]
        return pitch, wrap_deg(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    rest = _matmul(_transpose(orientation(pitch, yaw, 0.0)), rot)
    roll = math.degrees(math.atan2(rest[2][1], rest[1][1]))
    return pitch, wrap_deg(yaw), wrap_deg(roll)


# ---------------------------------------------------------------------------
# measured arm state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmPose:
    """One arm's measured TCP pose and gripper reading."""

    tcp: tuple[float, float, float]  # metres, world
    rot: tuple[tuple[float, ...], ...]  # link6 rotation, world
    opening: float | None = None

    @property
    def angles_deg(self) -> tuple[float, float, float]:
        return angles([list(r) for r in self.rot])


def _flat(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    out: list[float] = []

    def walk(v: Any) -> None:
        if isinstance(v, (list, tuple)):
            for item in v:
                walk(item)
        else:
            out.append(float(v))

    try:
        walk(value)
    except (TypeError, ValueError):
        return None
    return out


def _scalar(value: Any) -> float | None:
    flat = _flat(value)
    return flat[0] if flat else None


def _fk_pose(arm: str, joints: Sequence[float]) -> tuple[list[float], Matrix] | None:
    try:  # numpy-backed host kinematics; optional
        from .hosts import robodojo_kinematics as kin
    except Exception:  # noqa: BLE001 - no numpy / host in this environment
        return None
    pose = kin.fk(arm, list(joints)[:6])
    return [float(v) for v in pose[:3, 3]], [[float(v) for v in row] for row in pose[:3, :3]]


def arm_pose(state: Mapping[str, Any], arm: str) -> ArmPose | None:
    """The TCP pose of ``arm`` from a RoboDojo state dict.

    ``<arm>_ee_pose`` (link6 ``[x, y, z, qw, qx, qy, qz]``) is used when present, else forward
    kinematics of ``<arm>_arm_joint_state`` (needs numpy).
    """

    link6 = None
    pose = _flat(state.get(f"{arm}_ee_pose"))
    if pose and len(pose) >= 7:
        link6 = (pose[:3], rot_from_quat(pose[3:7]))
    else:
        joints = _flat(state.get(f"{arm}_arm_joint_state"))
        if joints and len(joints) >= 6:
            link6 = _fk_pose(arm, joints)
    if link6 is None:
        return None
    pos, rot = link6
    tcp = tuple(pos[i] + rot[i][0] * TCP_OFFSET for i in range(3))
    return ArmPose(
        tcp=tcp,  # type: ignore[arg-type]
        rot=tuple(tuple(row) for row in rot),
        opening=_scalar(state.get(f"{arm}_ee_joint_state")),
    )


def arm_state_json(pose: ArmPose) -> dict[str, float]:
    """PEAR measured state of one arm: cm (1 decimal), rad (2 decimals)."""

    pitch, yaw, roll = pose.angles_deg
    return {
        "x": round(pose.tcp[0] * 100.0, 1),
        "y": round(pose.tcp[1] * 100.0, 1),
        "z": round(pose.tcp[2] * 100.0, 1),
        "roll": round(math.radians(roll), 2),
        "pitch": round(math.radians(pitch), 2),
        "yaw": round(math.radians(yaw), 2),
    }


def state_json(poses: Mapping[str, ArmPose], arm: str) -> dict[str, Any]:
    """The PEAR observation JSON for ``arm`` (``left`` / ``right`` / ``both``)."""

    if arm == "both":
        out: dict[str, Any] = {}
        for side in ARMS:
            if side in poses:
                out[f"{side}_arm"] = arm_state_json(poses[side])
                out[f"{side}_gripper"] = {"width": width_cm(poses[side].opening)}
        return out
    pose = poses[arm]
    return {"arm": arm_state_json(pose), "gripper": {"width": width_cm(pose.opening)}}


def pose_delta(
    current: ArmPose,
    target: ArmPose,
    frame: str = "world",
    linear_decimals: int = 2,
    angular_decimals: int = 3,
) -> dict[str, float]:
    """PEAR arm delta ``target - current``: cm for xyz, wrapped rad for roll/pitch/yaw."""

    if frame not in ("world", "tcp"):
        raise ValueError(f"frame must be 'world' or 'tcp', got {frame!r}")
    d = [target.tcp[i] - current.tcp[i] for i in range(3)]
    if frame == "tcp":
        d = _matvec(_transpose([list(r) for r in current.rot]), d)
    p0, y0, r0 = current.angles_deg
    p1, y1, r1 = target.angles_deg
    out = {axis: round(v * 100.0, linear_decimals) + 0.0 for axis, v in zip("xyz", d)}
    out["roll"] = round(math.radians(wrap_deg(r1 - r0)), angular_decimals) + 0.0
    out["pitch"] = round(math.radians(wrap_deg(p1 - p0)), angular_decimals) + 0.0
    out["yaw"] = round(math.radians(wrap_deg(y1 - y0)), angular_decimals) + 0.0
    return out


def apply_delta(current: ArmPose, delta: Mapping[str, float], frame: str = "world") -> ArmPose:
    """Inverse of :func:`pose_delta` (what an executor targets), for round-trip checks."""

    d = [float(delta.get(axis, 0.0)) / 100.0 for axis in "xyz"]
    if frame == "tcp":
        d = _matvec([list(r) for r in current.rot], d)
    p, y, r = current.angles_deg
    rot = orientation(
        p + math.degrees(float(delta.get("pitch", 0.0))),
        y + math.degrees(float(delta.get("yaw", 0.0))),
        r + math.degrees(float(delta.get("roll", 0.0))),
    )
    return ArmPose(
        tcp=tuple(current.tcp[i] + d[i] for i in range(3)),  # type: ignore[arg-type]
        rot=tuple(tuple(row) for row in rot),
        opening=current.opening,
    )


# ---------------------------------------------------------------------------
# recorded frames and the macro-step coalescer
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """One recorded state; its metadata describes the transition that produced it.

    Frame 0 is the initial state.  ``gripper`` holds the ee command level per arm in force
    after the transition (``1.0`` open, ``0.0`` closed).
    """

    index: int
    poses: dict[str, ArmPose]
    native_step: int | None = None
    phase: str | None = None
    tool: str | None = None
    gripper: dict[str, float] = field(default_factory=dict)
    stalled: bool = False
    stop_reason: str | None = None
    images: dict[str, str] = field(default_factory=dict)
    ok: bool = True


@dataclass(frozen=True)
class CoalesceConfig:
    """Thresholds of :func:`coalesce` (the caps match ``robo.nudge``'s per-call bounds)."""

    max_linear_m: float = 0.06
    max_angular_deg: float = 20.0
    min_linear_m: float = 0.001
    min_angular_deg: float = 0.5
    split_on_phase: bool = True
    opening_switch: float = 0.05
    frame: str = "world"


@dataclass
class MacroStep:
    """A PEAR macro delta: from ``frames[start]`` to ``frames[end]``."""

    start: int
    end: int
    arm: str
    action: dict[str, Any]
    phase: str | None
    native_steps: int
    status: str
    stop_reason: str | None
    linear_m: float
    angular_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "arm": self.arm,
            "action": self.action,
            "phase": self.phase,
            "native_steps": self.native_steps,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "linear_m": round(self.linear_m, 5),
            "angular_deg": round(self.angular_deg, 3),
        }


def _motion(a: ArmPose, b: ArmPose) -> tuple[float, float]:
    linear = math.dist(a.tcp, b.tcp)
    pa, ya, ra = a.angles_deg
    pb, yb, rb = b.angles_deg
    angular = max(abs(wrap_deg(pb - pa)), abs(wrap_deg(yb - ya)), abs(wrap_deg(rb - ra)))
    return linear, angular


def _arms_of(arm: str) -> tuple[str, ...]:
    return ARMS if arm == "both" else (arm,)


def gripper_switch(a: Frame, b: Frame, side: str, opening_switch: float = 0.05) -> int:
    """PEAR switch for ``side`` between two frames, from commands (or openings if absent)."""

    ca, cb = a.gripper.get(side), b.gripper.get(side)
    if ca is not None and cb is not None:
        return ee_command_to_pear(cb, ca)
    if cb is not None and ca is None and cb < 0.5:
        return PEAR_CLOSE  # first command after the (open) start: a close is a switch
    pa, pb = a.poses.get(side), b.poses.get(side)
    if pa is None or pb is None or pa.opening is None or pb.opening is None:
        return PEAR_KEEP
    change = pb.opening - pa.opening
    if change > opening_switch:
        return PEAR_OPEN
    if change < -opening_switch:
        return PEAR_CLOSE
    return PEAR_KEEP


def select_arm(frames: Sequence[Frame], config: CoalesceConfig = CoalesceConfig()) -> str:
    """``left`` / ``right`` / ``both``: the arm(s) that moved or switched their gripper."""

    active = []
    for side in ARMS:
        poses = [f.poses[side] for f in frames if side in f.poses]
        moved = False
        for a, b in zip(poses, poses[1:]):
            lin, ang = _motion(a, b)
            if lin >= config.min_linear_m or ang >= config.min_angular_deg:
                moved = True
                break
        if not moved:
            moved = any(
                gripper_switch(a, b, side, config.opening_switch)
                for a, b in zip(frames, frames[1:])
            )
        if moved:
            active.append(side)
    if len(active) == 2:
        return "both"
    return active[0] if active else "right"


def _action(
    a: Frame, b: Frame, arm: str, config: CoalesceConfig
) -> tuple[dict, float, float, bool]:
    action: dict[str, Any] = {}
    linear = angular = 0.0
    switched = False
    for side in _arms_of(arm):
        pa, pb = a.poses.get(side), b.poses.get(side)
        if pa is not None and pb is not None:
            delta = pose_delta(pa, pb, config.frame)
            lin, ang = _motion(pa, pb)
            linear, angular = max(linear, lin), max(angular, ang)
        else:
            delta = {axis: 0.0 for axis in AXES}
        switch = gripper_switch(a, b, side, config.opening_switch)
        switched = switched or switch != PEAR_KEEP
        if arm == "both":
            key_arm, key_grip = f"{side}_arm", f"{side}_gripper"
        else:
            key_arm, key_grip = "arm", "gripper"
        action[key_arm] = delta
        action[key_grip] = {"width": switch}
    return action, linear, angular, switched


def coalesce(
    frames: Sequence[Frame], arm: str = "auto", config: CoalesceConfig = CoalesceConfig()
) -> list[MacroStep]:
    """Group consecutive frames into PEAR macro deltas.

    A macro step closes at a contact/stall stop, at a gripper switch, before a phase change,
    before its displacement would exceed ``max_linear_m`` / ``max_angular_deg``, and at the
    last frame.  Macro steps with no motion, no gripper switch and no stop are dropped.
    """

    if len(frames) < 2:
        return []
    if arm == "auto":
        arm = select_arm(frames, config)
    sides = _arms_of(arm)

    def motion(i: int, j: int) -> tuple[float, float]:
        lin = ang = 0.0
        for side in sides:
            a, b = frames[i].poses.get(side), frames[j].poses.get(side)
            if a is not None and b is not None:
                lv, av = _motion(a, b)
                lin, ang = max(lin, lv), max(ang, av)
        return lin, ang

    steps: list[MacroStep] = []
    start = 0
    native = 0
    last = len(frames) - 1
    for t in range(1, len(frames)):
        prev, cur = frames[t - 1], frames[t]
        if prev.native_step is not None and cur.native_step is not None:
            native += max(0, cur.native_step - prev.native_step)
        else:
            native += 1
        close = t == last or cur.stalled
        close = close or any(gripper_switch(prev, cur, s, config.opening_switch) for s in sides)
        if not close and config.split_on_phase and frames[t + 1].phase != cur.phase:
            close = True
        if not close:
            lin, ang = motion(start, t + 1)
            close = lin > config.max_linear_m or ang > config.max_angular_deg
        if not close:
            continue
        action, linear, angular, switched = _action(frames[start], cur, arm, config)
        still = linear < config.min_linear_m and angular < config.min_angular_deg
        if not (still and not switched and not cur.stalled):
            steps.append(
                MacroStep(
                    start=start,
                    end=t,
                    arm=arm,
                    action=action,
                    phase=cur.phase,
                    native_steps=native,
                    status="stopped" if cur.stalled else "reached",
                    stop_reason=cur.stop_reason if cur.stalled else None,
                    linear_m=linear,
                    angular_deg=angular,
                )
            )
        start, native = t, 0
    return steps


# ---------------------------------------------------------------------------
# PEAR prompts and conversations
# ---------------------------------------------------------------------------

IMAGE_TOKEN = "<image>"

SYSTEM_RULES = """You control a robot arm by relative end-effector deltas, one JSON action per turn.
Each turn you see the measured state and labelled camera images. Reply with optional
<think>...</think> reasoning and then one fenced ```json block.

Action JSON (all fields required, 0 for unchanged):
- single arm: {"arm": {"x": cm, "y": cm, "z": cm, "roll": rad, "pitch": rad, "yaw": rad},
  "gripper": {"width": g}}
- both arms: {"left_arm": {...}, "right_arm": {...}, "left_gripper": {"width": g},
  "right_gripper": {"width": g}}
- x, y, z are centimetres; roll, pitch, yaw are radians; gripper width is a switch:
  +1 open, -1 close, 0 keep.
- Add "done": true when the sub-goal is complete.
The arm moves until it reaches the target or stops against contact; the next turn reports
"reached" or "stopped"."""

ENVIRONMENT = {
    "world": (
        "Coordinates are the table frame in centimetres: +x to the robot's right, +y away "
        "from the robot, +z up. The state reports the grasp point between the fingertips."
    ),
    "tcp": (
        "Deltas are in the gripper frame in centimetres: +x along the pointing direction, "
        "+y along the finger-closing axis, +z the gripper normal. The state reports the grasp "
        "point in the table frame (+x right, +y away from the robot, +z up)."
    ),
}

ANGLES_NOTE = (
    "Angles: pitch is how far the pointing direction is below horizontal, yaw its heading on "
    "the table (0 away from the robot, + to the robot's left), roll the turn of the fingers "
    "about the pointing direction."
)


def system_prompt(task: str, frame: str = "world", cameras: Sequence[str] = CAMERAS) -> str:
    cams = ", ".join(f"[{c}]" for c in cameras)
    return (
        f"{SYSTEM_RULES}\n\n## ENVIRONMENT\n{ENVIRONMENT[frame]}\n{ANGLES_NOTE}\n"
        f"Cameras: {cams} (wrist crops are labelled with a _crop suffix).\n\n## TASK\n{task}"
    )


def user_prompt(
    state: Mapping[str, Any],
    images: Sequence[tuple[str, str]],
    status: str | None = None,
    stop_reason: str | None = None,
) -> str:
    """User turn: execution status of the previous action, state JSON, labelled images."""

    lines = []
    if status is None:
        lines.append("## LAST ACTION\nnone (first turn)")
    else:
        detail = f" ({stop_reason})" if stop_reason else ""
        lines.append(f"## LAST ACTION\n{status}{detail}")
    lines.append("## STATE\n```json\n" + json.dumps(state, separators=(", ", ": ")) + "\n```")
    if images:
        lines.append("## IMAGES\n" + "\n".join(f"[{label}] {IMAGE_TOKEN}" for label, _ in images))
    return "\n".join(lines)


_WORDS = {
    "world": {"x": ("right", "left"), "y": ("away", "toward the robot"), "z": ("up", "down")},
    "tcp": {"x": ("forward", "back"), "y": ("along +closing axis", "along -closing axis"),
            "z": ("along +normal", "along -normal")},
}


def think_text(action: Mapping[str, Any], frame: str = "world", phase: str | None = None) -> str:
    """Short geometric rationale for a reference action (templated, deterministic)."""

    parts = []
    arms = [(k, action[k]) for k in ("arm", "left_arm", "right_arm") if k in action]
    for key, delta in arms:
        who = "" if key == "arm" else key.replace("_arm", "") + " arm: "
        moves = []
        for axis in "xyz":
            v = float(delta.get(axis, 0.0))
            if abs(v) >= 0.05:
                pos, neg = _WORDS[frame][axis]
                moves.append(f"{abs(v):.1f} cm {pos if v > 0 else neg}")
        for axis in ("roll", "pitch", "yaw"):
            v = float(delta.get(axis, 0.0))
            if abs(v) >= 0.005:
                moves.append(f"{axis} {math.degrees(v):+.0f} deg")
        parts.append(who + (", ".join(moves) if moves else "hold position"))
    for key in ("gripper", "left_gripper", "right_gripper"):
        if key in action:
            g = float(action[key].get("width", 0))
            if g:
                who = "" if key == "gripper" else key.replace("_gripper", "") + " "
                parts.append(f"{who}gripper {'open' if g > 0 else 'close'}")
    if action.get("done"):
        parts.append("the sub-goal is complete")
    lead = f"Phase {phase}: " if phase else ""
    return lead + "; ".join(parts) + "."


def assistant_reply(
    action: Mapping[str, Any], frame: str = "world", phase: str | None = None, think: bool = True
) -> str:
    body = "```json\n" + json.dumps(action, separators=(", ", ": ")) + "\n```"
    return (f"<think>{think_text(action, frame, phase)}</think>\n" if think else "") + body


def zero_action(arm: str) -> dict[str, Any]:
    zero = {axis: 0.0 for axis in AXES}
    if arm == "both":
        return {
            "left_arm": dict(zero),
            "right_arm": dict(zero),
            "left_gripper": {"width": 0},
            "right_gripper": {"width": 0},
        }
    return {"arm": dict(zero), "gripper": {"width": 0}}


def frame_images(
    frame: Frame,
    arm: str,
    cameras: Sequence[str] | None = None,
    crop: Callable[[str, str], str | None] | None = None,
) -> list[tuple[str, str]]:
    """Ordered ``(label, path)`` images of a frame, wrist crops after their camera."""

    out = []
    for cam in cameras or CAMERAS:
        path = frame.images.get(cam)
        if not path:
            continue
        out.append((cam, path))
        is_active_wrist = cam in {WRIST_CAMERA[s] for s in _arms_of(arm)}
        if crop is not None and is_active_wrist:
            cropped = crop(cam, path)
            if cropped:
                out.append((f"{cam}_crop", cropped))
    return out


@dataclass(frozen=True)
class DatasetOptions:
    """How conversations are rendered."""

    frame: str = "world"
    think: bool = True
    images: bool = True
    cameras: tuple[str, ...] | None = None
    done_turn: bool = True  # append a terminal {"done": true} turn for successful segments


def build_conversation(
    frames: Sequence[Frame],
    steps: Sequence[MacroStep],
    task: str,
    options: DatasetOptions = DatasetOptions(),
    success: bool = False,
    crop: Callable[[str, str], str | None] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """PEAR multi-turn messages and the ordered image paths their ``<image>`` tokens refer to."""

    if not steps:
        return [], []
    arm = steps[0].arm
    cameras = options.cameras or CAMERAS
    messages = [{"role": "system", "content": system_prompt(task, options.frame, cameras)}]
    images: list[str] = []
    status: str | None = None
    reason: str | None = None

    def user_turn(frame: Frame) -> None:
        shown = frame_images(frame, arm, cameras, crop) if options.images else []
        images.extend(path for _, path in shown)
        state = state_json(frame.poses, arm)
        messages.append({"role": "user", "content": user_prompt(state, shown, status, reason)})

    for step in steps:
        user_turn(frames[step.start])
        messages.append(
            {
                "role": "assistant",
                "content": assistant_reply(step.action, options.frame, step.phase, options.think),
            }
        )
        status, reason = step.status, step.stop_reason
    if success and options.done_turn:
        user_turn(frames[steps[-1].end])
        final = dict(zero_action(arm), done=True)
        messages.append(
            {
                "role": "assistant",
                "content": assistant_reply(final, options.frame, None, options.think),
            }
        )
    return messages, images


def rl_samples(
    messages: Sequence[Mapping[str, str]], images: Sequence[str]
) -> list[dict[str, Any]]:
    """Split a conversation into per-turn RL samples: prompt so far and the reference action."""

    out = []
    used = 0
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            used += msg["content"].count(IMAGE_TOKEN)
        if msg["role"] != "assistant":
            continue
        text = msg["content"]
        block = text.split("```json", 1)[1].rsplit("```", 1)[0]
        out.append(
            {
                "prompt": [dict(m) for m in messages[:i]],
                "images": list(images[:used]),
                "solution": json.loads(block),
                "reference": text,
            }
        )
    return out


# ---------------------------------------------------------------------------
# images: PNG read / write / crop (Pillow when present, pure Python otherwise)
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def encode_png(rows: Sequence[Sequence[Sequence[int]]]) -> bytes:
    """8-bit RGB PNG bytes from ``rows[y][x] = (r, g, b[, a])`` (alpha dropped)."""

    height = len(rows)
    width = len(rows[0]) if height else 0
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for px in row:
            raw.extend(int(v) & 0xFF for v in list(px)[:3])
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(data: bytes) -> list[list[tuple[int, int, int]]]:
    """RGB rows of an 8-bit non-interlaced PNG (grey, grey+alpha, RGB or RGBA)."""

    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG")
    pos, idat = 8, bytearray()
    width = height = depth = ctype = interlace = 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.extend(body)
        elif kind == b"IEND":
            break
    if depth != 8 or ctype not in _CHANNELS or interlace:
        raise ValueError(
            f"unsupported PNG (depth {depth}, colour type {ctype}, interlace {interlace})"
        )
    bpp = _CHANNELS[ctype]
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    prev = bytearray(stride)
    rows = []
    for y in range(height):
        base = y * (stride + 1)
        kind = raw[base]
        line = bytearray(raw[base + 1 : base + 1 + stride])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 1:
                line[i] = (line[i] + a) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + b) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
            elif kind == 4:
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
        prev = line
        row = []
        for x in range(width):
            px = line[x * bpp : (x + 1) * bpp]
            row.append((px[0], px[0], px[0]) if bpp <= 2 else (px[0], px[1], px[2]))
        rows.append(row)
    return rows


def _image_rows(image: Any) -> list:
    """Rows of pixels from a numpy array or nested lists (H x W x C)."""

    if hasattr(image, "tolist"):
        image = image.tolist()
    return [[list(px)[:3] for px in row] for row in image]


def write_png(path: Path, image: Any) -> Path:
    """Write an H x W x C uint8 image (numpy array or nested lists) as PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        from PIL import Image

        Image.fromarray(np.asarray(image)[..., :3].astype("uint8")).save(path, format="PNG")
    except ImportError:
        path.write_bytes(encode_png(_image_rows(image)))
    return path


def crop_box(width: int, height: int, box: Sequence[float]) -> tuple[int, int, int, int]:
    """Pixel box of fractional ``(x0, y0, x1, y1)``, at least one pixel wide and high."""

    x0, y0, x1, y1 = (float(v) for v in box)
    left, top = int(round(x0 * width)), int(round(y0 * height))
    right = max(left + 1, int(round(x1 * width)))
    bottom = max(top + 1, int(round(y1 * height)))
    return max(0, left), max(0, top), min(width, right), min(height, bottom)


#: Default wrist ROI: the grasp point projects near the image centre of a wrist camera.
DEFAULT_CROP = (0.25, 0.25, 0.75, 0.75)


def crop_png(src: Path, dst: Path, box: Sequence[float] = DEFAULT_CROP) -> Path:
    """Crop a PNG to a fractional box (no resampling: keeps the native resolution)."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        with Image.open(src) as img:
            img.convert("RGB").crop(crop_box(img.width, img.height, box)).save(dst, format="PNG")
    except ImportError:
        rows = decode_png(Path(src).read_bytes())
        left, top, right, bottom = crop_box(len(rows[0]), len(rows), box)
        dst.write_bytes(encode_png([row[left:right] for row in rows[top:bottom]]))
    return dst


# ---------------------------------------------------------------------------
# offline episodes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """A finished episode directory, read back."""

    path: Path
    task: str
    success: bool | None
    frames: list[Frame]
    calls: list[dict[str, Any]]
    result: dict[str, Any]

    @property
    def native_steps(self) -> int | None:
        steps = self.result.get("env_steps")
        if isinstance(steps, int):
            return steps
        known = [f.native_step for f in self.frames if f.native_step is not None]
        return max(known) - min(known) if known else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def load_calls(path: Path) -> list[dict[str, Any]]:
    """Records of ``calls.jsonl`` (malformed lines skipped)."""

    out = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def call_phase(call: Mapping[str, Any]) -> str | None:
    """The phase of a call: explicit ``phase`` (kwargs, then summary), else its tool name."""

    for source in (call.get("kwargs") or {}, call.get("summary") or {}):
        phase = source.get("phase") if isinstance(source, Mapping) else None
        if isinstance(phase, str) and phase:
            return phase
    tool = call.get("tool")
    return str(tool) if tool else None


def call_stop(call: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Whether a call ended on contact / stall, and why."""

    summary = call.get("summary") or {}
    if not isinstance(summary, Mapping):
        return False, None
    reason = summary.get("stop_reason")
    reason = str(reason) if reason else None
    stalled = bool(summary.get("stalled")) or summary.get("status") == "stopped"
    if reason and any(word in reason for word in ("contact", "stall")):
        stalled = True
    return stalled, (reason or ("contact_or_stall" if stalled else None))


def call_gripper(call: Mapping[str, Any]) -> dict[str, float]:
    """Gripper levels (1.0 open, 0.0 closed) a call commands, by arm."""

    tool = str(call.get("tool") or "")
    kwargs = call.get("kwargs") or {}
    level = None
    if tool in ("open_gripper", "open", "release"):
        level = EE_OPEN_CMD
    elif tool in ("close_gripper", "close"):
        level = EE_CLOSE_CMD
    else:
        value = kwargs.get("gripper")
        if value in ("open", ROBODOJO_OPEN):
            level = EE_OPEN_CMD
        elif value in ("close", "closed", ROBODOJO_CLOSE):
            level = EE_CLOSE_CMD
    if level is None:
        return {}
    arm = kwargs.get("arm") or "right"
    return {side: level for side in (ARMS if arm == "both" else (arm,)) if side in ARMS}


def _frame_images(folder: Path, meta: Mapping[str, Any]) -> dict[str, str]:
    images = {}
    declared = meta.get("images") if isinstance(meta.get("images"), Mapping) else {}
    for cam in CAMERAS:
        artifact = (declared.get(cam) or {}).get("artifact") if cam in declared else None
        path = folder / (artifact or f"{cam}.png")
        if path.is_file():
            images[cam] = str(path.resolve())
    return images


def load_episode(directory: str | Path) -> Episode:
    """Read ``result.json``, ``card.json``, ``calls.jsonl`` and ``steps/`` of an episode."""

    root = Path(directory)
    result = _read_json(root / "result.json")
    card = _read_json(root / "card.json")
    calls = load_calls(root / "calls.jsonl")
    records = []
    steps_dir = root / "steps"
    if steps_dir.is_dir():
        for folder in sorted(p for p in steps_dir.iterdir() if p.is_dir()):
            record = _read_json(folder / "observation.json")
            if record:
                records.append((folder, record))
    by_action = {c.get("action_id"): c for c in calls if c.get("action_id")}
    # cumulative native-step intervals of stateful calls, for records without action ids
    intervals, total = [], 0
    for call in calls:
        if call.get("stateful") and call.get("env_steps"):
            intervals.append((total, total + int(call["env_steps"]), call))
            total += int(call["env_steps"])

    frames: list[Frame] = []
    owner: list[dict[str, Any] | None] = []
    levels: dict[str, float] = {}
    for index, (folder, record) in enumerate(records):
        meta = record.get("observation") or {}
        state = record.get("state") or {}
        poses = {side: p for side in ARMS if (p := arm_pose(state, side)) is not None}
        native = meta.get("native_steps")
        native = int(native) if isinstance(native, (int, float)) else None
        call = by_action.get(meta.get("action_id"))
        if call is None and index > 0 and native is not None:
            base = frames[0].native_step or 0
            call = next((c for lo, hi, c in intervals if lo < native - base <= hi), None)
        commands = meta.get("gripper_commands") or {}
        explicit = {s: v for s in ARMS if (v := _scalar(commands.get(s))) is not None}
        if call is not None:
            levels.update(call_gripper(call))
        levels.update(explicit)
        frames.append(
            Frame(
                index=index,
                poses=poses,
                native_step=native,
                phase=call_phase(call) if call else None,
                tool=str(call.get("tool")) if call else None,
                gripper=dict(levels),
                images=_frame_images(folder, meta),
                ok=bool(call.get("ok", True)) if call else True,
            )
        )
        owner.append(call)
        recorded = record.get("visual_delta")  # written by VisualDeltaEnvRecorder
        if call is None and isinstance(recorded, Mapping):
            frames[-1].phase = recorded.get("phase")
            frames[-1].stalled = bool(recorded.get("stalled"))
            frames[-1].stop_reason = recorded.get("stop_reason")
    # a stop belongs to the last frame a stopping call produced
    for i, call in enumerate(owner):
        if call is None or (i + 1 < len(owner) and owner[i + 1] is call):
            continue
        frames[i].stalled, frames[i].stop_reason = call_stop(call)

    task = (
        card.get("task_language")
        or result.get("task_language")
        or next((r.get("instruction") for _, r in records if r.get("instruction")), None)
        or root.name
    )
    success = result.get("environment_success")
    return Episode(
        path=root,
        task=str(task),
        success=bool(success) if success is not None else None,
        frames=frames,
        calls=calls,
        result=result,
    )


def find_episodes(paths: Iterable[str | Path]) -> list[Path]:
    """Episode directories (holding ``steps/`` or ``result.json``) under the given paths."""

    found: set[Path] = set()
    for p in paths:
        p = Path(p)
        if (p / "steps").is_dir() or (p / "result.json").is_file():
            found.add(p)
            continue
        if p.is_dir():
            for marker in p.rglob("result.json"):
                found.add(marker.parent)
            for marker in p.rglob("steps"):
                if marker.is_dir() and (marker / "0000").exists():
                    found.add(marker.parent)
    return sorted(found)


# ---------------------------------------------------------------------------
# extraction: episodes -> PEAR JSONL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractOptions:
    """What :func:`extract_episode` keeps and how it renders it."""

    success_only: bool = False
    last_mile: bool = False
    phases: frozenset[str] = CONTACT_PHASES
    radius_m: float | None = None
    target: tuple[float, float, float] | None = None
    arm: str = "auto"
    min_turns: int = 1
    format: str = "sft"  # or "rl"
    crop: bool = False
    crop_box: tuple[float, float, float, float] = DEFAULT_CROP
    coalesce: CoalesceConfig = CoalesceConfig()
    dataset: DatasetOptions = DatasetOptions()


def segments(
    frames: Sequence[Frame], options: ExtractOptions, arm: str
) -> list[tuple[int, int]]:
    """``(first, last)`` frame ranges to turn into conversations.

    Without last-mile filters, the whole episode.  With them, maximal runs of transitions
    whose phase is a contact phase or that end within ``radius_m`` of the target (the explicit
    ``target``, else the final TCP position), each including the frame before the run.
    """

    if len(frames) < 2:
        return []
    if not options.last_mile and options.radius_m is None:
        return [(0, len(frames) - 1)]
    sides = _arms_of(arm)
    target = options.target
    keep = [False] * len(frames)
    for t in range(1, len(frames)):
        frame = frames[t]
        if options.last_mile and (frame.phase in options.phases or frame.tool in options.phases):
            keep[t] = True
        elif options.radius_m is not None:
            for side in sides:
                pose, last = frame.poses.get(side), frames[-1].poses.get(side)
                goal = target or (last.tcp if last else None)
                if pose is None or goal is None:
                    continue
                if math.dist(pose.tcp, goal) < options.radius_m:
                    keep[t] = True
    out, t = [], 1
    while t < len(frames):
        if not keep[t]:
            t += 1
            continue
        start = t
        while t + 1 < len(frames) and keep[t + 1]:
            t += 1
        out.append((start - 1, t))
        t += 1
    return out


def _cropper(
    out_dir: Path | None, episode_name: str, box: Sequence[float]
) -> Callable[[str, str], str | None]:
    def crop(cam: str, path: str) -> str | None:
        src = Path(path)
        base = out_dir if out_dir is not None else src.parent
        dst = base / "crops" / episode_name / f"{src.parent.name}_{cam}_crop.png"
        try:
            return str(crop_png(src, dst, box).resolve())
        except (OSError, ValueError, zlib.error):
            return None

    return crop


def episode_records(
    episode: Episode, options: ExtractOptions = ExtractOptions(), out_dir: Path | None = None
) -> list[dict[str, Any]]:
    """PEAR SFT conversations (or RL samples) of one episode."""

    if options.success_only and not episode.success:
        return []
    frames = episode.frames
    if len(frames) < 2:
        return []
    arm = select_arm(frames, options.coalesce) if options.arm == "auto" else options.arm
    crop = _cropper(out_dir, episode.path.name, options.crop_box) if options.crop else None
    records = []
    for number, (first, last) in enumerate(segments(frames, options, arm)):
        window = frames[first : last + 1]
        steps = coalesce(window, arm, options.coalesce)
        if len(steps) < options.min_turns:
            continue
        phases = sorted({s.phase for s in steps if s.phase})
        task = episode.task
        if options.last_mile or options.radius_m is not None:
            where = ", ".join(phases) or "near target"
            task += f"\nSub-goal: precise last-mile alignment / contact ({where})."
        ends_episode = last == len(frames) - 1
        messages, images = build_conversation(
            window, steps, task, options.dataset, bool(episode.success) and ends_episode, crop
        )
        base = {
            "id": f"{episode.path.name}#{number}",
            "protocol": PROTOCOL,
            "task": episode.task,
            "arm": arm,
            "frame": options.dataset.frame,
            "success": episode.success,
            "source": {"episode": str(episode.path), "frames": [first, last]},
            "phases": phases,
            "macro_steps": len(steps),
            "native_steps": sum(s.native_steps for s in steps),
            "stopped_steps": sum(1 for s in steps if s.status == "stopped"),
            "steps": [s.to_dict() for s in steps],
        }
        if options.format == "rl":
            for turn, sample in enumerate(rl_samples(messages, images)):
                records.append(dict(base, id=f"{base['id']}:{turn}", turn=turn, **sample))
        else:
            records.append(dict(base, messages=messages, images=images))
    return records


def extract(
    paths: Iterable[str | Path], out: str | Path, options: ExtractOptions = ExtractOptions()
) -> dict[str, int]:
    """Convert episode directories under ``paths`` into one JSONL file; returns counts."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"episodes": 0, "used": 0, "records": 0, "macro_steps": 0, "native_steps": 0}
    with out.open("w", encoding="utf-8") as handle:
        for directory in find_episodes(paths):
            counts["episodes"] += 1
            records = episode_records(load_episode(directory), options, out.parent)
            if records:
                counts["used"] += 1
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["records"] += 1
                if options.format != "rl" or record.get("turn") == 0:
                    counts["macro_steps"] += record["macro_steps"]
                    counts["native_steps"] += record["native_steps"]
    return counts


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    mid = ordered[len(ordered) // 2]
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(mid, 4),
        "max": round(ordered[-1], 4),
    }


def stats(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro chunks vs native steps, delta distributions and contact-stop ratio."""

    seen: set[str] = set()
    macro = native = stopped = conversations = 0
    linear: list[float] = []
    angular: list[float] = []
    per_axis: dict[str, list[float]] = {axis: [] for axis in AXES}
    switches = {"open": 0, "close": 0}
    phases: dict[str, int] = {}
    for record in records:
        key = str(record.get("id", "")).split(":")[0]
        if key in seen:
            continue  # RL samples repeat their conversation's steps
        seen.add(key)
        conversations += 1
        for step in record.get("steps") or []:
            macro += 1
            native += int(step.get("native_steps") or 0)
            stopped += step.get("status") == "stopped"
            linear.append(float(step.get("linear_m") or 0.0) * 100.0)
            angular.append(float(step.get("angular_deg") or 0.0))
            phases[step.get("phase") or "none"] = phases.get(step.get("phase") or "none", 0) + 1
            action = step.get("action") or {}
            for k, v in action.items():
                if k.endswith("arm") and isinstance(v, Mapping):
                    for axis in AXES:
                        per_axis[axis].append(abs(float(v.get(axis, 0.0))))
                elif k.endswith("gripper") and isinstance(v, Mapping):
                    g = float(v.get("width", 0))
                    if g > 0:
                        switches["open"] += 1
                    elif g < 0:
                        switches["close"] += 1
    return {
        "conversations": conversations,
        "macro_steps": macro,
        "native_steps": native,
        "native_per_macro": round(native / macro, 3) if macro else None,
        "contact_stop_ratio": round(stopped / macro, 4) if macro else None,
        "linear_cm": _summary(linear),
        "angular_deg": _summary(angular),
        "abs_delta": {axis: _summary(v) for axis, v in per_axis.items()},
        "gripper_switches": switches,
        "phases": dict(sorted(phases.items())),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return load_calls(Path(path))


# ---------------------------------------------------------------------------
# live rollout recording
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class _Phase:
    def __init__(self, recorder: VisualDeltaEnvRecorder, name: str | None) -> None:
        self.recorder, self.name, self.previous = recorder, name, None

    def __enter__(self) -> VisualDeltaEnvRecorder:
        self.previous = self.recorder.current_phase
        self.recorder.current_phase = self.name
        return self.recorder

    def __exit__(self, *exc: Any) -> None:
        self.recorder.current_phase = self.previous


class VisualDeltaEnvRecorder:
    """Wrap an env (``step`` / ``get_obs`` / ``get_status``) and record PEAR trajectories.

    Every observation read after an action becomes a :class:`Frame` (state, gripper levels
    from the actions' ``<arm>_ee_joint_state``, phase, stops, camera PNGs).  On
    :meth:`on_trial_end` the trial is coalesced into macro deltas and appended to
    ``<out_dir>/visual_delta.jsonl``; ``steps/NNNN/observation.json`` files are written too,
    so :func:`load_episode` can re-extract a trial later with other options.

    Everything else is delegated to the wrapped env, so it drops into evaluation loops.
    """

    def __init__(
        self,
        env: Any,
        out_dir: str | Path,
        *,
        task: str | None = None,
        options: ExtractOptions = ExtractOptions(),
        save_images: bool = True,
        cameras: Sequence[str] = CAMERAS,
        image_every: int = 1,
        jsonl_name: str = "visual_delta.jsonl",
    ) -> None:
        self._env = env
        self.out_dir = Path(out_dir)
        self.task = task
        self.options = options
        self.save_images = save_images
        self.cameras = tuple(cameras)
        self.image_every = max(1, int(image_every))
        self.jsonl_path = self.out_dir / jsonl_name
        self.current_phase: str | None = None
        self.trial = 0
        self._reset_trial()

    # -- delegation ---------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def _reset_trial(self) -> None:
        self.frames: list[Frame] = []
        self._levels: dict[str, float] = {}
        self._pending = True  # the first observation is always recorded
        self._actions = 0
        self._stop: str | None = None
        self._instruction: str | None = None
        self._last_status: dict[str, Any] = {}

    @property
    def trial_dir(self) -> Path:
        return self.out_dir / f"trial_{self.trial:04d}"

    # -- annotations --------------------------------------------------------------------------

    def phase(self, name: str | None) -> _Phase:
        """Context manager labelling the transitions recorded inside it."""

        return _Phase(self, name)

    def set_phase(self, name: str | None) -> None:
        self.current_phase = name

    def mark_stop(self, reason: str = "contact_or_stall") -> None:
        """The last motion stopped against contact / stalled."""

        if not self._pending and self.frames:
            self.frames[-1].stalled, self.frames[-1].stop_reason = True, reason
        else:
            self._stop = reason

    # -- env API ------------------------------------------------------------------------------

    def get_obs(self, *args: Any, **kwargs: Any) -> Any:
        obs = self._env.get_obs(*args, **kwargs)
        if self._pending:
            self._capture(obs)
        return obs

    def get_status(self, *args: Any, **kwargs: Any) -> Any:
        status = self._env.get_status(*args, **kwargs)
        if isinstance(status, Mapping):
            self._last_status = dict(status)
        return status

    def step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(action, Mapping):
            for side in ARMS:
                level = _scalar(action.get(f"{side}_ee_joint_state"))
                if level is not None:
                    self._levels[side] = level
        result = self._env.step(action, *args, **kwargs)
        self._actions += 1
        self._pending = True
        self._last_status = {}  # stale once the env has stepped
        if isinstance(result, Mapping) and isinstance(result.get("state"), Mapping):
            self._capture(result)
        return result

    # -- recording ----------------------------------------------------------------------------

    def _capture(self, obs: Any) -> None:
        if not isinstance(obs, Mapping):
            return
        state = obs.get("state") or {}
        index = len(self.frames)
        native = self._last_status.get("step")
        native = int(native) if isinstance(native, (int, float)) else self._actions
        frame = Frame(
            index=index,
            poses={s: p for s in ARMS if (p := arm_pose(state, s)) is not None},
            native_step=native,
            phase=self.current_phase,
            gripper=dict(self._levels),
            stalled=self._stop is not None,
            stop_reason=self._stop,
        )
        self._stop = None
        self._pending = False
        if self._instruction is None:
            text = obs.get("instruction") or obs.get("instructions")
            self._instruction = str(text) if text else None
        folder = self.trial_dir / "steps" / f"{index:04d}"
        if self.save_images and index % self.image_every == 0:
            for cam in self.cameras:
                color = ((obs.get("vision") or {}).get(cam) or {}).get("color")
                if color is not None:
                    frame.images[cam] = str(write_png(folder / f"{cam}.png", color).resolve())
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "state": _plain(state),
            "instruction": self._instruction or "",
            "observation": {
                "step": index,
                "native_steps": native,
                "gripper_commands": dict(self._levels),
                "action_id": None,
            },
            "visual_delta": {
                "phase": frame.phase,
                "stalled": frame.stalled,
                "stop_reason": frame.stop_reason,
            },
        }
        (folder / "observation.json").write_text(json.dumps(record) + "\n", encoding="utf-8")
        self.frames.append(frame)

    def _rewrite_stop(self, frame: Frame) -> None:
        path = self.trial_dir / "steps" / f"{frame.index:04d}" / "observation.json"
        record = _read_json(path)
        if record:
            record["visual_delta"] = {
                "phase": frame.phase,
                "stalled": frame.stalled,
                "stop_reason": frame.stop_reason,
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def records(self, success: bool | None = None) -> list[dict[str, Any]]:
        """PEAR records of the trial so far (nothing written)."""

        episode = Episode(
            path=self.trial_dir,
            task=self.task or self._instruction or "complete the task",
            success=success,
            frames=list(self.frames),
            calls=[],
            result={"environment_success": success, "env_steps": self._actions},
        )
        return episode_records(episode, self.options, self.out_dir)

    def on_trial_end(self, result: Any = None) -> list[dict[str, Any]]:
        """Flush the trial to JSONL and start the next one; returns the records written."""

        if self._pending and self._actions:
            with contextlib.suppress(Exception):
                self._capture(self._env.get_obs())
        for frame in self.frames:
            if frame.stalled:
                self._rewrite_stop(frame)
        if isinstance(result, Mapping):
            success = result.get("environment_success", result.get("success"))
        else:
            success = result
        success = bool(success) if success is not None else None
        records = self.records(success)
        _write_json(self.trial_dir / "result.json", {
            "environment_success": success,
            "env_steps": self._actions,
            "task_language": self.task or self._instruction,
        })
        if records:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.trial += 1
        self._reset_trial()
        return records

    flush = on_trial_end


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codeaspolicy.visual_delta_distill",
        description="Distil CodeAsPolicy episodes into PEAR Visual-Delta JSONL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="convert episode directories to JSONL")
    ex.add_argument("paths", nargs="+", type=Path, help="episode dirs or run roots")
    ex.add_argument("-o", "--out", type=Path, required=True, help="output .jsonl")
    ex.add_argument("--format", choices=("sft", "rl"), default="sft")
    ex.add_argument("--frame", choices=("world", "tcp"), default="world")
    ex.add_argument("--arm", choices=("auto", "left", "right", "both"), default="auto")
    ex.add_argument("--success-only", action="store_true")
    ex.add_argument("--last-mile", action="store_true", help="keep contact phases only")
    ex.add_argument("--phases", default=None, help="comma-separated contact phases")
    ex.add_argument("--radius-m", type=float, default=None)
    ex.add_argument("--min-turns", type=int, default=1)
    ex.add_argument("--crop", action="store_true", help="add wrist ROI crops")
    ex.add_argument("--no-images", action="store_true")
    ex.add_argument("--no-think", action="store_true")
    ex.add_argument("--max-linear-m", type=float, default=CoalesceConfig.max_linear_m)
    ex.add_argument("--max-angular-deg", type=float, default=CoalesceConfig.max_angular_deg)
    ex.add_argument("--min-linear-m", type=float, default=CoalesceConfig.min_linear_m)
    st = sub.add_parser("stats", help="summarise a JSONL dataset or episode directories")
    st.add_argument("paths", nargs="+", type=Path, help=".jsonl files or episode dirs")
    st.add_argument("--frame", choices=("world", "tcp"), default="world")
    return parser


def _options(args: argparse.Namespace) -> ExtractOptions:
    phases = (
        frozenset(p.strip() for p in args.phases.split(",") if p.strip())
        if args.phases
        else CONTACT_PHASES
    )
    return ExtractOptions(
        success_only=args.success_only,
        last_mile=args.last_mile,
        phases=phases,
        radius_m=args.radius_m,
        arm=args.arm,
        min_turns=args.min_turns,
        format=args.format,
        crop=args.crop,
        coalesce=CoalesceConfig(
            max_linear_m=args.max_linear_m,
            max_angular_deg=args.max_angular_deg,
            min_linear_m=args.min_linear_m,
            frame=args.frame,
        ),
        dataset=DatasetOptions(
            frame=args.frame, think=not args.no_think, images=not args.no_images
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        counts = extract(args.paths, args.out, _options(args))
        print(json.dumps(dict(counts, out=str(args.out))))
        return 0
    records: list[dict[str, Any]] = []
    episodes = 0
    native_total = 0
    for path in args.paths:
        if path.is_file():
            records.extend(read_jsonl(path))
            continue
        options = ExtractOptions(
            coalesce=CoalesceConfig(frame=args.frame),
            dataset=DatasetOptions(frame=args.frame, images=False),
        )
        for directory in find_episodes([path]):
            episode = load_episode(directory)
            episodes += 1
            native_total += episode.native_steps or 0
            records.extend(episode_records(episode, options))
    summary = stats(records)
    if episodes:
        summary["episodes"] = episodes
        summary["episode_native_steps"] = native_total
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "ArmPose",
    "CAMERAS",
    "CONTACT_PHASES",
    "CoalesceConfig",
    "DatasetOptions",
    "Episode",
    "ExtractOptions",
    "Frame",
    "MacroStep",
    "PROTOCOL",
    "VisualDeltaEnvRecorder",
    "angles",
    "apply_delta",
    "arm_pose",
    "build_conversation",
    "coalesce",
    "crop_png",
    "decode_png",
    "ee_command_to_pear",
    "encode_png",
    "episode_records",
    "extract",
    "find_episodes",
    "load_episode",
    "main",
    "orientation",
    "pear_gripper_to_robodojo",
    "pear_to_ee_command",
    "pose_delta",
    "robodojo_gripper_to_pear",
    "rot_from_quat",
    "state_json",
    "stats",
    "system_prompt",
    "user_prompt",
    "width_cm",
]


if __name__ == "__main__":
    sys.exit(main())
