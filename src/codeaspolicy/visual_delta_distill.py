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
* :func:`episode_records` reads a finished episode directory (``result.json``, ``card.json``,
  ``calls.jsonl``, ``steps/NNNN/observation.json`` + camera PNGs) and builds one SFT sample
  (or RL prompt / solution) per turn, rendered exactly like the requests of the online loop
  (``visual_delta.SYSTEM_PROMPT`` / ``build_user_prompt``, the same views and wrist crops,
  ``format_feedback``-shaped feedback), optionally restricted to last-mile contact phases.
* :class:`VisualDeltaEnvRecorder` wraps a live env (``step`` / ``get_obs`` / ``get_status``)
  and flushes the same JSONL when a trial ends.
* ``python -m codeaspolicy.visual_delta_distill extract|stats`` is the command line.

Protocol v1 conventions (shared with :mod:`codeaspolicy.visual_delta`):

* ``x, y, z`` deltas are centimetres of the TCP grasp point (link6 origin plus
  ``TCP_OFFSET`` = 0.145 m along link6 +x), ``next - current``.  ``frame="tcp"`` expresses
  the world delta in the current link6 axes (+x pointing, +y finger closing, +z normal), and
  its angles are the relative rotation ``R0^T R1 = Rz(yaw) Ry(pitch) Rx(roll)``.
* ``roll, pitch, yaw`` deltas are radians: the wrapped difference of
  ``robodojo_kinematics.angles()`` (pitch, yaw, roll in degrees); an executor applies
  ``target = current + delta`` through ``kin.orientation(pitch, yaw, roll)``; beyond 85
  degrees of pitch, roll folds into ``yaw - sign(pitch) * roll`` (:func:`canonical_angles`).
* Gripper action ``width`` is a switch: ``+1`` open, ``-1`` close, ``0`` keep.  RoboDojo's
  internal codes are ``OPEN = -1`` / ``CLOSE = 1`` and its ``ee_joint_state`` command is
  ``1.0`` open / ``0.0`` closed (the host reads ``<= 0.05`` as closed).  Observed width
  is the jaw span in cm,
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
#: The host reads an ee command at or below this as closed (``robodojo_tools``).
EE_CLOSED_MAX = 0.05
#: Opening above which a closed gripper is holding something (``HOLD_MIN_OPENING``).
HOLD_MIN_OPENING = 0.01

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


def robodojo_gripper_to_pear(code: float | None) -> int:
    """RoboDojo ``OPEN=-1`` / ``CLOSE=1`` / ``KEEP=0`` (or ``None``) to the PEAR switch."""

    if code is None:
        return PEAR_KEEP
    code = float(code)
    if code < 0:
        return PEAR_OPEN
    if code > 0:
        return PEAR_CLOSE
    return PEAR_KEEP


def ee_command_to_pear(command: float | None, previous: float | None = None) -> int:
    """An ``ee_joint_state`` command level (1.0 open, 0.0 closed) as a PEAR switch.

    A command at or below :data:`EE_CLOSED_MAX` is a close (the host's own threshold),
    anything above it an open.  With ``previous`` given, an unchanged open/closed state is
    ``0`` (keep): PEAR actions are switches, the level itself is state.
    """

    if command is None:
        return PEAR_KEEP
    closed = ee_closed(command)
    if previous is not None and ee_closed(previous) == closed:
        return PEAR_KEEP
    return PEAR_CLOSE if closed else PEAR_OPEN


def ee_closed(command: float) -> bool:
    """Whether an ``ee_joint_state`` command closes the gripper (``<= 0.05``, as the host)."""

    return float(command) <= EE_CLOSED_MAX


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

    pitch, yaw, roll = canonical_angles(*pose.angles_deg)
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


#: Beyond this pitch (degrees, either sign) the gripper counts as near vertical: yaw and roll
#: then turn about nearly the same axis and ``angles()`` may trade one for the other between
#: two close poses.  Shared with :mod:`codeaspolicy.visual_delta`.
NEAR_VERTICAL_DEG = 85.0


def _sign(value: float) -> float:
    return 1.0 if value >= 0 else -1.0


def canonical_angles(pitch: float, yaw: float, roll: float) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees with near-vertical roll folded into yaw.

    At pitch +90 ``orientation`` is ``Rz(90 + yaw - roll) Ry(90)``, at -90
    ``Rz(90 + yaw + roll) Ry(-90)``: only ``yaw - sign(pitch) * roll`` is well defined.  Beyond
    :data:`NEAR_VERTICAL_DEG` that is reported as the yaw, with roll 0.
    """

    if abs(pitch) > NEAR_VERTICAL_DEG:
        return pitch, wrap_deg(yaw - _sign(pitch) * roll), 0.0
    return pitch, yaw, roll


def rotation_error(a: Matrix, b: Matrix) -> float:
    """Geodesic angle between two rotations, degrees (``kin.rotation_error``)."""

    c = (sum(a[i][j] * b[i][j] for i in range(3) for j in range(3)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _rot(pose: ArmPose) -> Matrix:
    return [list(r) for r in pose.rot]


def angle_delta(current: ArmPose, target: ArmPose) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees to add to ``current``'s :func:`angles` to reach ``target``.

    The wrapped difference of :func:`angles`; when both poses are beyond
    :data:`NEAR_VERTICAL_DEG`, the difference of :func:`canonical_angles` instead (roll 0,
    the well-defined ``yaw - sign(pitch) * roll`` in yaw), the rule shared with
    :mod:`codeaspolicy.visual_delta`.  Applied as ``orientation(current angles + delta)``.
    """

    p0, y0, r0 = current.angles_deg
    p1, y1, r1 = target.angles_deg
    if abs(p0) > NEAR_VERTICAL_DEG and abs(p1) > NEAR_VERTICAL_DEG:
        dyaw = wrap_deg((y1 - _sign(p1) * r1) - (y0 - _sign(p0) * r0))
        return p1 - p0, dyaw, 0.0
    return wrap_deg(p1 - p0), wrap_deg(y1 - y0), wrap_deg(r1 - r0)


def zyx_deg(rot: Matrix) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees of ``rot = Rz(yaw) Ry(pitch) Rx(roll)`` (host ``_zyx_deg``)."""

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -rot[2][0]))))
    yaw = math.degrees(math.atan2(rot[1][0], rot[0][0]))
    roll = math.degrees(math.atan2(rot[2][1], rot[2][2]))
    return yaw, pitch, roll


def rotation_delta(
    current: ArmPose, target: ArmPose, frame: str = "world"
) -> tuple[float, float, float]:
    """(pitch, yaw, roll) degrees of the delta in ``frame`` (see :func:`pose_delta`)."""

    if frame == "tcp":
        yaw, pitch, roll = zyx_deg(_matmul(_transpose(_rot(current)), _rot(target)))
        return pitch, yaw, roll
    return angle_delta(current, target)


def pose_delta(
    current: ArmPose,
    target: ArmPose,
    frame: str = "world",
    linear_decimals: int = 2,
    angular_decimals: int = 3,
) -> dict[str, float]:
    """PEAR arm delta ``target - current``: cm for xyz, rad for roll/pitch/yaw.

    ``world``: TCP displacement in the table frame, angles by :func:`angle_delta`.  ``tcp``:
    displacement in the current gripper axes and the relative rotation ``R0^T R1`` as
    ``Rz(yaw) Ry(pitch) Rx(roll)`` about those axes (the executor's ``target_pose``).
    """

    if frame not in ("world", "tcp"):
        raise ValueError(f"frame must be 'world' or 'tcp', got {frame!r}")
    d = [target.tcp[i] - current.tcp[i] for i in range(3)]
    if frame == "tcp":
        d = _matvec(_transpose(_rot(current)), d)
    dp, dy, dr = rotation_delta(current, target, frame)
    out = {axis: round(v * 100.0, linear_decimals) + 0.0 for axis, v in zip("xyz", d)}
    out["roll"] = round(math.radians(dr), angular_decimals) + 0.0
    out["pitch"] = round(math.radians(dp), angular_decimals) + 0.0
    out["yaw"] = round(math.radians(dy), angular_decimals) + 0.0
    return out


def apply_delta(current: ArmPose, delta: Mapping[str, float], frame: str = "world") -> ArmPose:
    """Inverse of :func:`pose_delta` (what an executor targets), for round-trip checks."""

    d = [float(delta.get(axis, 0.0)) / 100.0 for axis in "xyz"]
    dp, dy, dr = (math.degrees(float(delta.get(k, 0.0))) for k in ("pitch", "yaw", "roll"))
    if frame == "tcp":
        d = _matvec(_rot(current), d)
        rot = _matmul(
            _matmul(_matmul(_rot(current), _rz(math.radians(dy))), _ry(math.radians(dp))),
            _rx(math.radians(dr)),
        )
    else:
        p, y, r = current.angles_deg
        rot = orientation(p + dp, y + dy, r + dr)
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
    """Thresholds of :func:`coalesce`.

    The caps default to ``visual_align_and_insert``'s per-round limits (3 cm, 15 degrees;
    a free ``nudge`` allows 6 cm, 20 degrees) and are the limits rendered in every prompt.
    """

    max_linear_m: float = 0.03
    max_angular_deg: float = 15.0
    min_linear_m: float = 0.001
    min_angular_deg: float = 0.5
    split_on_phase: bool = True
    opening_switch: float = 0.05
    frame: str = "world"
    #: Macro steps above the caps (one recorded transition already larger than a nudge can
    #: execute): "drop" (flag them; extraction ends the conversation there), "split" (N equal
    #: sub-deltas within the caps) or "keep" (flag only).
    oversize: str = "drop"


OVERSIZE_MODES = ("drop", "split", "keep")


@dataclass
class MacroStep:
    """A PEAR macro delta: from ``frames[start]`` to ``frames[end]``.

    Parts of a split oversize step share ``start`` / ``end`` and carry ``part = (k, n)``;
    ``state_poses`` then holds the interpolated poses the part starts from.
    """

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
    oversize: bool = False
    part: tuple[int, int] | None = None
    state_poses: dict[str, ArmPose] | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
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
            "oversize": self.oversize,
        }
        if self.part is not None:
            out["part"] = list(self.part)
        return out


def _motion(a: ArmPose, b: ArmPose, frame: str = "world") -> tuple[float, float]:
    """Linear (m) and angular (deg) motion between two poses.

    The angle is the larger of the geodesic rotation and the largest angle of the delta in
    ``frame`` (``nudge`` checks its limit per angle), so a step within the caps is a step one
    nudge executes.
    """

    turn = max(abs(v) for v in rotation_delta(a, b, frame))
    return math.dist(a.tcp, b.tcp), max(rotation_error(_rot(a), _rot(b)), turn)


def _arms_of(arm: str) -> tuple[str, ...]:
    return ARMS if arm == "both" else (arm,)


def gripper_switch(a: Frame, b: Frame, side: str, opening_switch: float = 0.05) -> int:
    """PEAR switch for ``side`` between two frames, from commands (or openings if absent)."""

    ca, cb = a.gripper.get(side), b.gripper.get(side)
    if ca is not None and cb is not None:
        return ee_command_to_pear(cb, ca)
    if cb is not None and ca is None and ee_closed(cb):
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
            lin, ang = _motion(a, b, config.frame)
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
            lin, ang = _motion(pa, pb, config.frame)
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
    Motion is measured as TCP distance and geodesic rotation angle.  A step still above the
    caps (one recorded transition too large for a nudge) is flagged ``oversize``, and split
    into parts within the caps when ``config.oversize == "split"``.
    """

    if config.oversize not in OVERSIZE_MODES:
        raise ValueError(f"oversize must be one of {OVERSIZE_MODES}, got {config.oversize!r}")
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
                lv, av = _motion(a, b, config.frame)
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
            close = _over_caps(lin, ang, config)
        if not close:
            continue
        action, linear, angular, switched = _action(frames[start], cur, arm, config)
        still = linear < config.min_linear_m and angular < config.min_angular_deg
        if not (still and not switched and not cur.stalled):
            step = MacroStep(
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
                oversize=_over_caps(linear, angular, config),
            )
            if step.oversize and config.oversize == "split":
                steps.extend(split_oversize(step, frames, config))
            else:
                steps.append(step)
        start, native = t, 0
    return steps


def _over_caps(linear: float, angular: float, config: CoalesceConfig) -> bool:
    return linear > config.max_linear_m + 1e-6 or angular > config.max_angular_deg + 1e-6


def _interpolate(a: ArmPose, b: ArmPose, fraction: float) -> ArmPose:
    """The pose ``fraction`` of the way from ``a`` to ``b`` (linear TCP, linear angles)."""

    if fraction <= 0.0:
        return a
    if fraction >= 1.0:
        return b
    full = pose_delta(a, b, "world", linear_decimals=9, angular_decimals=12)
    return apply_delta(a, {k: v * fraction for k, v in full.items()}, "world")


def split_oversize(
    step: MacroStep, frames: Sequence[Frame], config: CoalesceConfig = CoalesceConfig()
) -> list[MacroStep]:
    """Split a macro step above the caps into ``n`` equal sub-deltas within them.

    Every part is shown with the start frame's images (no camera saw the intermediate poses)
    and its interpolated state; the gripper switch and the stop belong to the last part.
    """

    n = max(
        1,
        math.ceil(step.linear_m / config.max_linear_m - 1e-9),
        math.ceil(step.angular_deg / config.max_angular_deg - 1e-9),
    )
    first, last = frames[step.start], frames[step.end]
    sides = [s for s in _arms_of(step.arm) if s in first.poses and s in last.poses]
    knots = []
    for k in range(n + 1):
        poses = dict(first.poses if k < n else last.poses)
        for side in sides:
            poses[side] = _interpolate(first.poses[side], last.poses[side], k / n)
        knots.append(poses)
    parts = []
    for k in range(n):
        final = k == n - 1
        action: dict[str, Any] = {}
        linear = angular = 0.0
        for side in _arms_of(step.arm):
            if step.arm == "both":
                arm_key, grip_key = f"{side}_arm", f"{side}_gripper"
            else:
                arm_key, grip_key = "arm", "gripper"
            if side in sides:
                a, b = knots[k][side], knots[k + 1][side]
                action[arm_key] = pose_delta(a, b, config.frame)
                lin, ang = _motion(a, b, config.frame)
                linear, angular = max(linear, lin), max(angular, ang)
            else:
                action[arm_key] = dict(step.action[arm_key])
            switch = step.action.get(grip_key, {}).get("width", 0) if final else 0
            action[grip_key] = {"width": switch}
        parts.append(
            MacroStep(
                start=step.start,
                end=step.end,
                arm=step.arm,
                action=action,
                phase=step.phase,
                native_steps=step.native_steps // n + (step.native_steps % n if final else 0),
                status=step.status if final else "reached",
                stop_reason=step.stop_reason if final else None,
                linear_m=linear,
                angular_deg=angular,
                oversize=_over_caps(linear, angular, config),
                part=(k, n),
                state_poses=knots[k],
            )
        )
    return parts


def split_runs(steps: Sequence[MacroStep]) -> list[list[MacroStep]]:
    """Runs of consecutive steps without an oversize step (dropped ones end a run)."""

    runs: list[list[MacroStep]] = [[]]
    for step in steps:
        if step.oversize:
            runs.append([])
        else:
            runs[-1].append(step)
    return [run for run in runs if run]


# ---------------------------------------------------------------------------
# PEAR prompts and conversations
# ---------------------------------------------------------------------------

IMAGE_TOKEN = "<image>"

# Training samples must read exactly like the requests ``visual_align_and_insert`` sends, so
# the prompts are taken from :mod:`codeaspolicy.visual_delta` when it is importable; these are
# verbatim copies for environments without it (a test asserts they are identical).
_SYSTEM_PROMPT = """You control the last few centimetres of a robot manipulation step.
Each turn you see labelled camera views and the current proprioceptive state, and you
answer with ONE small relative motion of the grasp point (PEAR Visual-Delta v1).

Action JSON (relative deltas; write 0 for every unchanged field):
- single arm: {"arm": {"x": dx, "y": dy, "z": dz, "roll": dr, "pitch": dp, "yaw": dw},
  "gripper": {"width": g}}
- both arms: {"left_arm": {...}, "right_arm": {...}, "left_gripper": {"width": g},
  "right_gripper": {"width": g}}
- optional "done": true when the alignment / insertion is complete.
Units: x, y, z in centimetres; roll, pitch, yaw in radians. Gripper width in the
action is a switch: +1 opens, -1 closes, 0 keeps. In the state, gripper width is the
measured jaw span in centimetres.
Frames: "world" = table frame (x across the table to the robot's right, y away from
the robot, z up; angle deltas add to the gripper's pitch / yaw / roll). "tcp" = grasp
frame (+x pointing direction, +y finger-closing axis, +z normal).
Feedback after each move: "reached" (the move completed), "stopped" (contact or stall
before the target; the arm is in contact or blocked), "out_of_reach" (nothing moved).
Prefer small moves (0.2 - 2 cm, <= 0.15 rad) near contact.

Answer format: optionally <think>brief reasoning</think>, then exactly one fenced
json code block containing the action object and nothing else."""

_ENV_PROMPT_TEMPLATE = """Goal: {goal}
Arm mode: {arm}   Frame: {frame}   Round: {round}/{max_rounds}   Steps left: {steps_left}
Limits per move: {max_linear_cm} cm, {max_angular_rad} rad
Images: {images}
State: {state}
Last execution: {feedback}
Reply with the next delta action."""


def _online_module() -> Any:
    try:
        from . import visual_delta as online
    except Exception:  # noqa: BLE001 - absent, or its optional dependencies are missing
        return None
    if hasattr(online, "SYSTEM_PROMPT") and hasattr(online, "build_user_prompt"):
        return online
    return None


def system_prompt() -> str:
    """The system prompt of the online loop (``visual_delta.SYSTEM_PROMPT``)."""

    online = _online_module()
    return online.SYSTEM_PROMPT if online is not None else _SYSTEM_PROMPT


def _fallback_user_prompt(
    *,
    goal: str,
    arm: str,
    frame: str,
    state: Mapping[str, Any],
    feedback: Mapping[str, Any] | None,
    round_index: int,
    max_rounds: int,
    steps_left: int,
    image_labels: Sequence[str] = (),
    max_linear_m: float = 0.03,
    max_angular_deg: float = 15.0,
) -> str:
    return _ENV_PROMPT_TEMPLATE.format(
        goal=goal.strip() or "(none)",
        arm=arm,
        frame=frame,
        round=round_index + 1,
        max_rounds=max_rounds,
        steps_left=steps_left,
        max_linear_cm=round(max_linear_m * 100.0, 1),
        max_angular_rad=round(math.radians(max_angular_deg), 2),
        images=" ".join(f"[{label}]" for label in image_labels) or "(none)",
        state=json.dumps(state, separators=(", ", ": ")),
        feedback=json.dumps(feedback, separators=(", ", ": ")) if feedback else "none (first turn)",
    )


def build_user_prompt(**kwargs: Any) -> str:
    """The per-turn text of the online loop (``visual_delta.build_user_prompt``)."""

    online = _online_module()
    if online is not None:
        return online.build_user_prompt(**kwargs)
    return _fallback_user_prompt(**kwargs)


def user_content(images: Sequence[tuple[str, str]], text: str) -> str:
    """A user turn as one string: ``[label]<image>`` per view, then the text.

    Online the content is the list ``[label]``, image, ..., text; chat templates join list
    items without separators, and ``<image>`` is the multimodal placeholder of LLaMA-Factory
    and ms-swift.
    """

    return "".join(f"[{label}]{IMAGE_TOKEN}" for label, _ in images) + text


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


def default_cameras(arm: str) -> tuple[str, ...]:
    """The views the online loop sends: head + the active wrist, all three for both arms."""

    return CAMERAS if arm == "both" else ("cam_head", WRIST_CAMERA[arm])


def frame_images(
    frame: Frame,
    cameras: Sequence[str],
    crop: Callable[[str, str], str | None] | None = None,
) -> list[tuple[str, str]]:
    """Ordered ``(label, path)`` images of a frame, each wrist crop right after its camera."""

    out = []
    for cam in cameras:
        path = frame.images.get(cam)
        if not path:
            continue
        out.append((cam, path))
        if crop is not None and cam.endswith("_wrist"):
            cropped = crop(cam, path)
            if cropped:
                out.append((f"{cam}_crop", cropped))
    return out


@dataclass(frozen=True)
class DatasetOptions:
    """How samples are rendered (defaults mirror ``visual_align_and_insert``)."""

    frame: str = "world"
    think: bool = True
    images: bool = True
    cameras: tuple[str, ...] | None = None  # None: default_cameras(arm), as online
    done_turn: bool = True  # a final {"done": true} turn for segments ending a success
    #: Previous (user text, assistant) pairs in each sample, like
    #: ``OpenAIVisualDeltaClient(history_turns=...)``: 0 (online default) or -1 for all.
    history_turns: int = 0
    max_rounds: int = 8
    max_steps: int = 40
    goal: str | None = None  # default: the episode's task


def _holding(frame: Frame, side: str) -> bool:
    command = frame.gripper.get(side)
    pose = frame.poses.get(side)
    opening = pose.opening if pose is not None else None
    return bool(
        command is not None
        and ee_closed(command)
        and opening is not None
        and opening > HOLD_MIN_OPENING
    )


def feedback_json(step: MacroStep, frames: Sequence[Frame]) -> dict[str, Any]:
    """``visual_delta.format_feedback`` of a macro step, as the next turn reports it."""

    achieved = {
        key: {axis: round(float(v[axis]), 1 if axis in "xyz" else 2) for axis in AXES}
        for key, v in step.action.items()
        if key.endswith("arm")
    }
    out: dict[str, Any] = {
        "status": step.status,
        "reason": step.stop_reason,
        "steps_used": int(step.native_steps),
        "achieved": achieved,
    }
    if step.status == "stopped":
        out["stalled"] = True
    # a split step switches its gripper on its last part
    finished = step.part is None or step.part[0] == step.part[1] - 1
    after = frames[step.end] if finished else frames[step.start]
    holding = {side: _holding(after, side) for side in _arms_of(step.arm)}
    out["holding"] = holding if step.arm == "both" else holding[step.arm]
    return out


def build_samples(
    frames: Sequence[Frame],
    steps: Sequence[MacroStep],
    goal: str,
    options: DatasetOptions = DatasetOptions(),
    success: bool = False,
    crop: Callable[[str, str], str | None] | None = None,
    max_linear_m: float = CoalesceConfig.max_linear_m,
    max_angular_deg: float = CoalesceConfig.max_angular_deg,
) -> list[dict[str, Any]]:
    """One training sample per turn, rendered as the online loop renders its requests.

    Each sample has ``messages`` (system, optional text-only history, the user turn with
    ``<image>`` placeholders, the reference assistant reply), ``images`` (paths in
    placeholder order), ``action`` and the macro ``step`` (``None`` for the done turn).
    The feedback of a turn is the previous step's outcome; ``Steps left`` counts down the
    native steps the steps used.
    """

    if not steps:
        return []
    arm = steps[0].arm
    cameras = options.cameras or default_cameras(arm)
    system = system_prompt()
    turns: list[tuple[Frame, Mapping[str, ArmPose] | None, MacroStep | None]] = [
        (frames[s.start], s.state_poses, s) for s in steps
    ]
    if success and options.done_turn:
        turns.append((frames[steps[-1].end], None, None))
    max_rounds = max(options.max_rounds, len(turns))
    budget = max(options.max_steps, sum(s.native_steps for s in steps) + 1)
    history: list[tuple[str, str]] = []
    feedback: dict[str, Any] | None = None
    used = 0
    out = []
    for index, (frame, poses, step) in enumerate(turns):
        shown = frame_images(frame, cameras, crop) if options.images else []
        text = build_user_prompt(
            goal=goal,
            arm=arm,
            frame=options.frame,
            state=state_json(poses or frame.poses, arm),
            feedback=feedback,
            round_index=index,
            max_rounds=max_rounds,
            steps_left=budget - used,
            image_labels=[label for label, _ in shown],
            max_linear_m=max_linear_m,
            max_angular_deg=max_angular_deg,
        )
        action = step.action if step is not None else dict(zero_action(arm), done=True)
        reply = assistant_reply(action, options.frame, step.phase if step else None, options.think)
        messages = [{"role": "system", "content": system}]
        if options.history_turns:
            keep = history if options.history_turns < 0 else history[-options.history_turns :]
            for user_text, assistant_text in keep:
                messages.append({"role": "user", "content": user_text})
                messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": user_content(shown, text)})
        messages.append({"role": "assistant", "content": reply})
        out.append(
            {
                "turn": index,
                "messages": messages,
                "images": [path for _, path in shown],
                "action": action,
                "step": step,
            }
        )
        history.append((text, reply))
        if step is not None:
            feedback = feedback_json(step, frames)
            used += step.native_steps
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


#: Fraction of a wrist view the online crop keeps (``robodojo_visual_delta.CROP_FRACTION``).
CROP_FRACTION = 0.5


def _center_box(width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    ch, cw = max(1, round(height * fraction)), max(1, round(width * fraction))
    top, left = (height - ch) // 2, (width - cw) // 2
    return left, top, left + cw, top + ch


def crop_center_png(src: Path, dst: Path, fraction: float = CROP_FRACTION) -> Path:
    """The online wrist crop: the central ``fraction`` resized back to full size (2x zoom).

    Same box and bicubic resampling as ``robodojo_visual_delta.crop_center`` (Pillow);
    without Pillow, nearest-neighbour upsampling.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        with Image.open(src) as img:
            rgb = img.convert("RGB")
            box = _center_box(rgb.width, rgb.height, fraction)
            rgb.crop(box).resize((rgb.width, rgb.height), Image.BICUBIC).save(dst, format="PNG")
    except ImportError:
        rows = decode_png(Path(src).read_bytes())
        height, width = len(rows), len(rows[0])
        left, top, right, bottom = _center_box(width, height, fraction)
        cw, ch = right - left, bottom - top
        dst.write_bytes(
            encode_png(
                [
                    [rows[top + y * ch // height][left + x * cw // width] for x in range(width)]
                    for y in range(height)
                ]
            )
        )
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
    crop: bool = True  # zoomed wrist crops, as visual_align_and_insert(crop_wrist=True)
    crop_fraction: float = CROP_FRACTION
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
    out_dir: Path | None, episode: Path, fraction: float
) -> Callable[[str, str], str | None]:
    """Writes crops to ``<out_dir>/crops/<episode>/``, else ``<episode>/visual_delta_crops/``."""

    if out_dir is not None:
        base = out_dir / "crops" / episode.name
    else:
        base = episode / "visual_delta_crops"

    def crop(cam: str, path: str) -> str | None:
        src = Path(path)
        dst = base / f"{src.parent.name}_{cam}_crop.png"
        try:
            return str(crop_center_png(src, dst, fraction).resolve())
        except (OSError, ValueError, zlib.error):
            return None

    return crop


def episode_records(
    episode: Episode, options: ExtractOptions = ExtractOptions(), out_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Per-turn PEAR SFT samples (or RL prompt / solution samples) of one episode."""

    if options.success_only and not episode.success:
        return []
    frames = episode.frames
    if len(frames) < 2:
        return []
    arm = select_arm(frames, options.coalesce) if options.arm == "auto" else options.arm
    crop = _cropper(out_dir, episode.path, options.crop_fraction) if options.crop else None
    records: list[dict[str, Any]] = []
    for first, last in segments(frames, options, arm):
        window = frames[first : last + 1]
        steps = coalesce(window, arm, options.coalesce)
        dropped = sum(1 for s in steps if s.oversize) if options.coalesce.oversize == "drop" else 0
        runs = split_runs(steps) if options.coalesce.oversize == "drop" else [steps]
        for run in runs:
            if len(run) < options.min_turns:
                continue
            # the segment's dropped oversize steps are counted once, on its first record
            number = len({r["run"] for r in records})
            records.extend(
                _run_records(episode, options, window, first, run, arm, crop, number, dropped)
            )
            dropped = 0
    return records


def _run_records(
    episode: Episode,
    options: ExtractOptions,
    window: Sequence[Frame],
    offset: int,
    steps: Sequence[MacroStep],
    arm: str,
    crop: Callable[[str, str], str | None] | None,
    number: int,
    dropped: int,
) -> list[dict[str, Any]]:
    phases = sorted({s.phase for s in steps if s.phase})
    goal = options.dataset.goal or episode.task
    first, last = offset + steps[0].start, offset + steps[-1].end
    ends_episode = last == len(episode.frames) - 1
    samples = build_samples(
        window,
        steps,
        goal,
        options.dataset,
        bool(episode.success) and ends_episode,
        crop,
        options.coalesce.max_linear_m,
        options.coalesce.max_angular_deg,
    )
    run_id = f"{episode.path.name}#{number}"
    out = []
    for sample in samples:
        step = sample["step"]
        entry = None
        if step is not None:
            entry = step.to_dict()
            entry["start"], entry["end"] = step.start + offset, step.end + offset
        record = {
            "id": f"{run_id}:{sample['turn']}",
            "run": run_id,
            "turn": sample["turn"],
            "turns": len(samples),
            "protocol": PROTOCOL,
            "task": episode.task,
            "goal": goal,
            "arm": arm,
            "frame": options.dataset.frame,
            "success": episode.success,
            "source": {
                "episode": str(episode.path),
                "frames": [first, last],
                "frame": offset + (step.start if step is not None else steps[-1].end),
            },
            "phases": phases,
            "limits": {
                "max_linear_m": options.coalesce.max_linear_m,
                "max_angular_deg": options.coalesce.max_angular_deg,
            },
            "step": entry,
            "oversize_dropped": dropped if sample["turn"] == 0 else 0,
        }
        if options.format == "rl":
            messages = sample["messages"]
            record.update(
                prompt=messages[:-1],
                images=sample["images"],
                solution=sample["action"],
                reference=messages[-1]["content"],
            )
        else:
            record.update(messages=sample["messages"], images=sample["images"])
        out.append(record)
    return out


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
                if record.get("step"):
                    counts["macro_steps"] += 1
                    counts["native_steps"] += int(record["step"]["native_steps"])
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

    runs: set[str] = set()
    macro = native = stopped = oversize = dropped = samples = 0
    linear: list[float] = []
    angular: list[float] = []
    per_axis: dict[str, list[float]] = {axis: [] for axis in AXES}
    switches = {"open": 0, "close": 0}
    phases: dict[str, int] = {}
    for record in records:
        samples += 1
        runs.add(str(record.get("run") or str(record.get("id", "")).split(":")[0]))
        dropped += int(record.get("oversize_dropped") or 0)
        step = record.get("step")
        if step:
            macro += 1
            native += int(step.get("native_steps") or 0)
            stopped += step.get("status") == "stopped"
            oversize += bool(step.get("oversize"))
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
        "samples": samples,
        "conversations": len(runs),
        "macro_steps": macro,
        "native_steps": native,
        "native_per_macro": round(native / macro, 3) if macro else None,
        "contact_stop_ratio": round(stopped / macro, 4) if macro else None,
        "oversize_steps": oversize,
        "oversize_dropped": dropped,
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
    ex.add_argument("--no-crop", action="store_true", help="no zoomed wrist crops")
    ex.add_argument("--history-turns", type=int, default=0,
                    help="previous text turns per sample (-1 all), like history_turns online")
    ex.add_argument("--goal", default=None, help="goal text (default: the episode's task)")
    ex.add_argument("--max-rounds", type=int, default=DatasetOptions.max_rounds)
    ex.add_argument("--max-steps", type=int, default=DatasetOptions.max_steps)
    ex.add_argument("--no-images", action="store_true")
    ex.add_argument("--no-think", action="store_true")
    ex.add_argument("--max-linear-m", type=float, default=CoalesceConfig.max_linear_m)
    ex.add_argument("--max-angular-deg", type=float, default=CoalesceConfig.max_angular_deg)
    ex.add_argument("--min-linear-m", type=float, default=CoalesceConfig.min_linear_m)
    ex.add_argument(
        "--oversize", choices=OVERSIZE_MODES, default=CoalesceConfig.oversize,
        help="macro steps above the caps: drop (default), split, keep",
    )
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
        crop=not args.no_crop,
        coalesce=CoalesceConfig(
            max_linear_m=args.max_linear_m,
            max_angular_deg=args.max_angular_deg,
            min_linear_m=args.min_linear_m,
            frame=args.frame,
            oversize=args.oversize,
        ),
        dataset=DatasetOptions(
            frame=args.frame,
            think=not args.no_think,
            images=not args.no_images,
            history_turns=args.history_turns,
            max_rounds=args.max_rounds,
            max_steps=args.max_steps,
            goal=args.goal,
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
    "angle_delta",
    "build_samples",
    "build_user_prompt",
    "canonical_angles",
    "coalesce",
    "crop_center_png",
    "crop_png",
    "default_cameras",
    "decode_png",
    "ee_closed",
    "ee_command_to_pear",
    "feedback_json",
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
    "rotation_error",
    "rot_from_quat",
    "split_oversize",
    "split_runs",
    "state_json",
    "stats",
    "system_prompt",
    "user_content",
    "width_cm",
    "zyx_deg",
]


if __name__ == "__main__":
    sys.exit(main())
