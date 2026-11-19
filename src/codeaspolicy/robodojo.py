# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""RoboDojo robot API for code policies.

Perception (``segment``, ``back_project``) and the frozen Pi0.5 policy
(``pi0_pick``) are RPent's RoboDojo tools (``robots/robodojo/tools.py``) with
RPent's argument names and defaults.  Motion and grasping are this library's
own (``hosts/robodojo_tools.py``, registered on RPent's toolkit at boot, so they
are executed and recorded like RPent's tools): RPent's toolset moves an arm by
position only and its IK lets the wrist drift, which leaves a program unable to
orient the fingers for a grasp or to bring an arm back to its starting pose,
and RoboDojo counts 45 of its 54 tasks as solved only with both arms back.
``move_to`` solves a full-pose IK in the host (``hosts/robodojo_kinematics.py``,
the benchmark's URDF fitted to the reported poses exactly) and moves the grasp
point along a straight line; ``set_gripper`` (a full stroke that reports what
it holds), ``locate`` (an object's 3D extent from its mask and the depth image)
and ``home`` complete the set.  Only what a program cannot compose from the
others is a primitive: a button press is a ``move_to`` past the surface, a
twist is a ``move_to`` with another ``roll``.  The calling convention is
the library's: a policy is an ordinary Python program that calls these methods
and branches on their typed return values.

RoboDojo is the one robot whose success flag RPent keeps from its planner:
official success is read by the runner after the planner has finished, and no
tool result carries it.  This class keeps that regime, so the code arm sees
what the tool-calling arm sees: it has no ``done`` (``ABSENT_MEMBERS``; the
attribute raises rather than answering a constant False that a
``while not robo.done`` loop would spin on), its ``State`` has no
``terminated`` and no result has a success-of-the-task field.  The host reads
the verdict from the backend when the episode is over.

The docstrings in this module are the model-facing documentation: the prompt
is generated from them (see :mod:`codeaspolicy.prompt`), so keep them accurate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ._backend import ToolError
from ._robot import RobotBase
from .coordination import CoordinationMixin, Pose
from .intent import IntentMixin
from .motion import MotionProfile, MotionWindow, resolve_motion
from .object_evidence import ObjectEvidenceMixin
from .perception import PerceptionResult
from .robots import RobotAdapter

#: Native actions one Pi0.5 chunk can execute (the policy's prediction horizon).
VLA_CHUNK_ACTIONS = 50
#: ``move_to`` has arrived when the grasp point is this close (metres) ...
MOVE_TOL_M = 0.01
#: ... and the gripper is turned less than this from the asked orientation (degrees).
MOVE_TOL_DEG = 5.0
#: The grasp point, between the finger pads, on the pointing axis: metres from the
#: end-effector link (RoboDojo's ``gripper_bias``).
TCP_OFFSET = 0.145
#: Native actions one ``home`` can execute before it returns.
HOME_MAX_ACTIONS = 25
#: ``home`` stops once every joint is this close to its starting angle (radians).
HOME_TOL_RAD = 0.01


# ---------------------------------------------------------------------------
# Result types (returned to the policy; all fields are plain Python values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmState:
    """One arm at a recorded step.

    tcp_pos: world grasp-point [x, y, z] (m), middle of finger pads, 0.145 m ahead of eef_pos.
    pitch / yaw / roll: gripper orientation (deg); pitch 0 level / 90 down, yaw heading (+90 left),
        roll about pointing axis (at pitch=90, yaw sets jaw-closing line, roll=0).
    eef_pos / eef_quat: wrist-flange position (m) and unit quaternion [w, x, y, z].
    gripper: measured opening (1.0 open, 0.0 closed; held width ~= 0.002 + 0.109 * gripper m).
    joints: six joint angles (rad); joints[5] rotates fingers about pointing axis.
    """

    tcp_pos: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    eef_pos: tuple[float, float, float]
    eef_quat: tuple[float, float, float, float]
    gripper: float
    joints: tuple[float, ...]


@dataclass(frozen=True)
class State:
    """Snapshot of both arms at one recorded step (0 = initial, ``state.arm(name)`` selects one)."""

    step: int
    task: str
    left: ArmState
    right: ArmState
    truncated: bool

    def arm(self, name: str) -> ArmState:
        if name == "left":
            return self.left
        if name == "right":
            return self.right
        raise ValueError("arm must be 'left' or 'right'")


@dataclass(frozen=True)
class Move:
    """Outcome of ``move_to`` (motion status, not task success).

    reached: within 0.01 m / 5 deg without tracking interruption.
    status: completed | rejected (0 steps) | partial.
    planned_path / path: "line" or "joint" (path is None if 0 steps).
    stop_reason / ik_error / note: refusal or stop detail; blocked flags >5 mm lag (not force).
    """

    arm: str
    target_xyz: tuple[float, float, float]
    start_xyz: tuple[float, float, float]
    final_xyz: tuple[float, float, float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    path: str | None
    ik_error: str | None
    note: str | None
    blocked: bool
    steps_used: int
    truncated: bool
    trajectory: dict | None = None
    status: str = "rejected"
    planned_path: str | None = None
    stop_reason: str | None = None
    tracking: dict | None = None


@dataclass(frozen=True)
class Home:
    """Outcome of ``home``: ``reached`` when every joint is within 0.01 rad of start."""

    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int
    truncated: bool
    stop_reason: str | None = None
    note: str | None = None
    clearance_retreat_m: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Gripper:
    """Outcome of ``set_gripper``.

    opening: 1.0 open, 0.0 closed. holding: opening > 0.01 after close (verify visually).
    width_m: estimated held width (m) after close, else None.
    """

    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float | None
    steps_used: int
    step: int | None
    step_limit: int | None
    truncated: bool


@dataclass(frozen=True)
class Pick:
    """Outcome of ``pi0_pick``. ``success`` is a provisional lift heuristic; verify in wrist view."""

    success: bool
    instruction: str
    chunks_used: int
    left: dict[str, float]
    right: dict[str, float]
    monitored_arm: str
    budget_exhausted: bool
    truncated: bool


@dataclass(frozen=True)
class Nudge:
    """Outcome of ``nudge``.

    status: completed | stopped (contact_or_stall) | partial | rejected (0 steps).
    achieved: measured ``dxyz`` (m) and ``dpitch``/``dyaw``/``droll`` (deg) in requested frame.
    stalled: kinematic progress stall (not force); inspect wrist view. ``arms`` holds both if arm="both".
    """

    arm: str
    frame: str
    status: str
    reached: bool
    stalled: bool
    stop_reason: str | None
    steps_used: int
    planned_steps: int
    final_xyz: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    dist_to_target_m: float
    orientation_error_deg: float
    achieved: dict[str, Any]
    holding: bool
    opening: float
    ik_error: str | None
    note: str | None
    truncated: bool
    arms: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureAlignment:
    """Outcome of ``align_feature``.

    status: completed | stopped (contact_or_stall) | partial | planned (preview) | rejected (0 steps).
    aligned: transverse error <= align_tol_m. stroke_status / achieved_stroke_m report optional stroke.
    """

    arm: str
    status: str
    aligned: bool
    reached: bool
    accepted: bool
    preview: bool
    stalled: bool
    stop_reason: str | None
    feature_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float]
    axis: tuple[float, float, float]
    axial_gap_m: float
    transverse_delta: tuple[float, float, float]
    planned_dxyz: tuple[float, float, float]
    transverse_error_before_m: float
    transverse_error_after_m: float
    standoff_m: float | None
    stroke_status: str | None
    requested_stroke_m: float | None
    achieved_stroke_m: float | None
    align_steps_used: int
    stroke_steps_used: int
    steps_used: int
    planned_steps: int
    final_xyz: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    holding: bool
    opening: float
    ik_error: str | None
    note: str | None
    truncated: bool
    rim_contact_suspected: bool = False
    suggested_backoff_dxyz: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class DeltaGrasp:
    """Outcome of ``delta_grasp``.

    status: completed | stopped | partial | planned (preview) | rejected (0 steps).
    table_contact_stalled: guarded tabletop seating occurred before close.
    closed_opening / opening_after_lift / retained_closure / holding / width_m: grasp telemetry.
    """

    arm: str
    status: str
    aligned: bool
    reached: bool
    accepted: bool
    preview: bool
    stalled: bool
    table_contact_stalled: bool
    stop_reason: str | None
    target_xyz: tuple[float, float, float]
    axis: tuple[float, float, float]
    standoff_m: float
    overdrive_m: float
    contact_table: bool
    transverse_error_before_m: float
    transverse_error_after_m: float
    opening_before: float
    opening_after_preopen: float | None
    closed: bool
    closed_opening: float | None
    opening_after_lift: float | None
    retained_closure: bool
    holding: bool
    opening: float
    width_m: float | None
    lift_status: str | None
    preopen_steps_used: int
    align_steps_used: int
    approach_steps_used: int
    close_steps_used: int
    lift_steps_used: int
    steps_used: int
    planned_steps: int
    final_xyz: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    ik_error: str | None
    note: str | None
    truncated: bool


@dataclass(frozen=True)
class VisualAlign:
    """Outcome of ``visual_servo`` / ``visual_align_and_insert`` (status: completed | partial | rejected)."""

    goal: str
    arm: str
    status: str
    reached: bool
    done: bool
    converged: bool
    contact_complete: bool
    stalled: bool
    stop_reason: str | None
    rounds_used: int
    steps_used: int
    step_allowance: int
    final_xyz: tuple[float, float, float]
    holding: bool
    policy: str | None
    rounds: list[dict[str, Any]]
    last_feedback: dict[str, Any] | None
    policy_error: str | None
    truncated: bool


@dataclass(frozen=True, repr=False)
class Segment(PerceptionResult):
    """Outcome of ``segment``: mask ``centroid`` (row, col), ``area_px``, ``score``, ``box``, ``provenance``."""

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str
    provenance: dict[str, Any] | None = field(default=None, kw_only=True)


@dataclass(frozen=True, repr=False)
class WorldPoint(PerceptionResult):
    """Outcome of ``back_project``: visible surface point ``xyz`` (m), ``depth_m``, ``provenance``."""

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str
    provenance: dict[str, Any] | None = field(default=None, kw_only=True)


@dataclass(frozen=True, repr=False)
class CameraRay(PerceptionResult):
    """Outcome of ``camera_ray``: camera ``origin`` (m) and unit world ``direction`` at ``pixel``."""

    camera: str
    pixel: tuple[float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True, repr=False)
class PlanePoint(WorldPoint):
    """Outcome of ``ray_at_z``: ray intersection ``xyz`` (m) on horizontal plane ``z_plane``."""

    pixel: tuple[float, float]
    z_plane: float = 0.765


@dataclass(frozen=True, repr=False)
class TriangulatedPoint(PerceptionResult):
    """Outcome of ``triangulate``: 3D midpoint ``xyz`` (m), ray skew ``miss_m`` (<0.005 m typical), ``depths_m``."""

    xyz: tuple[float, float, float]
    miss_m: float
    cameras: tuple[str, str]
    pixels: tuple[tuple[float, float], tuple[float, float]]
    depths_m: tuple[float, float]
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class PixelPoint:
    """Outcome of ``project_point``: subpixel ``(row, col)``, ``depth_m``, and ``in_bounds``."""

    camera: str
    row: float
    col: float
    depth_m: float
    in_bounds: bool


@dataclass(frozen=True)
class IKSolution:
    """Outcome of ``ik``: local URDF kinematics check (0 steps).

    ok: reachable within tolerance and clear of table & inter-arm envelopes.
    min_margin_rad / joint_margins_rad: distance to nearest simulator joint limit (rad).
    flipped_jaw: True when ``jaw_symmetry=True`` picked the 180-deg rolled jaw branch.
    """

    ok: bool
    collides: bool
    joints: tuple[float, ...]
    pose: Pose
    pos_err_m: float
    rot_err_deg: float
    min_margin_rad: float
    joint_margins_rad: tuple[float, ...]
    max_delta_rad: float
    flipped_jaw: bool = False
    reason: str | None = None


@dataclass(frozen=True, repr=False)
class ObjectPose(PerceptionResult):
    """Outcome of ``locate``: 3D footprint ``center_xyz``, ``top_z``, ``height_m``, ``length_m``/``width_m``, ``grasp_yaw``."""

    found: bool
    center_xyz: tuple[float, float, float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float | None
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: tuple[int, int] | None
    area_px: int
    camera: str
    note: str | None
    provenance: dict[str, Any] | None = field(default=None, kw_only=True)


STATEFUL_TOOLS = frozenset({"move_pose", "grip", "home", "pi0_pick", "motion_execute", "try_grasp",
                            "intent_execute", "grasp_step", "operation_resume", "nudge",
                            "visual_align_and_insert", "visual_servo", "align_feature",
                            "delta_grasp"})


def _has_provenance(value: Any) -> bool:
    if isinstance(value, PerceptionResult):
        return True
    if isinstance(value, Mapping):
        return value.get("provenance") is not None
    return getattr(value, "provenance", None) is not None


def _extract_xyz(value: Any, name: str) -> tuple[float, float, float]:
    raw: Any
    if isinstance(value, Mapping):
        for field_name in ("xyz", "world_xyz", "center_xyz"):
            if value.get(field_name) is not None:
                raw = value[field_name]
                break
        else:
            raise ValueError(f"{name} mapping must contain 'xyz'")
    elif getattr(value, "xyz", None) is not None:
        raw = value.xyz
    elif getattr(value, "center_xyz", None) is not None:
        raw = value.center_xyz
    else:
        raw = value
    vals = _floats(raw, 3, name)
    return (vals[0], vals[1], vals[2])


def _floats(value: Any, n: int, name: str) -> tuple[float, ...]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {n} numbers, got {value!r}") from exc
    if len(vals) != n or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} must be {n} finite numbers, got {value!r}")
    return vals


def _tuple3(value: Any) -> tuple[float, float, float]:
    try:
        vals = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        vals = ()
    if len(vals) != 3:
        return (math.nan, math.nan, math.nan)
    return vals  # type: ignore[return-value]


def _calibration_provenance(snapshot: dict[str, Any], camera: str) -> dict[str, Any]:
    """Describe the capture used for a camera matrix without asserting currency."""
    image = (snapshot.get("images") or {}).get(camera) or {}
    out = {
        key: deepcopy(snapshot.get(key))
        for key in (
            "session_id", "observation_id", "source_id", "step", "native_steps",
            "source_freshness", "aligned",
        )
    }
    out.update(
        schema="codeaspolicy.calibration.v1",
        camera=camera,
        image_source_id=image.get("source_id"),
        calibration_source_id=image.get("calibration_source_id"),
        calibration_method=image.get("calibration_method"),
    )
    return out


def _arm(name: Any) -> str:
    if name not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {name!r}")
    return str(name)


def _camera(name: Any, cameras: tuple[str, ...]) -> str:
    if name not in cameras:
        raise ValueError(f"camera must be one of {cameras}, got {name!r}")
    return str(name)


def _command(value: Any, *, allow_keep: bool) -> int:
    """A gripper command: 1 closes, -1 opens, 0 keeps (``move_to`` only)."""
    allowed = (-1, 0, 1) if allow_keep else (-1, 1)
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if number not in allowed:
        words = "1 (close), -1 (open) or 0 (keep)" if allow_keep else "1 (close) or -1 (open)"
        raise ValueError(f"gripper must be {words}, got {value!r}")
    return int(number)


def _scalar(value: Any, default: float = 0.0) -> float:
    """First element of a possibly nested one-value sequence (``[0.97]``, a numpy array)."""
    for _ in range(4):
        if value is None or isinstance(value, (int, float)):
            break
        try:
            value = next(iter(value))
        except (TypeError, StopIteration):
            break
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence(value: Any) -> tuple[float, ...]:
    """A list or numpy array as a tuple of floats; ``()`` when it is neither."""
    if value is None:
        return ()
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return ()


def _rotation(quat: Sequence[float]) -> list[list[float]]:
    """[w, x, y, z] to a 3x3 rotation (rows)."""
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def gripper_angles(quat: Sequence[float]) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees of an end-effector quaternion [w, x, y, z].

    The end-effector link's x axis is the pointing direction and its y axis the
    line the fingers close along; the orientation is ``Rz(90 + yaw) Ry(pitch)
    Rx(roll)`` (the starting pose is 0, 0, 0: pointing away from the robot,
    level, fingers side by side).  Within a degree of straight down, yaw and
    roll turn about the same axis: yaw carries their difference, roll reads 0.
    """
    r = _rotation(quat)
    point = (r[0][0], r[1][0], r[2][0])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -point[2]))))
    if math.hypot(point[0], point[1]) < math.sin(math.radians(1.0)):  # within 1 degree
        jaw = (r[0][1], r[1][1])
        return pitch, _wrap(math.degrees(math.atan2(-jaw[1], -jaw[0]))), 0.0
    yaw = math.degrees(math.atan2(point[1], point[0])) - 90.0
    a, p = math.radians(90.0 + yaw), math.radians(pitch)
    ca, sa, cp, sp = math.cos(a), math.sin(a), math.cos(p), math.sin(p)
    base = [[ca * cp, -sa, ca * sp], [sa * cp, ca, sa * sp], [-sp, 0.0, cp]]  # Rz(a) Ry(p)
    # rest = base^T r is a rotation about x by roll: read it off its (2,1) and (1,1) entries
    r21 = sum(base[k][2] * r[k][1] for k in range(3))
    r11 = sum(base[k][1] * r[k][1] for k in range(3))
    return pitch, _wrap(yaw), _wrap(math.degrees(math.atan2(r21, r11)))


def _arm_state(raw: dict[str, Any], name: str) -> ArmState:
    pose = _sequence(raw.get(f"{name}_ee_pose"))
    if len(pose) != 7:
        pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    r = _rotation(pose[3:])
    tcp = tuple(pose[i] + r[i][0] * TCP_OFFSET for i in range(3))
    pitch, yaw, roll = gripper_angles(pose[3:])
    return ArmState(
        tcp_pos=tcp,  # type: ignore[arg-type]
        pitch=round(pitch, 2),
        yaw=round(yaw, 2),
        roll=round(roll, 2),
        eef_pos=pose[:3],  # type: ignore[arg-type]
        eef_quat=pose[3:],  # type: ignore[arg-type]
        gripper=_scalar(raw.get(f"{name}_ee_joint_state")),
        joints=_sequence(raw.get(f"{name}_arm_joint_state")),
    )


def _angle(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number of degrees or None, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


_PICK_ARM_KEYS = (
    "start_eef_z",
    "min_eef_z",
    "peak_eef_z",
    "peak_lift_m",
    "start_gripper",
    "final_gripper",
)


def _pick_arm(raw: Any) -> dict[str, float]:
    raw = raw if isinstance(raw, dict) else {}
    return {key: float(raw.get(key, math.nan)) for key in _PICK_ARM_KEYS}


# ---------------------------------------------------------------------------
# The robot
# ---------------------------------------------------------------------------


class RobodojoRobot(ObjectEvidenceMixin, IntentMixin, CoordinationMixin, RobotBase):
    """The RoboDojo dual-arm station: two 6-DoF ARX-X5 arms with parallel grippers at a table.

    World frame: x runs across the table (the LEFT arm is at negative x and
    appears on the LEFT of the head image), y points away from the robot
    towards the far edge of the table, z is up; metres. State and ``Pose``
    orientations are unit quaternions [w, x, y, z]; ``move_to`` uses the
    gripper angles below, not generic XYZ Euler angles. The arms are
    ``"left"`` and ``"right"``
    (``robo.LEFT`` / ``robo.RIGHT``). ``move_to`` names one arm and
    holds the other; ``move_both`` moves both on one native action clock.
    Choose roles, ordering and recovery in your Python program.

    ``move_to`` drives an arm's GRASP POINT, the middle of the finger pads
    (``state().left.tcp_pos``), to a world position, with an orientation given
    in degrees: ``pitch`` is how far the gripper points below the horizontal
    (0 level, 90 straight down), ``yaw`` the heading of the pointing direction
    on the table (0 away from the robot, +90 towards the robot's left, -90
    towards its right) and ``roll`` the turn of the fingers about the pointing
    direction (0: side by side, as at the start).  Pointing straight down, the
    fingers close along the line at ``yaw`` degrees from +x (counter-clockwise
    seen from above): ``locate``'s ``grasp_yaw`` is that line across an
    object's narrow side.  The table
    surface is at z = 0.765; the arms stand on it at x = -0.30 and +0.30,
    y = -0.45.

    Two gripper conventions, do not mix them up.  COMMANDS are signed:
    ``robo.CLOSE`` (1) closes, ``robo.OPEN`` (-1) opens, ``robo.KEEP`` (0)
    retains the last command. ``move_both`` uses 'close'/'open'/'keep'.
    Gripper commands in either motion take effect from its first native action,
    not on arrival; use a separate close/open call when timing requires it.
    MEASUREMENTS (``state().left.gripper``,
    ``Gripper.opening``) are openings: 1.0 = open, 0.0 = closed on nothing.

    Every native action (one step of ``move_to``, ``set_gripper`` or
    ``home``, one VLA action) counts against the episode's
    native-action budget; when it is spent the state reports ``truncated``
    and motion calls raise ``EpisodeFinished``.  Motion calls block until the
    primitive finishes and return a typed result; nothing is queued.

    Perception is camera based: a fixed head camera (``cam_head``, the whole
    table) and one wrist camera per arm (``cam_left_wrist``,
    ``cam_right_wrist``), 640x480 each.  ``locate`` finds an object by a text
    prompt and measures it in 3D (centre, top, footprint, heading);
    ``segment`` / ``back_project`` give a mask centroid and the world point at
    one pixel.  Object poses are never given.

    The frozen Pi0.5 VLA (``pi0_pick``) is a learned dual-arm policy trained on
    demonstrations of this benchmark's tasks.
    """

    ARMS: tuple[str, str] = ("left", "right")
    LEFT: str = "left"
    RIGHT: str = "right"
    CLOSE: int = 1
    OPEN: int = -1
    KEEP: int = 0

    #: Rendered as the result-type section of the API reference (not a constant).
    RESULT_TYPES: ClassVar[tuple[type, ...]] = (
        State,
        ArmState,
        Move,
        Gripper,
        Home,
        ObjectPose,
        Pick,
        Nudge,
        DeltaGrasp,
        FeatureAlignment,
        VisualAlign,
        Segment,
        WorldPoint,
        CameraRay,
        PlanePoint,
        TriangulatedPoint,
        PixelPoint,
        IKSolution,
    )
    NAME: ClassVar[str] = "robodojo"
    #: No success flag while the episode runs (RPent keeps it from its planner
    #: too) and no recorded world maps: neither member is in the reference.
    ABSENT_MEMBERS: ClassVar[frozenset[str]] = frozenset({"done", "world_map"})
    CAMERAS: ClassVar[tuple[str, ...]] = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    IMAGE_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {
        ("cam_head", "high"): "cam_head.png",
        ("cam_left_wrist", "high"): "cam_left_wrist.png",
        ("cam_right_wrist", "high"): "cam_right_wrist.png",
    }
    # RoboDojo records no depth or world-map artifacts: depth is only reachable
    # through ``back_project`` on a pinned observation.
    WORLD_MAP_ARTIFACTS: ClassVar[dict[tuple[str, str], str]] = {}
    # RPent embeds all three views in every ``view_env_state``; keep parity.
    ON_MOTION_IMAGES: ClassVar[tuple[str, ...]] = (
        "cam_head.png",
        "cam_left_wrist.png",
        "cam_right_wrist.png",
    )
    STATEFUL_TOOLS: ClassVar[frozenset[str]] = STATEFUL_TOOLS
    #: The frozen Pi0.5 pick (hidden and refused under the no-VLA primitive set).
    VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ("pi0_pick",)
    #: External visual-delta VLM primitives (hidden in prompt unless configured).
    DELTA_VLA_PRIMITIVES: ClassVar[tuple[str, ...]] = ("visual_align_and_insert", "visual_servo")
    #: Redundant multi-turn grasp ceremony / preview helpers hidden in default prompt.
    INTENT_HIDDEN_METHODS: ClassVar[frozenset[str]] = frozenset({
        "begin_grasp",
        "grasp_step",
        "grasp_status",
        "align_local",
        "preview_align_feature",
        "preview_delta_grasp",
        "operation_status",
        "resume_operation",
    })
    #: Concise docstring overrides for inherited mixin methods under ``intent_only=True``.
    INTENT_DOC_OVERRIDES: ClassVar[dict[str, str]] = {
        "execute_motion": (
            "Execute an immutable coarse C2 motion intent for one or both arms (do not wrap in once(); "
            "completed IDs never move again).\n\n"
            "left/right: world TCP xyz (keep orientation), Pose, or {'xyz': ..., 'angles': [p, y, r]}; "
            "omitted arms hold. interaction: 'free' (clear route; falls back to smooth joint swing if "
            "SLERP hits singularity), 'approach' or 'contact' (strict line). waypoints=[{'left': ..., "
            "'right': ...}] (1..64) runs one continuous C2 curve. directions={'right': [dx, dy, dz]} "
            "constrains positive axis travel. holding: 'keep'/'empty'/'retain' (never opens/closes). "
            "completion: position_m/orientation_deg/verify; constraints: max_tilt_deg/z_bounds/keep_out/"
            "corridor_m/geometry. max_steps caps cumulative actions; reserve_steps protects cleanup. "
            "preview=True plans without moving (with optional following=[...]). Resume paused intents "
            "with resume_motion(intent_id)."
        ),
        "motion_status": "Read retained motion intent state without moving or refreshing sensors.",
        "resume_motion": (
            "Continue remaining motion of intent_id from measured state after receipt reconciliation; "
            "max_steps optionally raises the cumulative cap."
        ),
        "motion_capabilities": (
            "Report live controller capabilities (speed_control, continuous_trajectory, force_*, collision checks)."
        ),
        "object_ref": (
            "Register a stable session-local logical object reference (short visible-property phrase; "
            "0 steps; not physical identity proof)."
        ),
        "observe_hold": (
            "Compare held-object features against the fixed post-closure baseline (0 steps).\n\n"
            "Returns state ('following'/'relative_motion' in 3D, 'image_consistent'/'image_relative_motion' "
            "in 2D wrist view, 'conflicting_evidence', or 'unknown'). Pass landmarks on capture_observation_id "
            "to repair an empty baseline; set reset=True only after a new close command."
        ),
        "snapshot": "Read latest observation metadata (step, observation_id, aligned); refresh=True requests a new capture.",
        "action_receipt": "Return recorded execution receipt(s) for action_id (or all actions if None).",
        "reconcile": "Inspect pending/unknown action receipts against the latest captured observation before resuming.",
    }
    #: VLA sentences of the remaining docstrings, removed from the no-VLA reference.
    VLA_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "    ``home``, one VLA action) counts against the episode's\n",
            "    ``home``) counts against the episode's\n",
        ),
        (
            "\n\n    The frozen Pi0.5 VLA (``pi0_pick``) is a learned dual-arm policy trained on\n"
            "    demonstrations of this benchmark's tasks.",
            "",
        ),
    )
    #: SAM3 / mask-or-depth perception primitives hidden under ``no-sam``.
    SAM_PRIMITIVES: ClassVar[tuple[str, ...]] = ("locate", "segment", "back_project")
    #: Docstring edits applied after ``VLA_REFERENCE_EDITS`` under ``no-sam``.
    SAM_REFERENCE_EDITS: ClassVar[tuple[tuple[str, str], ...]] = (
        (
            "    seen from above): ``locate``'s ``grasp_yaw`` is that line across an\n"
            "    object's narrow side.  The table\n",
            "    seen from above), across an object's narrow side.  The table\n",
        ),
        (
            "``locate`` finds an object by a text\n"
            "    prompt and measures it in 3D (centre, top, footprint, heading);\n"
            "    ``segment`` / ``back_project`` give a mask centroid and the world point at\n"
            "    one pixel.",
            "``camera_meta`` gives calibrated intrinsics and extrinsics; inspect\n"
            "    ``image`` / ``crop_grid`` to find ``(row, col)`` features, then map them\n"
            "    into 3D with ``ray_at_z`` on a known plane or ``triangulate`` /\n"
            "    ``observe_object`` across two views.",
        ),
        (
            "    Accept a locate/segment/back_project or object-evidence result from this session.\n",
            "    Accept a snapshot-bound geometry or object-evidence result from this session.\n",
        ),
        (
            "    Omit landmarks to acquire distinctive material features inside text\n"
            "    masks, with mutual descriptor matching across views. Once an anchor\n"
            "    exists, recover its SAME features by distinctive descriptor/template\n"
            "    matching against that frozen keyframe, even after empty intermediate\n"
            "    observations. Failed/ambiguous matches are omitted, never substituted\n"
            "    with old pixels or a predicted rigid-grasp pose. Masks nominate object\n"
            "    regions; neither their centres nor silhouettes are triangulated.\n",
            "    Supply landmarks on the initial capture to seed the frozen keyframe.\n"
            "    Once an anchor exists, recover its SAME features by distinctive\n"
            "    descriptor/template matching against that frozen keyframe, even after\n"
            "    empty intermediate observations. Failed/ambiguous matches are omitted,\n"
            "    never substituted with old pixels or a predicted rigid-grasp pose.\n",
        ),
        (
            "    Pair ``step`` with ``observation_id`` in ``segment`` / ``back_project``.",
            "    Pair ``step`` with ``observation_id`` in ``crop_grid`` / ``ray_at_z`` / ``triangulate``.",
        ),
        (
            "    ``holding`` is not proof of identity: confirm lift in wrist view or ``locate``.",
            "    ``holding`` is not proof of identity: confirm lift in wrist view or ``observe_hold``.",
        ),
    )
    SUMMARY_KEYS: ClassVar[tuple[str, ...]] = (
        "success",
        "reached",
        "dist_to_target_m",
        "orientation_error_deg",
        "final_xyz",
        "path",
        "planned_path",
        "status",
        "stop_reason",
        "recoverable",
        "note",
        "blocked",
        "steps_used",
        "trajectory",
        "max_joint_error_rad",
        "holding",
        "opening",
        "width_m",
        "chunks_used",
        "ik_error",
        "found",
        "score",
        "centroid_rc",
        "center_xyz",
        "top_z",
        "grasp_yaw",
        "world_xyz",
        "xyz",
        "camera",
        "cameras",
        "pixel",
        "pixels",
        "origin",
        "direction",
        "miss_m",
        "depth_m",
        "depths_m",
        "z_plane",
        "truncated",
        "step",
        "phase",
        "retained",
        "closure_obstructed",
        "thin_or_unobstructed_closure",
        "needs_visual_confirmation",
        "planned_step_bound",
        "tracking",
        "stalled",
        "planned_steps",
        "rounds_used",
        "converged",
        "done",
        "aligned",
        "accepted",
        "preview",
        "axial_gap_m",
        "transverse_delta",
        "planned_dxyz",
        "transverse_error_before_m",
        "transverse_error_after_m",
        "stroke_status",
        "achieved_stroke_m",
        "align_steps_used",
        "stroke_steps_used",
        "table_contact_stalled",
        "closed",
        "closed_opening",
        "opening_after_lift",
        "retained_closure",
        "lift_status",
        "preopen_steps_used",
        "approach_steps_used",
        "close_steps_used",
        "lift_steps_used",
    )

    # -- observation --------------------------------------------------------

    @property
    def task(self) -> str:
        """The task instruction the benchmark generated for this layout (authoritative).

        Read it before planning.
        """
        return super().task

    @property
    def done(self) -> bool:
        """Absent (not in the API reference): there is no success flag to read.

        It raises instead of returning False, which a ``while not robo.done``
        loop carried over from another robot would spin on until the budget
        is gone.  ``AttributeError`` also makes ``hasattr(robo, "done")`` false.
        """
        raise AttributeError(
            "RoboDojo reports no task success while the episode runs: there is no robo.done; "
            "judge progress from the images and the robot state"
        )

    def _reported_success(self) -> bool:
        return False

    def state(self, step: int = -1) -> State:
        """Read recorded state of both arms (``-1`` = latest, ``0`` = initial)."""
        return super().state(step)

    def image(self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load recorded RGB ``uint8`` array ``(480, 640, 3)`` for ``cam_head`` / ``cam_left_wrist`` / ``cam_right_wrist``.

        Pixels are ``(row, col)`` (row 0 top, col 0 left).
        Pair ``step`` with ``observation_id`` in ``segment`` / ``back_project``.
        """
        return super().image(camera, resolution=resolution, step=step)

    def world_map(
        self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1
    ) -> Any:
        """Absent (not in the API reference): no per-pixel world maps are recorded."""
        raise ValueError(
            "RoboDojo records no world maps; use back_project(row, col, camera) per pixel"
        )

    def _camera_snapshot(
        self, step: int, observation_id: str | None,
    ) -> tuple[dict[str, Any] | None, int]:
        """Resolve a source capture while retaining legacy recorded-step reads."""
        step = int(step)
        if not self._supports_observation_snapshots:
            if observation_id is not None:
                raise NotImplementedError(
                    "This backend cannot bind calibration to an observation_id"
                )
            return None, step
        if step != -1:
            recorded = self._readonly(self.STATE_TOOL, {"step": step})
            snapshot = deepcopy(recorded.get("observation") or {})
            if not snapshot.get("observation_id"):
                raise ValueError("Recorded step has no observation identity")
            if observation_id is not None and observation_id != snapshot["observation_id"]:
                raise ValueError("step and observation_id refer to different observations")
            # A source can appear in multiple records. Preserve the requested record's
            # state/calibration instead of resolving the ID through the latest cache entry.
            snapshot["state"] = deepcopy(recorded.get("state") or {})
            snapshot["image_state"] = deepcopy(snapshot["state"].get("image_state") or {})
            snapshot["cameras"] = deepcopy(snapshot["state"].get("cameras") or {})
            return snapshot, int(snapshot["step"])
        if observation_id is not None:
            snapshot = self._perception_snapshot(observation_id)
            return snapshot, int(snapshot["step"])
        snapshot = self._perception_snapshot(observation_id)
        return snapshot, int(snapshot["step"])

    def camera_meta(
        self, camera: str = "cam_head", *, step: int = -1,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        """Return camera calibration dict (``shape``, ``intrinsic_matrix``, ``extrinsic_matrix``, ``has_depth``, ``provenance``)."""
        camera = _camera(camera, self.CAMERAS)
        snapshot, step = self._camera_snapshot(step, observation_id)
        payload = self._readonly(self.STATE_TOOL, {"step": step})
        cameras = (payload.get("state") or {}).get("cameras") or {}
        result = deepcopy(cameras.get(camera) or {})
        if snapshot is not None:
            result["provenance"] = _calibration_provenance(snapshot, camera)
        return result

    def show(self, camera: str = "cam_head") -> None:
        """Attach current image of ``camera`` to this turn's feedback (use only when a decision depends on it)."""
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return raw bytes of a recorded artifact (``cam_head.png``, ``cam_left_wrist.png``, ``cam_right_wrist.png``)."""
        return super().artifact(name, step=step)

    # -- perception ---------------------------------------------------------

    def _camera_view(
        self, tool: str, camera: str, source: dict[str, Any] | None = None,
        *, observation_id: str | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Build a calibrated ``robodojo_sight.View`` from ``camera_meta``."""
        import numpy as np

        from .hosts import robodojo_sight as sight

        obs_id = source["observation_id"] if source is not None else observation_id
        meta = self.camera_meta(camera, observation_id=obs_id)
        try:
            k = np.asarray(meta.get("intrinsic_matrix"), dtype=np.float64)
            t = np.asarray(meta.get("extrinsic_matrix"), dtype=np.float64)
            raw_shape = meta.get("shape") or (480, 640)
            shape = (int(raw_shape[0]), int(raw_shape[1]))
            valid = (
                k.shape == (3, 3)
                and t.shape == (4, 4)
                and shape[0] > 0
                and shape[1] > 0
                and bool(np.isfinite(k).all())
                and bool(np.isfinite(t).all())
                and float(k[0, 0]) > 0.0
                and float(k[1, 1]) > 0.0
            )
        except (TypeError, ValueError, IndexError):
            valid = False
            shape = (0, 0)
        if not valid:
            raise ToolError(tool, f"Missing or invalid camera calibration for {camera}")
        return sight.View(k, t, shape), deepcopy(meta.get("provenance"))

    def camera_ray(
        self,
        row: float,
        col: float,
        camera: str = "cam_head",
        *,
        observation_id: str | None = None,
    ) -> CameraRay:
        """Return calibrated 3D camera ``origin`` and unit world ``direction`` through ``(row, col)``."""
        import numpy as np

        cam = _camera(camera, self.CAMERAS)
        r_f, c_f = _floats((row, col), 2, "pixel")
        kwargs: dict[str, Any] = {"row": r_f, "col": c_f, "camera": cam}

        def compute(source: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
            view, cal_prov = self._camera_view("camera_ray", cam, source)
            d_per_depth = np.asarray(view.rays(np.float64(r_f), np.float64(c_f)), dtype=np.float64)
            norm = float(np.linalg.norm(d_per_depth))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise ToolError("camera_ray", f"Degenerate camera ray for {cam}")
            u = d_per_depth / norm
            origin = (float(view.pos[0]), float(view.pos[1]), float(view.pos[2]))
            direction = (float(u[0]), float(u[1]), float(u[2]))
            payload: dict[str, Any] = {
                "camera": cam,
                "pixel": (r_f, c_f),
                "origin": origin,
                "direction": direction,
                "calibration": {cam: cal_prov} if cal_prov is not None else None,
            }
            geometry = {
                "basis": "calibrated_camera_ray",
                "world_frame": "robodojo_world",
                "length_unit": "m",
                "depth_is_measured": False,
                "height_prior_used": False,
            }
            return payload, geometry

        data, provenance = self._local_perception(
            "camera_ray", kwargs, compute, observation_id=observation_id
        )
        return CameraRay(
            camera=cam,
            pixel=data["pixel"],
            origin=data["origin"],
            direction=data["direction"],
            provenance=provenance,
        )

    def ray_at_z(
        self,
        row: float,
        col: float,
        z: float = 0.765,
        camera: str = "cam_head",
        *,
        observation_id: str | None = None,
    ) -> PlanePoint:
        """Intersect calibrated pixel ray ``(row, col)`` with horizontal plane ``world_z = z`` (default ``0.765`` m = table)."""
        import numpy as np

        cam = _camera(camera, self.CAMERAS)
        r_f, c_f = _floats((row, col), 2, "pixel")
        z_f = _floats((z,), 1, "z")[0]
        kwargs: dict[str, Any] = {"row": r_f, "col": c_f, "z": z_f, "camera": cam}

        def compute(source: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
            view, cal_prov = self._camera_view("ray_at_z", cam, source)
            d_per_depth = np.asarray(view.rays(np.float64(r_f), np.float64(c_f)), dtype=np.float64)
            norm = float(np.linalg.norm(d_per_depth))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise ToolError("ray_at_z", f"Degenerate camera ray for {cam}")
            u = d_per_depth / norm
            if abs(float(u[2])) < 1e-8:
                raise ToolError(
                    "ray_at_z",
                    f"Camera ray at ({r_f:.1f}, {c_f:.1f}) in {cam} is nearly parallel to plane z={z_f:.4f}",
                )
            t_param = (z_f - float(view.pos[2])) / float(u[2])
            depth_m = t_param / norm
            if not math.isfinite(t_param) or t_param <= 0.0 or depth_m <= 0.0:
                raise ToolError(
                    "ray_at_z",
                    f"Plane z={z_f:.4f} lies behind camera {cam} along ray ({r_f:.1f}, {c_f:.1f})",
                )
            pt = view.pos + depth_m * d_per_depth
            xyz = (float(pt[0]), float(pt[1]), float(z_f))
            payload: dict[str, Any] = {
                "camera": cam,
                "pixel": (r_f, c_f),
                "xyz": xyz,
                "world_xyz": xyz,
                "depth_m": float(depth_m),
                "z_plane": float(z_f),
                "calibration": {cam: cal_prov} if cal_prov is not None else None,
            }
            geometry = {
                "basis": "calibrated_plane_intersection",
                "world_frame": "robodojo_world",
                "length_unit": "m",
                "table_z_m": float(z_f),
                "depth_is_measured": False,
                "height_prior_used": True,
                "prior_scope": "horizontal_plane_z",
                "assumptions": [f"point lies on horizontal plane world z = {float(z_f):.6g} m"],
            }
            return payload, geometry

        data, provenance = self._local_perception(
            "ray_at_z", kwargs, compute, observation_id=observation_id
        )
        return PlanePoint(
            pixel=data["pixel"],
            xyz=data["xyz"],
            depth_m=data["depth_m"],
            camera=cam,
            z_plane=data["z_plane"],
            provenance=provenance,
        )

    def triangulate(
        self,
        cam1: str,
        rc1: Sequence[float],
        cam2: str,
        rc2: Sequence[float],
        *,
        observation_id: str | None = None,
        max_miss_m: float = 0.025,
    ) -> TriangulatedPoint:
        """Triangulate one 3D feature from ``rc1`` in ``cam1`` and ``rc2`` in ``cam2`` on the same snapshot (raises ToolError if ``miss_m > max_miss_m``)."""
        import numpy as np

        c1_name = _camera(cam1, self.CAMERAS)
        c2_name = _camera(cam2, self.CAMERAS)
        if c1_name == c2_name:
            raise ToolError("triangulate", f"triangulate requires two distinct cameras, got {c1_name!r}")
        r1, col1 = _floats(rc1, 2, "rc1")
        r2, col2 = _floats(rc2, 2, "rc2")
        tol = _floats((max_miss_m,), 1, "max_miss_m")[0]
        if tol <= 0.0:
            raise ValueError(f"max_miss_m must be positive, got {max_miss_m!r}")
        kwargs: dict[str, Any] = {
            "cam1": c1_name,
            "rc1": [r1, col1],
            "cam2": c2_name,
            "rc2": [r2, col2],
            "max_miss_m": tol,
        }

        def compute(source: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
            view1, cal1 = self._camera_view("triangulate", c1_name, source)
            view2, cal2 = self._camera_view("triangulate", c2_name, source)
            d1 = np.asarray(view1.rays(np.float64(r1), np.float64(col1)), dtype=np.float64)
            d2 = np.asarray(view2.rays(np.float64(r2), np.float64(col2)), dtype=np.float64)
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))
            if not (math.isfinite(n1) and math.isfinite(n2)) or n1 <= 1e-12 or n2 <= 1e-12:
                raise ToolError("triangulate", "Degenerate camera ray in triangulation")
            u1 = d1 / n1
            u2 = d2 / n2
            cross_norm = float(np.linalg.norm(np.cross(u1, u2)))
            if not math.isfinite(cross_norm) or cross_norm < 1e-6:
                raise ToolError(
                    "triangulate", "Camera rays are nearly parallel; insufficient stereo baseline"
                )
            p1 = np.asarray(view1.pos, dtype=np.float64)
            p2 = np.asarray(view2.pos, dtype=np.float64)
            w0 = p1 - p2
            b = float(np.dot(u1, u2))
            d = float(np.dot(u1, w0))
            e = float(np.dot(u2, w0))
            denom = 1.0 - b * b
            if denom <= 1e-12:
                raise ToolError(
                    "triangulate", "Camera rays are nearly parallel; insufficient stereo baseline"
                )
            s1 = (b * e - d) / denom
            s2 = (e - b * d) / denom
            if not (math.isfinite(s1) and math.isfinite(s2)) or s1 <= 0.0 or s2 <= 0.0:
                raise ToolError(
                    "triangulate", "Triangulated point lies behind at least one camera"
                )
            q1 = p1 + s1 * u1
            q2 = p2 + s2 * u2
            mid = 0.5 * (q1 + q2)
            miss_m = float(np.linalg.norm(q1 - q2))
            _, _, depth1_arr = view1.project(mid)
            _, _, depth2_arr = view2.project(mid)
            depth1 = float(depth1_arr)
            depth2 = float(depth2_arr)
            if not (math.isfinite(depth1) and math.isfinite(depth2)) or depth1 <= 0.0 or depth2 <= 0.0:
                raise ToolError(
                    "triangulate", "Triangulated point lies behind at least one camera"
                )
            if miss_m > tol:
                raise ToolError(
                    "triangulate",
                    f"Rays miss by {miss_m*1000:.1f} mm (> {tol*1000:.1f} mm); "
                    "pixels do not match the same 3D feature",
                )
            xyz = (float(mid[0]), float(mid[1]), float(mid[2]))
            cal_map = {k: v for k, v in ((c1_name, cal1), (c2_name, cal2)) if v is not None}
            payload: dict[str, Any] = {
                "xyz": xyz,
                "world_xyz": xyz,
                "miss_m": miss_m,
                "cameras": (c1_name, c2_name),
                "pixels": ((r1, col1), (r2, col2)),
                "depths_m": (depth1, depth2),
                "calibration": cal_map or None,
            }
            geometry = {
                "basis": "calibrated_stereo_rays",
                "world_frame": "robodojo_world",
                "length_unit": "m",
                "depth_is_measured": True,
                "height_prior_used": False,
                "agreement_view_count": 2,
                "uncertainty_note": f"ray skew miss {miss_m * 1000.0:.2f} mm",
            }
            return payload, geometry

        data, provenance = self._local_perception(
            "triangulate", kwargs, compute, observation_id=observation_id
        )
        return TriangulatedPoint(
            xyz=data["xyz"],
            miss_m=data["miss_m"],
            cameras=data["cameras"],
            pixels=data["pixels"],
            depths_m=data["depths_m"],
            provenance=provenance,
        )

    def project_point(
        self,
        xyz: Sequence[float],
        camera: str = "cam_head",
        *,
        observation_id: str | None = None,
    ) -> PixelPoint | None:
        """Project world ``xyz`` (m) into ``camera`` ``(row, col)`` pixels; return ``None`` if behind camera."""
        import numpy as np

        pt = _floats(xyz, 3, "xyz")
        cam = _camera(camera, self.CAMERAS)
        view, _ = self._camera_view("project_point", cam, observation_id=observation_id)
        r_arr, c_arr, depth_arr = view.project(np.asarray(pt, dtype=np.float64))
        depth_f = float(depth_arr)
        if not math.isfinite(depth_f) or depth_f <= 0.0:
            return None
        r_f, c_f = float(r_arr), float(c_arr)
        h, w = view.shape
        return PixelPoint(
            camera=cam,
            row=r_f,
            col=c_f,
            depth_m=depth_f,
            in_bounds=bool(0.0 <= r_f < h and 0.0 <= c_f < w),
        )

    def crop_grid(
        self,
        camera: str = "cam_head",
        box: Sequence[int] | None = None,
        *,
        step_px: int = 25,
        points: Sequence[Sequence[float]] = (),
        scale: int = 2,
        observation_id: str | None = None,
    ) -> Any:
        """Return upscaled RGB ``uint8`` crop of ``box=(r0, c0, r1, c1)`` with original ``(row, col)`` rulers and optional numbered ``points`` crosshairs for ``display()``."""
        import numpy as np
        from PIL import Image, ImageDraw

        cam = _camera(camera, self.CAMERAS)
        if isinstance(scale, bool) or int(scale) != scale or int(scale) < 1:
            raise ValueError(f"scale must be an integer >= 1, got {scale!r}")
        if isinstance(step_px, bool) or int(step_px) != step_px or int(step_px) < 5:
            raise ValueError(f"step_px must be an integer >= 5, got {step_px!r}")
        scale_i = int(scale)
        step_i = int(step_px)
        pts = [_floats(pt, 2, "point") for pt in points]

        if observation_id is not None:
            _, img_step = self._camera_snapshot(-1, observation_id)
            raw = self.image(cam, step=img_step)
        else:
            raw = self.image(cam)
        arr = np.asarray(raw, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Expected HxWx3 RGB image array, got shape {arr.shape!r}")
        arr = arr[:, :, :3]
        h, w = int(arr.shape[0]), int(arr.shape[1])

        if box is None:
            r0, c0, r1, c1 = 0, 0, h, w
        else:
            if isinstance(box, (str, bytes)):
                raise ValueError(f"box must be (row0, col0, row1, col1), got {box!r}")
            vals = _floats(box, 4, "box")
            if any(isinstance(v, bool) or int(f) != f for v, f in zip(box, vals)):
                raise ValueError(f"box bounds must be integers, got {box!r}")
            r0, c0, r1, c1 = (int(f) for f in vals)
            if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
                raise ValueError(
                    f"box (row0, col0, row1, col1) must satisfy 0 <= r0 < r1 <= {h} and "
                    f"0 <= c0 < c1 <= {w}, got {(r0, c0, r1, c1)!r}"
                )

        crop = arr[r0:r1, c0:c1]
        crop_h, crop_w = r1 - r0, c1 - c0
        scaled_h, scaled_w = crop_h * scale_i, crop_w * scale_i
        pil_crop = Image.fromarray(crop)
        if scale_i > 1:
            resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
            pil_crop = pil_crop.resize((scaled_w, scaled_h), resample=resample)

        margin_top = 26
        margin_left = 36
        canvas_w = margin_left + scaled_w
        canvas_h = margin_top + scaled_h

        # Alpha-composite subtle gridlines over the upscaled crop so features stay visible.
        crop_rgba = pil_crop.convert("RGBA")
        grid_layer = Image.new("RGBA", (scaled_w, scaled_h), (0, 0, 0, 0))
        grid_draw = ImageDraw.Draw(grid_layer)

        first_r = ((r0 + step_i - 1) // step_i) * step_i
        row_ticks: list[tuple[int, int]] = []
        for r in range(first_r, r1, step_i):
            y_local = min(scaled_h - 1, max(0, (r - r0) * scale_i + (scale_i // 2)))
            grid_draw.line([(0, y_local), (scaled_w - 1, y_local)], fill=(255, 255, 255, 72), width=1)
            row_ticks.append((r, margin_top + y_local))

        first_c = ((c0 + step_i - 1) // step_i) * step_i
        col_ticks: list[tuple[int, int]] = []
        for c in range(first_c, c1, step_i):
            x_local = min(scaled_w - 1, max(0, (c - c0) * scale_i + (scale_i // 2)))
            grid_draw.line([(x_local, 0), (x_local, scaled_h - 1)], fill=(255, 255, 255, 72), width=1)
            col_ticks.append((c, margin_left + x_local))

        blended_crop = Image.alpha_composite(crop_rgba, grid_layer).convert("RGB")
        canvas = Image.new("RGB", (canvas_w, canvas_h), (24, 26, 30))
        canvas.paste(blended_crop, (margin_left, margin_top))
        draw = ImageDraw.Draw(canvas)

        # Ruler borders and original-image (row, col) tick labels.
        draw.line([(margin_left - 1, 0), (margin_left - 1, canvas_h - 1)], fill=(96, 102, 115), width=1)
        draw.line([(0, margin_top - 1), (canvas_w - 1, margin_top - 1)], fill=(96, 102, 115), width=1)
        draw.text((2, 7), "r\\c", fill=(165, 175, 195))

        last_label_y = -100
        for r, y_canvas in row_ticks:
            draw.line([(margin_left - 4, y_canvas), (margin_left - 1, y_canvas)], fill=(210, 215, 225), width=1)
            if y_canvas - last_label_y >= 11:
                label_y = max(margin_top, min(canvas_h - 10, y_canvas - 5))
                draw.text((2, label_y), f"r={r}" if r < 100 else f"{r}", fill=(232, 236, 244))
                last_label_y = y_canvas

        last_label_x = -100
        for c, x_canvas in col_ticks:
            draw.line([(x_canvas, margin_top - 4), (x_canvas, margin_top - 1)], fill=(210, 215, 225), width=1)
            label = f"c={c}" if not col_ticks or c == col_ticks[0][0] else f"{c}"
            min_gap = max(24, len(label) * 6 + 4)
            if x_canvas - last_label_x >= min_gap:
                label_x = max(margin_left, min(canvas_w - len(label) * 6 - 1, x_canvas - 6))
                draw.text((label_x, 7), label, fill=(232, 236, 244))
                last_label_x = x_canvas

        palette = (
            (255, 64, 64),
            (64, 255, 128),
            (64, 200, 255),
            (255, 220, 64),
            (255, 96, 220),
            (255, 160, 64),
        )
        arm_px = max(5, 3 * scale_i)
        for idx, (pr, pc) in enumerate(pts):
            if not (r0 <= pr < r1 and c0 <= pc < c1):
                continue
            x_c = margin_left + min(
                scaled_w - 1, max(0, int(math.floor((pc - c0) * scale_i + (scale_i // 2) + 0.5)))
            )
            y_c = margin_top + min(
                scaled_h - 1, max(0, int(math.floor((pr - r0) * scale_i + (scale_i // 2) + 0.5)))
            )
            color = palette[idx % len(palette)]
            x0_line = max(margin_left, x_c - arm_px)
            x1_line = min(canvas_w - 1, x_c + arm_px)
            y0_line = max(margin_top, y_c - arm_px)
            y1_line = min(canvas_h - 1, y_c + arm_px)
            draw.line([(x0_line, y_c), (x1_line, y_c)], fill=(0, 0, 0), width=3)
            draw.line([(x_c, y0_line), (x_c, y1_line)], fill=(0, 0, 0), width=3)
            draw.line([(x0_line, y_c), (x1_line, y_c)], fill=color, width=1)
            draw.line([(x_c, y0_line), (x_c, y1_line)], fill=color, width=1)
            tag = str(idx)
            tx = min(canvas_w - len(tag) * 6 - 2, x_c + 4)
            ty = max(margin_top + 1, y_c - 11)
            draw.rectangle([(tx - 1, ty - 1), (tx + len(tag) * 6, ty + 9)], fill=(0, 0, 0))
            draw.text((tx, ty), tag, fill=color)

        return np.asarray(canvas, dtype=np.uint8)

    def segment(
        self,
        text_prompt: str,
        camera: str = "cam_head",
        min_score: float = 0.2,
        *,
        observation_id: str | None = None,
    ) -> Segment:
        """Segment one object in ``camera`` by ``text_prompt`` (SAM3); pass ``centroid`` + ``observation_id`` to ``back_project``."""
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        p = self._perception_call("segment", kwargs, observation_id=observation_id)
        provenance = deepcopy(p.get("provenance"))
        if not p.get("found"):
            return Segment(
                False, None, 0, None, None, str(p.get("camera", camera)),
                provenance=provenance,
            )
        centroid = p.get("centroid_rc") or ()
        moments = p.get("mask_moments") or ()
        return Segment(
            found=True,
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(moments[0]) if moments else 0,
            score=float(p["score"]) if p.get("score") is not None else None,
            box=p.get("box_px"),
            camera=str(p.get("camera", camera)),
            provenance=provenance,
        )

    def back_project(
        self, row: int, col: int, camera: str = "cam_head", *,
        observation_id: str | None = None,
    ) -> WorldPoint:
        """Return world surface point ``xyz`` at pixel ``(row, col)`` in ``camera`` (480x640) for ``observation_id``."""
        camera = _camera(camera, self.CAMERAS)
        kwargs: dict[str, Any] = {"row": int(row), "col": int(col), "camera": camera}
        p = self._perception_call("back_project", kwargs, observation_id=observation_id)
        pixel = p.get("pixel") or (int(row), int(col))
        return WorldPoint(
            pixel=(int(pixel[0]), int(pixel[1])),
            xyz=_tuple3(p.get("world_xyz")),
            depth_m=float(p.get("depth_m", math.nan)),
            camera=str(p.get("camera", camera)),
            provenance=deepcopy(p.get("provenance")),
        )

    def locate(
        self,
        text_prompt: str,
        camera: str = "cam_head",
        min_score: float = 0.2,
        *,
        observation_id: str | None = None,
    ) -> ObjectPose:
        """Segment ``text_prompt`` (SAM3) and measure 3D ``center_xyz``, ``top_z``, footprint, and ``grasp_yaw`` through the depth image (0 steps; verify mask against image)."""
        camera = _camera(camera, self.CAMERAS)
        prompt = str(text_prompt)
        if not prompt.strip():
            raise ValueError("text_prompt must not be empty")
        kwargs: dict[str, Any] = {
            "text_prompt": prompt,
            "camera": camera,
            "min_score": float(min_score),
        }
        p = self._perception_call("locate", kwargs, observation_id=observation_id)
        centroid = p.get("centroid_rc") or ()

        def num(key: str) -> float | None:
            return float(p[key]) if p.get(key) is not None else None

        center = p.get("center_xyz")
        return ObjectPose(
            found=bool(p.get("found")),
            center_xyz=_tuple3(center) if center is not None else None,
            top_z=num("top_z"),
            height_m=num("height_m"),
            length_m=num("length_m"),
            width_m=num("width_m"),
            long_axis_deg=num("long_axis_deg"),
            grasp_yaw=num("grasp_yaw"),
            score=num("score"),
            centroid=(int(centroid[0]), int(centroid[1])) if len(centroid) == 2 else None,
            area_px=int(p.get("area_px") or 0),
            camera=str(p.get("camera", camera)),
            note=str(p["geometry_error"]) if p.get("geometry_error") else None,
            provenance=deepcopy(p.get("provenance")),
        )

    # -- the VLA ------------------------------------------------------------

    def pi0_pick(self, prompt: str | None = None, arm: str = "right", max_chunks: int = 8) -> Pick:
        """Run frozen Pi0.5 whole-task dual-arm policy for up to ``max_chunks`` chunks (<=50 actions each) until lift heuristic fires.

        ``prompt=None`` uses ``robo.task``; overrides must not contain ``<...>``. Confirm held object in wrist view.
        """
        arm = _arm(arm)
        if int(max_chunks) < 1:
            raise ValueError("max_chunks must be at least 1")
        kwargs: dict[str, Any] = {"arm": arm, "max_chunks": int(max_chunks)}
        if prompt is not None:
            text = str(prompt)
            if not text.strip():
                raise ValueError("prompt must not be empty; pass None to use the task instruction")
            if "<" in text or ">" in text:
                raise ValueError("prompt must not contain template markers such as <target>")
            kwargs["prompt"] = text

        def build(p: dict[str, Any]) -> Pick:
            arms = p.get("arms") or {}
            return Pick(
                success=bool(p.get("success")),
                instruction=str(p.get("instruction") or ""),
                chunks_used=int(p.get("chunks_used", 0)),
                left=_pick_arm(arms.get("left")),
                right=_pick_arm(arms.get("right")),
                monitored_arm=str(p.get("monitored_arm", arm)),
                budget_exhausted=bool(p.get("terminated")),
                truncated=self._latest_state().truncated,
            )

        return self._stateful("pi0_pick", kwargs, build)

    # -- motion -------------------------------------------------------------

    def move_to(
        self,
        xyz: Sequence[float],
        arm: str = "right",
        gripper: int = 0,
        pitch: float | None = None,
        yaw: float | None = None,
        roll: float | None = None,
        path: str = "auto",
        *,
        motion: MotionProfile | MotionWindow | str | None = None,
        preview: bool = False,
    ) -> Move | dict:
        """Move one arm's grasp point to a world position, with an orientation.

        The grasp point is the middle of the finger pads (``tcp_pos``).  The
        library solves the arm's inverse kinematics for the full pose and,
        with ``path="auto"``, moves the grasp point along a straight line to
        it, with nominal waypoints at most 2 cm / 8 degrees apart. Joint-step
        limits may subdivide them; this is not a speed guarantee. When the
        line leaves the arm's reach it swings the joints to the solution instead.
        Nothing moves when the pose is out of reach or every way there takes
        the arm into the table: ``ik_error`` says why.

        Args:
            xyz: target grasp-point position [x, y, z], metres.
            arm: ``"left"`` or ``"right"``.  The other arm holds its joints.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1) to command the
                moving arm's gripper on every action of the motion,
                ``robo.KEEP`` (0) to keep commanding what it was last told (a
                held object stays squeezed).  Keep it closed (1) while
                carrying an object.
            pitch / yaw / roll: the orientation to arrive with, degrees (the
                class docstring defines them).  An angle left None keeps its
                current value in the target. With an explicitly requested
                orientation and an EMPTY open gripper the fingers are
                interchangeable, so the pose turned half a turn about the
                pointing direction may be taken when it needs less joint
                travel (``note`` says so; yaw / roll then read 180 off);
                holding something, no half-turn substitution is made.
                Check the reported pose errors, including any near-miss note.
            path: ``"auto"`` (default), ``"line"`` (straight line or nothing)
                or ``"joint"`` (swing the joints: fewer actions, but the grasp
                point does not follow a line and orientation may change en
                route even if the target preserves it: not near objects).
            motion: 'native' uses the original simulator waypoint execution
                (no C2/derivative guarantees); prefer it for observed clear
                empty-hand travel and clear, orientation-preserving transport
                after a broad stable grasp is verified. 'transit' is the smooth travel alternative;
                'approach' for the
                final approach, 'carry' for delicate held-object transport, 'precision'
                for placement/insertion, or robo.motion_profile(velocity, acceleration, jerk), in
                rad/s, rad/s², rad/s³. On timed backends, omitted uses defaults;
                legacy hosts retain geometric stepping. Explicit limits require
                timed substep support and otherwise reject without movement.
                One call ramps from/to rest; use move_both(via=...) to connect
                multiple targets in a single continuous joint trajectory.
                trajectory reports simulation timing and command bounds, not
                measured speed or force. Tracking misses interrupt both native
                and timed moves before later waypoints or endpoint settling.
                Select phases from task intent and current observations, not
                gripper opening alone. Omitted limits remain 1/4/40. Explicit
                carry is faster. motion_window(far='transit', near='approach',
                distance_m=0.02) slows only the last nominal path region in
                the same C2 move. at='start' slows departure before speeding up.
                Use far='carry', near='precision' for a placement approach.
            preview: True returns a read-only planning dict with accepted,
                trajectory.planned_steps, planned_duration_s, limiting_factor
                and budget information. Neither arms nor grippers move. An
                over-budget rejection keeps its estimate. Execution always
                replans from current state; a preview grants no motion permission.
        Returns:
            A planning dict when preview=True, otherwise Move; inspect
            status/stop_reason and test ``reached``. A rejected
            move consumes zero native actions and path is None; planned_path
            may still name the selected geometry. ``ik_error`` only describes
            geometry refusals, never timing support or budget failures. ``blocked``
            flags tracking lag, not confirmed contact. Inspect the actual
            endpoint and scene before a dependent action or retry.
        Raises:
            ValueError: a malformed argument (nothing is sent).
        """
        self._legacy_motion_guard()
        arm = _arm(arm)
        target = _floats(xyz, 3, "xyz")
        command = _command(gripper, allow_keep=True)
        if path not in ("auto", "line", "joint"):
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
        kwargs: dict[str, Any] = {"xyz": list(target), "arm": arm, "gripper": command, "path": path}
        motion = resolve_motion(motion)
        if type(preview) is not bool:
            raise ValueError("preview must be bool")
        if motion is not None:
            kwargs["motion"] = motion.payload()
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            angle = _angle(value, name)
            if angle is not None:
                kwargs[name] = angle
        if preview:
            return self._readonly("preview_move_pose", kwargs)
        return self._stateful("move_pose", kwargs, lambda p: self._move(arm, target, p))

    def try_grasp(
        self, xyz: Sequence[float], arm: str = "right", *,
        pitch: float | None = None, yaw: float | None = None, roll: float | None = None,
        approach: Sequence[float] = (0, 0, 0.05),
        lift: Sequence[float] = (0, 0, 0.025), opening: float = 1.0,
        max_steps: int = 64, preview: bool = False,
    ) -> dict:
        """Try one bounded grasp with fast travel, short local approach and probe.

        xyz is the finger-pad centre at closure, chosen from current images.
        approach is a world offset FROM xyz to the clear pregrasp (2–12 cm);
        lift is a short world offset FROM xyz for the probe (1–6 cm).
        Angles use the same convention as move_to; omitted angles are kept.
        For a vertical grasp specify pitch=90, yaw=<closing axis>, roll=0.
        opening is the initial normalized finger separation in (0, 1].

        Before any movement/opening, preflight the approach, closure position
        and probe with one fixed orientation and a conservative step bound.
        preview=True only returns that plan. No approximate pose or equivalent
        half-turn is substituted. Geometry checks cover robot/table limits,
        not objects, finger clearance or the other arm. Choose a clear route.
        Preview a subsequent carry/turn/place separately before grasping.

        Execution uses native interpolation for clear travel and 4 mm targets
        for the last 2 cm and probe, with measured tracking checks; no C2 or
        force-control guarantee. At most two local target holds are allowed
        while measurable progress continues. A stall, empty close or closure
        lost during the probe stops immediately, with no automatic regrasp.
        max_steps caps ALL actions including opening/closing and corrections.

        Returns phase, phases, status, stop_reason, final_xyz, steps_used and
        retained. retained only means a short probe completed with obstructed
        closure; grasp_verified remains False. Show the wrist/head images and
        confirm the intended object followed before carrying or releasing a
        donor. Prefer a broad solid grasp patch with exposed finger clearance.
        A failed probe leaves the last gripper command in place. Inspect the
        actual state before any recovery. Wrap this attempt in once(); after
        interruption reconcile receipts and continue only remaining work, never
        repeat the entire grasp on an object that may already be held.
        """
        kwargs = dict(xyz=list(_floats(xyz, 3, "xyz")), arm=_arm(arm),
                      approach=list(_floats(approach, 3, "approach")),
                      lift=list(_floats(lift, 3, "lift")), opening=opening, max_steps=max_steps)
        for name, value in (("pitch", pitch), ("yaw", yaw), ("roll", roll)):
            angle = _angle(value, name)
            if angle is not None:
                kwargs[name] = angle
        if type(preview) is not bool:
            raise ValueError("preview must be bool")
        if preview:
            return self._readonly("preview_grasp", kwargs)
        return self._stateful("try_grasp", kwargs, lambda payload: payload)

    def nudge(
        self,
        dxyz: Sequence[float] = (0.0, 0.0, 0.0),
        arm: str = "right",
        *,
        dpitch: float = 0.0,
        dyaw: float = 0.0,
        droll: float = 0.0,
        gripper: int = 0,
        frame: str = "world",
        left: dict | None = None,
        right: dict | None = None,
        stop_on_stall: bool = True,
        stall_steps: int = 2,
        max_steps: int | None = None,
        reserve_steps: int = 0,
        compact: bool = True,
        max_linear_m: float = 0.06,
        max_angular_deg: float = 20.0,
        allow_table_contact: bool = False,
    ) -> Nudge:
        """Guarded short relative TCP move for last-mile contact (<= ``max_linear_m`` <= 0.12 m, <= ``max_angular_deg`` <= 45 deg).

        ``frame="world"`` adds to world xyz and pitch/yaw/roll; ``frame="tcp"`` moves along TCP
        (+x pointing, +y jaw-closing, +z normal). ``arm="both"`` takes ``left={...}``/``right={...}``.
        ``compact=True`` steps 5 mm/action (3 mm with ``allow_table_contact=True`` or thin hold
        ``opening <= 0.045``); preflighted (refusal costs 0 steps). ``allow_table_contact=True``
        relaxes only the fingertip floor to ``TABLE_Z - 0.004`` while arm links stay strictly clear.
        ``stop_on_stall=True`` returns ``status="stopped", stalled=True, stop_reason="contact_or_stall"``
        without raising (kinematic progress test, not force; check views).
        """
        kwargs = self._nudge_kwargs(dxyz, arm, dpitch, dyaw, droll, gripper, frame, left,
                                    right, stop_on_stall, stall_steps, max_steps,
                                    reserve_steps, compact, max_linear_m, max_angular_deg,
                                    allow_table_contact)
        return self._stateful("nudge", kwargs, lambda p: self._nudge_result(kwargs, p))

    def delta_grasp(
        self,
        target: Any,
        arm: str = "right",
        *,
        axis: Sequence[float] = (0.0, 0.0, -1.0),
        standoff_m: float = 0.0,
        overdrive_m: float = 0.0,
        opening: float | None = None,
        contact_table: bool = False,
        lift_dxyz: Sequence[float] | None = None,
        align_tol_m: float = 0.0025,
        max_linear_m: float = 0.06,
        max_steps: int = 36,
        reserve_steps: int = 20,
        require_current_target: bool = False,
        preview: bool = False,
    ) -> DeltaGrasp:
        """Compact close-range grasp or in-air handoff clamp from a staged ~2-5 cm hover.

        Runs: (1) optional pre-open to ``opening`` (0 steps if within 0.05), (2) transverse
        alignment perpendicular to ``axis`` (gated by ``align_tol_m``), (3) guarded axial approach
        to ``target - standoff_m * axis + overdrive_m * axis``, (4) close, and (5) optional guarded
        ``lift_dxyz`` (3 mm micro-steps when ``closed_opening <= 0.045``).
        ``target`` accepts ``[x, y, z]``, ``Pose``, or any perception result with ``xyz``.
        ``contact_table=True`` uses 3 mm steps with ``allow_table_contact=True`` and treats tabletop
        stall as expected seating before close. ``preview=True`` runs 0-step preflight.
        """
        if type(preview) is not bool:
            raise ValueError("preview must be bool")
        if type(require_current_target) is not bool:
            raise ValueError("require_current_target must be bool")
        kwargs = self._delta_grasp_kwargs(
            target, arm, axis, standoff_m, overdrive_m, opening, contact_table,
            lift_dxyz, align_tol_m, max_linear_m, max_steps, reserve_steps,
        )
        if require_current_target:
            self.require_current(target)
        if preview:
            payload = self._readonly("preview_delta_grasp", kwargs)
            return self._delta_grasp_result(kwargs, payload, preview=True)
        return self._stateful(
            "delta_grasp", kwargs, lambda p: self._delta_grasp_result(kwargs, p, preview=False)
        )

    def align_feature(
        self,
        feature: Any,
        target: Any,
        arm: str = "right",
        *,
        axis: Sequence[float] = (0.0, 0.0, -1.0),
        standoff_m: float | None = None,
        stroke_m: float | None = None,
        stroke_dxyz: Sequence[float] | None = None,
        align_tol_m: float = 0.0025,
        max_linear_m: float = 0.06,
        max_steps: int = 15,
        reserve_steps: int = 20,
        stop_on_stall: bool = True,
        stall_steps: int = 2,
        allow_table_contact: bool = False,
        preview: bool = False,
        require_current_feature: bool = True,
        require_current_target: bool = False,
    ) -> FeatureAlignment:
        """Align a held distal working ``feature`` to ``target`` perpendicular to ``axis``, with optional guarded stroke.

        Accepts ``[x, y, z]``, ``Pose``, or perception results with ``xyz`` (requires current
        provenance on ``feature`` by default). Optional ``standoff_m`` positions upstream along
        ``-axis``; optional ``stroke_m`` (along ``+axis``) or ``stroke_dxyz`` runs only if aligned
        within ``align_tol_m``. ``preview=True`` runs 0-step preflight.
        """
        if type(preview) is not bool:
            raise ValueError("preview must be bool")
        if type(require_current_feature) is not bool or type(require_current_target) is not bool:
            raise ValueError("require_current_feature and require_current_target must be bool")
        kwargs = self._align_feature_kwargs(
            feature, target, arm, axis, standoff_m, stroke_m, stroke_dxyz,
            align_tol_m, max_linear_m, max_steps, reserve_steps,
            stop_on_stall, stall_steps, allow_table_contact,
        )
        if require_current_feature and _has_provenance(feature):
            self.require_current(feature)
        if require_current_target:
            self.require_current(target)
        if preview:
            payload = self._readonly("preview_align_feature", kwargs)
            return self._align_feature_result(kwargs, payload, preview=True)
        return self._stateful(
            "align_feature", kwargs, lambda p: self._align_feature_result(kwargs, p, preview=False)
        )

    def visual_align_and_insert(
        self,
        goal: str,
        arm: str = "right",
        frame: str = "world",
        max_rounds: int = 8,
        max_steps: int = 40,
        reserve_steps: int = 20,
        stop_on_stall: bool = True,
        crop_wrist: bool = True,
        cameras: Sequence[str] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        policy: Any = None,
        *,
        stop_on_contact_complete: bool = False,
        max_linear_m: float = 0.03,
        max_angular_deg: float = 15.0,
    ) -> VisualAlign:
        """Closed-loop last-mile alignment/insertion via a configured visual-delta (PEAR) policy.

        Each round sends current camera views (plus 2x wrist crop when ``crop_wrist``) and arm
        state to the policy and executes its clamped delta via ``nudge`` until ``done``, zero
        delta, contact completion, or round/step cap. ``policy`` may be a local callable.
        """
        kwargs = self._align_kwargs(goal, arm, frame, max_rounds, max_steps, reserve_steps,
                                    stop_on_stall, crop_wrist, cameras, model, base_url,
                                    stop_on_contact_complete, max_linear_m, max_angular_deg)
        if policy is not None:
            if isinstance(policy, (str, bytes)):
                raise ValueError("policy must be a callable or an object with predict()")
            payload = self._local_visual_align(kwargs, policy)
        else:
            payload = self._stateful("visual_align_and_insert", kwargs, lambda p: p)
        return self._align_result(payload)

    def _visual_servo_tool(self) -> str:
        """Pick ``visual_servo`` when supported as a stateful backend tool, else ``visual_align_and_insert``."""
        toolkit = getattr(self._backend, "_toolkit", None)
        if toolkit is not None:
            tools = getattr(toolkit, "_tools", None)
            if isinstance(tools, Mapping) and "visual_servo" in tools:
                from .hosts import robodojo_native as native

                if "visual_servo" in getattr(native, "STATEFUL", ()):
                    return "visual_servo"
            return "visual_align_and_insert"
        tools = getattr(self._backend, "_tools", None)
        if tools is None:
            tools = getattr(self._backend, "tools", None)
        if isinstance(tools, (Mapping, set, frozenset, list, tuple)):
            return "visual_servo" if "visual_servo" in tools else "visual_align_and_insert"
        return "visual_servo"

    def visual_servo(
        self,
        goal: str,
        arm: str = "right",
        frame: str = "world",
        max_rounds: int = 8,
        max_steps: int = 40,
        reserve_steps: int = 20,
        stop_on_stall: bool = True,
        crop_wrist: bool = True,
        cameras: Sequence[str] | None = None,
        model: str | None = None,
        base_url: str | None = None,
        policy: Any = None,
        *,
        stop_on_contact_complete: bool = False,
        max_linear_m: float = 0.03,
        max_angular_deg: float = 15.0,
    ) -> VisualAlign:
        """General closed-loop visual-delta servoing alias for ``visual_align_and_insert`` (grasp, hang, plug, insert, press)."""
        kwargs = self._align_kwargs(goal, arm, frame, max_rounds, max_steps, reserve_steps,
                                    stop_on_stall, crop_wrist, cameras, model, base_url,
                                    stop_on_contact_complete, max_linear_m, max_angular_deg)
        if policy is not None:
            if isinstance(policy, (str, bytes)):
                raise ValueError("policy must be a callable or an object with predict()")
            payload = self._local_visual_align(kwargs, policy)
        else:
            tool = self._visual_servo_tool()
            try:
                payload = self._stateful(tool, kwargs, lambda p: p)
            except ToolError as exc:
                msg = str(exc).lower()
                if tool == "visual_servo" and not self._pending_actions and (
                    "no tool" in msg or "unknown tool" in msg
                ):
                    payload = self._stateful("visual_align_and_insert", kwargs, lambda p: p)
                else:
                    raise
        return self._align_result(payload)

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> Gripper:
        """Close (``robo.CLOSE=1``) or open (``robo.OPEN=-1``) one gripper without moving arms (2-10 actions).

        Optional ``opening`` in ``(0, 1]`` goes with ``robo.OPEN`` (jaw gap ~= ``0.002 + 0.109 * opening`` m).
        ``holding`` is not proof of identity: confirm lift in wrist view or ``locate``.
        """
        arm = _arm(arm)
        command = _command(gripper, allow_keep=False)
        kwargs: dict[str, Any] = {"arm": arm, "gripper": command}
        if opening is not None:
            try:
                fraction = float(opening)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"opening must be a number in (0, 1], got {opening!r}") from exc
            if command != -1 or not 0.0 < fraction <= 1.0:
                raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
            kwargs["opening"] = fraction

        def build(p: dict[str, Any]) -> Gripper:
            status = p.get("status") or {}
            state = self._latest_state()
            width = p.get("width_m")
            return Gripper(
                arm=arm,
                commanded=str(p.get("gripper") or ("closed" if command > 0 else "open")),
                opening=float(p.get("opening", state.arm(arm).gripper)),
                holding=bool(p.get("holding")),
                width_m=float(width) if width is not None else None,
                steps_used=int(p.get("steps_used", 0)),
                step=int(status["step"]) if status.get("step") is not None else None,
                step_limit=(
                    int(status["step_limit"]) if status.get("step_limit") is not None else None
                ),
                truncated=state.truncated,
            )

        return self._stateful("grip", kwargs, build)

    def home(self, arm: str = "both") -> Home:
        """Return ``"left"``, ``"right"``, or ``"both"`` to start joints (<=25 swing actions; retains gripper command).

        Near a recorded release, withdraws 6 cm along ``-x_tcp`` and lifts clear before swinging
        (no scene collision checking; clear tall obstacles first). Check ``reached``.
        """
        if arm not in (*self.ARMS, "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        which = str(arm)

        def build(p: dict[str, Any]) -> Home:
            return Home(
                arm=str(p.get("arm", which)),
                reached=bool(p.get("reached")),
                max_joint_error_rad=float(p.get("max_joint_error_rad", math.inf)),
                steps_used=int(p.get("steps_used", 0)),
                truncated=self._latest_state().truncated,
                stop_reason=p.get("stop_reason"),
                note=p.get("note"),
                clearance_retreat_m=dict(p.get("clearance_retreat_m") or {}),
            )

        return self._stateful("home", {"arm": which}, build)

    def pose_from_axes(
        self,
        xyz: Sequence[float],
        point_dir: Sequence[float],
        close_dir: Sequence[float] | None = None,
        *,
        roll_deg: float = 0.0,
    ) -> Pose:
        """Build world TCP ``Pose`` from ``point_dir`` (``+x_tcp``) and optional ``close_dir`` (``+y_tcp``), plus ``roll_deg`` about ``point_dir`` (0 steps)."""
        import numpy as np

        from .hosts import robodojo_kinematics as kin

        pos = _floats(xyz, 3, "xyz")
        p_vec = np.asarray(_floats(point_dir, 3, "point_dir"), dtype=float)
        p_norm = float(np.linalg.norm(p_vec))
        if p_norm < 1e-9:
            raise ValueError("point_dir must be nonzero")
        x_axis = p_vec / p_norm

        if close_dir is None:
            horiz = math.hypot(float(x_axis[0]), float(x_axis[1]))
            if horiz < math.sin(math.radians(1.0)):
                y_axis = np.array([-1.0, 0.0, 0.0], dtype=float)
                y_axis = y_axis - float(np.dot(y_axis, x_axis)) * x_axis
                y_axis /= float(np.linalg.norm(y_axis))
            else:
                y_axis = np.array([-x_axis[1] / horiz, x_axis[0] / horiz, 0.0], dtype=float)
        else:
            c_vec = np.asarray(_floats(close_dir, 3, "close_dir"), dtype=float)
            c_proj = c_vec - float(np.dot(c_vec, x_axis)) * x_axis
            c_norm = float(np.linalg.norm(c_proj))
            if c_norm < 1e-6:
                raise ValueError("close_dir must not be parallel to point_dir")
            y_axis = c_proj / c_norm

        z_axis = np.cross(x_axis, y_axis)
        z_axis /= float(np.linalg.norm(z_axis))
        rot = np.column_stack((x_axis, y_axis, z_axis))
        roll_f = _angle(roll_deg, "roll_deg")
        assert roll_f is not None
        if roll_f != 0.0:
            rot = rot @ kin._rot_x(math.radians(roll_f))
        return Pose(pos, kin.quat_wxyz(rot))

    def fk(
        self,
        arm: str = "right",
        joints: Sequence[float] | None = None,
    ) -> Pose:
        """Local URDF forward kinematics to world TCP ``Pose`` (0 steps; ``joints=None`` uses latest state)."""
        from .hosts import robodojo_kinematics as kin

        which = _arm(arm)
        q = _floats(self._latest_state().arm(which).joints if joints is None else joints, 6, "joints")
        t = kin.fk(which, q)
        tcp_xyz = kin.tcp(t)
        return Pose((float(tcp_xyz[0]), float(tcp_xyz[1]), float(tcp_xyz[2])), kin.quat_wxyz(t[:3, :3]))

    def ik(
        self,
        arm: str,
        target: Any,
        *,
        seed: Sequence[float] | None = None,
        jaw_symmetry: bool = False,
        continuous: bool = False,
        pos_tol_m: float = 0.002,
        rot_tol_deg: float = 1.0,
        other_joints: Sequence[float] | None = None,
        check_other_arm: bool = True,
        opening: float | None = None,
        twist_rad: float = 0.0,
    ) -> IKSolution:
        """Solve local URDF IK (0 steps) with table, inter-arm envelope, camera-housing, and twist checks.

        ``target`` takes ``Pose``, ``{"xyz": ..., "angles": [p, y, r]}``, ``{"xyz": ..., "quat": ...}``,
        or ``xyz``. ``jaw_symmetry=True`` evaluates 0/180 deg empty-jaw roll, preferring camera-housing
        away from the other arm and ``twist_rad`` headroom on ``joint6``. ``continuous=True`` checks
        straight-line SLERP from ``seed``. ``check_other_arm=True`` rejects inter-arm collisions against
        ``other_joints`` (or the other arm's recorded state).
        """
        import inspect

        import numpy as np

        from .hosts import robodojo_kinematics as kin

        which = _arm(arm)
        other_arm = "right" if which == "left" else "left"
        if (
            type(jaw_symmetry) is not bool
            or type(continuous) is not bool
            or type(check_other_arm) is not bool
        ):
            raise ValueError("jaw_symmetry, continuous, and check_other_arm must be bool")
        tol_m, tol_deg = float(pos_tol_m), float(rot_tol_deg)
        if not (math.isfinite(tol_m) and tol_m > 0.0 and math.isfinite(tol_deg) and tol_deg > 0.0):
            raise ValueError("pos_tol_m and rot_tol_deg must be positive finite numbers")
        if isinstance(twist_rad, bool):
            raise ValueError(f"twist_rad must be a finite number, got {twist_rad!r}")
        twist_f = _floats((twist_rad,), 1, "twist_rad")[0]
        op_val: float | None = None
        if opening is not None:
            if isinstance(opening, bool):
                raise ValueError(f"opening must be in [0, 1], got {opening!r}")
            op_val = _floats((opening,), 1, "opening")[0]
            if not 0.0 <= op_val <= 1.0:
                raise ValueError(f"opening must be in [0, 1], got {opening!r}")

        state_obj: State | None = None
        if seed is not None:
            cur = _floats(seed, 6, "seed")
            state_obj = self._last_state
            if state_obj is None and check_other_arm and other_joints is None:
                try:
                    state_obj = self._latest_state()
                except Exception:  # noqa: BLE001 - optional when offline seed is explicit
                    state_obj = None
        else:
            state_obj = self._latest_state()
            cur = _floats(state_obj.arm(which).joints, 6, "seed")

        other_q: tuple[float, ...] | None = None
        if check_other_arm:
            if other_joints is not None:
                other_q = _floats(other_joints, 6, "other_joints")
            elif state_obj is not None and len(state_obj.arm(other_arm).joints) == 6:
                other_q = _floats(state_obj.arm(other_arm).joints, 6, "other_joints")

        if op_val is not None:
            arm_opening = op_val
        elif state_obj is not None and math.isfinite(state_obj.arm(which).gripper):
            arm_opening = max(0.0, min(1.0, float(state_obj.arm(which).gripper)))
        else:
            arm_opening = 0.5
        if state_obj is not None and math.isfinite(state_obj.arm(other_arm).gripper):
            other_opening = max(0.0, min(1.0, float(state_obj.arm(other_arm).gripper)))
        else:
            other_opening = 0.5

        start_t = kin.fk(which, cur)
        goal = np.eye(4, dtype=float)
        if isinstance(target, Pose):
            goal[:3, 3] = target.xyz
            goal[:3, :3] = kin.rot_from_quat(target.quat)
        elif isinstance(target, dict):
            if "angles" in target:
                if set(target) != {"xyz", "angles"}:
                    raise ValueError("angle target needs only xyz and angles")
                goal[:3, 3] = _floats(target["xyz"], 3, "target xyz")
                goal[:3, :3] = kin.orientation(*_floats(target["angles"], 3, "target angles"))
            elif "xyz" in target and "quat" in target:
                p = Pose(target["xyz"], target["quat"])
                goal[:3, 3] = p.xyz
                goal[:3, :3] = kin.rot_from_quat(p.quat)
            else:
                raise ValueError("dict target must provide (xyz, angles) or (xyz, quat)")
        else:
            goal[:3, 3] = _floats(target, 3, "target xyz")
            goal[:3, :3] = start_t[:3, :3]

        candidates = [(goal, False)]
        if jaw_symmetry:
            flipped_goal = goal.copy()
            flipped_goal[:3, :3] = goal[:3, :3] @ kin._rot_x(math.pi)
            candidates.append((flipped_goal, True))

        inter_fn = getattr(kin, "inter_arm_violations", None)
        housing_fn = getattr(kin, "housing_exposure_penalty", None)
        twist_fn = getattr(kin, "twist_margin_rad", None)
        best_params = inspect.signature(kin.best_solution).parameters

        def _check_inter(q_arm: Sequence[float]) -> list[str]:
            if not (check_other_arm and other_q is not None and callable(inter_fn)):
                return []
            if which == "left":
                return list(
                    inter_fn(
                        q_arm,
                        other_q,
                        left_opening=arm_opening,
                        right_opening=other_opening,
                    )
                )
            return list(
                inter_fn(
                    other_q,
                    q_arm,
                    left_opening=other_opening,
                    right_opening=arm_opening,
                )
            )

        weights = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5], dtype=float)
        cur_arr = np.asarray(cur, dtype=float)
        evaluated: list[tuple[tuple[Any, ...], IKSolution]] = []
        for cand_goal, is_flipped in candidates:
            inter_violations: list[str] = []
            if continuous:
                ways, why = kin.line_path(
                    which, cur, cand_goal, pos_tol=tol_m, rot_tol_deg=tol_deg
                )
                if why is None:
                    q_sol = tuple(float(v) for v in (ways[-1] if ways else cur))
                    end = kin.fk(which, q_sol)
                    pos_err = float(np.linalg.norm(cand_goal[:3, 3] - kin.tcp(end)))
                    rot_err = kin.rotation_error(cand_goal[:3, :3], end[:3, :3])
                    table_collides = bool(kin.clearance_violations(which, q_sol))
                    for wq in (ways or [q_sol]):
                        v_list = _check_inter(wq)
                        if v_list:
                            inter_violations = v_list
                            break
                    collides = table_collides or bool(inter_violations)
                    ok = pos_err <= tol_m and rot_err <= tol_deg and not collides
                    if ok:
                        reason = None
                    elif inter_violations:
                        reason = f"inter_arm_collision:{','.join(inter_violations)}"
                    elif table_collides:
                        reason = "collides_with_table"
                    else:
                        reason = "ik_tolerance"
                else:
                    last_q = ways[-1] if ways else cur
                    res = kin.solve(which, cand_goal, last_q, pos_tol=tol_m, rot_tol_deg=tol_deg)
                    q_sol = res.q
                    pos_err, rot_err = res.pos_err_m, res.rot_err_deg
                    inter_violations = _check_inter(q_sol)
                    collides = bool(res.collides or "into the table" in why or inter_violations)
                    ok = False
                    reason = why
            else:
                best_kwargs: dict[str, Any] = {
                    "pos_tol": tol_m,
                    "rot_tol_deg": tol_deg,
                    "extra_seeds": [(0.0,) * 6],
                }
                if "other_arm_q" in best_params and other_q is not None:
                    best_kwargs["other_arm_q"] = other_q
                if "opening" in best_params:
                    best_kwargs["opening"] = arm_opening
                if "other_opening" in best_params:
                    best_kwargs["other_opening"] = other_opening
                if "twist_rad" in best_params:
                    best_kwargs["twist_rad"] = twist_f
                res = kin.best_solution(which, cand_goal, cur, **best_kwargs)
                q_sol = res.q
                pos_err, rot_err = res.pos_err_m, res.rot_err_deg
                inter_violations = _check_inter(q_sol)
                collides = bool(res.collides or inter_violations)
                ok = bool(res.ok and not collides)
                if ok:
                    reason = None
                elif inter_violations:
                    reason = f"inter_arm_collision:{','.join(inter_violations)}"
                elif collides:
                    reason = "collides_with_table"
                else:
                    reason = "unreachable"

            end_fk = kin.fk(which, q_sol)
            end_tcp = kin.tcp(end_fk)
            sol_pose = Pose(
                (float(end_tcp[0]), float(end_tcp[1]), float(end_tcp[2])),
                kin.quat_wxyz(end_fk[:3, :3]),
            )
            margins = tuple(
                float(min(qv - lo, hi - qv))
                for qv, (lo, hi) in zip(q_sol, kin.JOINT_LIMITS, strict=True)
            )
            min_margin = min(margins)
            if callable(twist_fn):
                t_margin = float(twist_fn(q_sol, twist_rad=twist_f))
            else:
                lo5, hi5 = kin.JOINT_LIMITS[5]
                q5 = float(q_sol[5])
                t_margin = min(q5 - lo5, hi5 - q5, (q5 + twist_f) - lo5, hi5 - (q5 + twist_f))
            housing_pen = (
                float(housing_fn(which, q_sol, other_q))
                if (callable(housing_fn) and other_q is not None)
                else 0.0
            )
            q_arr = np.asarray(q_sol, dtype=float)
            max_delta = float(np.max(np.abs(q_arr - cur_arr)))
            weighted = float(np.sum(weights * np.abs(q_arr - cur_arr)))
            twist_insufficient = twist_f != 0.0 and t_margin < 0.0
            twist_tight = twist_f != 0.0 and t_margin < 0.25
            geom_ok = ok
            if ok and twist_insufficient:
                ok = False
                reason = "insufficient_twist_margin"
            score = (
                not ok,
                not geom_ok,
                collides,
                twist_insufficient,
                twist_tight,
                min_margin < 0.35,
                (
                    weighted
                    + 4.0 * housing_pen
                    - 0.5 * min(min_margin, 1.0)
                    - (0.3 * min(max(t_margin, 0.0), 1.0) if twist_f != 0.0 else 0.0)
                )
                if geom_ok
                else (pos_err + 0.1 * math.radians(rot_err)),
                is_flipped,
            )
            evaluated.append(
                (
                    score,
                    IKSolution(
                        ok=ok,
                        collides=collides,
                        joints=q_sol,
                        pose=sol_pose,
                        pos_err_m=pos_err,
                        rot_err_deg=rot_err,
                        min_margin_rad=min_margin,
                        joint_margins_rad=margins,
                        max_delta_rad=max_delta,
                        flipped_jaw=is_flipped,
                        reason=reason,
                    ),
                )
            )
        return min(evaluated, key=lambda item: item[0])[1]

    # -- internals ----------------------------------------------------------

    def _latest_state(self) -> State:
        if self._last_state is None:
            return self.state()
        return self._last_state

    def _move(self, arm: str, target: tuple[float, ...], p: dict[str, Any]) -> Move:
        commanded = p.get("target_xyz")
        angles = p.get("final_angles") or {}
        steps = int(p.get("steps_used", 0))
        reached = bool(p.get("reached"))
        planned = p.get("planned_path", p.get("path"))
        reason = p.get("stop_reason")
        if reason is None and not reached:  # older geometric hosts
            reason = "ik_path_rejected" if p.get("ik_error") else "final_target_not_reached"
        return Move(
            arm=str(p.get("arm", arm)),
            target_xyz=_tuple3(commanded if commanded is not None else target),
            start_xyz=_tuple3(p.get("start_xyz")),
            final_xyz=_tuple3(p.get("final_xyz")),
            dist_to_target_m=float(p.get("dist_to_target_m", math.inf)),
            orientation_error_deg=float(p.get("orientation_error_deg", math.nan)),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            reached=reached,
            path=str(p["path"]) if steps and p.get("path") else None,
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            blocked=bool(p.get("blocked")),
            steps_used=steps,
            truncated=self._latest_state().truncated,
            trajectory=p.get("trajectory"),
            status=p.get("status") or ("completed" if reached else
                                       "partial" if steps else "rejected"),
            planned_path=str(planned) if planned else None,
            stop_reason=reason,
            tracking=p.get("tracking"),
        )

    # -- guarded nudges and the visual delta loop (internals) -------------------

    @staticmethod
    def _bounded(value: Any, name: str, high: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number, got {value!r}") from exc
        if isinstance(value, bool) or not math.isfinite(out) or not 0 < out <= high:
            raise ValueError(f"{name} must be in (0, {high}], got {value!r}")
        return out

    @staticmethod
    def _count(value: Any, name: str, low: int, high: int | None = None) -> int:
        if type(value) is not int or value < low or (high is not None and value > high):
            span = f"[{low}, {high}]" if high is not None else f">= {low}"
            raise ValueError(f"{name} must be an integer {span}, got {value!r}")
        return value

    def _delta_spec(self, spec: Any, name: str) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError(f"{name} must be a dict with dxyz / dpitch / dyaw / droll / gripper")
        extra = sorted(set(spec) - {"dxyz", "dpitch", "dyaw", "droll", "gripper"})
        if extra:
            raise ValueError(f"{name} has unknown keys {extra}")
        out: dict[str, Any] = {"dxyz": list(_floats(spec.get("dxyz", (0, 0, 0)), 3,
                                                    f"{name}.dxyz"))}
        for key in ("dpitch", "dyaw", "droll"):
            out[key] = _angle(spec.get(key, 0.0), f"{name}.{key}") or 0.0
        out["gripper"] = _command(spec.get("gripper", 0), allow_keep=True)
        return out

    def _nudge_kwargs(self, dxyz, arm, dpitch, dyaw, droll, gripper, frame, left, right,
                      stop_on_stall, stall_steps, max_steps, reserve_steps, compact,
                      max_linear_m, max_angular_deg,
                      allow_table_contact: bool = False) -> dict[str, Any]:
        if arm not in (*self.ARMS, "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        if frame not in ("world", "tcp"):
            raise ValueError(f"frame must be 'world' or 'tcp', got {frame!r}")
        if (type(stop_on_stall) is not bool or type(compact) is not bool
                or type(allow_table_contact) is not bool):
            raise ValueError("stop_on_stall, compact and allow_table_contact must be bool")
        if allow_table_contact and not compact:
            raise ValueError("allow_table_contact=True requires compact=True")
        kwargs: dict[str, Any] = dict(
            arm=arm, frame=frame, stop_on_stall=stop_on_stall, compact=compact,
            allow_table_contact=allow_table_contact,
            stall_steps=self._count(stall_steps, "stall_steps", 1, 5),
            reserve_steps=self._count(reserve_steps, "reserve_steps", 0),
            max_linear_m=self._bounded(max_linear_m, "max_linear_m", 0.12),
            max_angular_deg=self._bounded(max_angular_deg, "max_angular_deg", 45.0))
        if max_steps is not None:
            kwargs["max_steps"] = self._count(max_steps, "max_steps", 1)
        single = self._delta_spec(dict(dxyz=dxyz, dpitch=dpitch, dyaw=dyaw, droll=droll,
                                       gripper=gripper), "delta")
        if arm == "both":
            if any(single["dxyz"]) or single["gripper"] or any(
                    single[k] for k in ("dpitch", "dyaw", "droll")):
                raise ValueError("arm='both' takes its deltas as left={...} / right={...}")
            if left is None and right is None:
                raise ValueError("arm='both' needs left={...} and/or right={...}")
            for name, spec in (("left", left), ("right", right)):
                if spec is not None:
                    kwargs[name] = self._delta_spec(spec, name)
        else:
            if left is not None or right is not None:
                raise ValueError("left= / right= deltas go with arm='both'")
            kwargs.update(single)
        return kwargs

    def _nudge_result(self, kwargs: dict[str, Any], p: dict[str, Any]) -> Nudge:
        arms = dict(p.get("arms") or {})
        angles = p.get("final_angles") or {}
        achieved = (p.get("achieved_delta") or {}).get(kwargs["frame"]) or {}
        dists = [float(a.get("dist_to_target_m", math.nan)) for a in arms.values()]
        turns = [float(a.get("orientation_error_deg", math.nan)) for a in arms.values()]
        return Nudge(
            arm=str(p.get("arm", kwargs["arm"])),
            frame=str(p.get("frame", kwargs["frame"])),
            status=str(p.get("status") or "rejected"),
            reached=bool(p.get("reached")),
            stalled=bool(p.get("stalled")),
            stop_reason=p.get("stop_reason"),
            steps_used=int(p.get("steps_used", 0)),
            planned_steps=int(p.get("planned_steps", 0)),
            final_xyz=_tuple3(p.get("final_xyz")),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            dist_to_target_m=float(p.get("dist_to_target_m", max(dists, default=math.nan))),
            orientation_error_deg=float(p.get("orientation_error_deg",
                                              max(turns, default=math.nan))),
            achieved=dict(achieved),
            holding=bool(p.get("holding")),
            opening=float(p.get("opening", math.nan)),
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            truncated=self._latest_state().truncated,
            arms=arms,
        )

    def _delta_grasp_kwargs(
        self, target, arm, axis, standoff_m, overdrive_m, opening, contact_table,
        lift_dxyz, align_tol_m, max_linear_m, max_steps, reserve_steps,
    ) -> dict[str, Any]:
        which = _arm(arm)
        t_xyz = _extract_xyz(target, "target")
        ax = _floats(axis, 3, "axis")
        if math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]) <= 1e-9:
            raise ValueError("axis must be non-zero")
        if type(contact_table) is not bool:
            raise ValueError("contact_table must be bool")
        if isinstance(standoff_m, bool):
            raise ValueError(f"standoff_m must be >= 0, got {standoff_m!r}")
        s_val = _floats((standoff_m,), 1, "standoff_m")[0]
        if s_val < 0.0:
            raise ValueError(f"standoff_m must be >= 0, got {standoff_m!r}")
        if isinstance(overdrive_m, bool):
            raise ValueError(f"overdrive_m must be >= 0, got {overdrive_m!r}")
        o_val = _floats((overdrive_m,), 1, "overdrive_m")[0]
        if o_val < 0.0:
            raise ValueError(f"overdrive_m must be >= 0, got {overdrive_m!r}")
        tol = self._bounded(align_tol_m, "align_tol_m", 0.02)
        if tol < 0.0005:
            raise ValueError(f"align_tol_m must be in [0.0005, 0.02], got {align_tol_m!r}")
        kwargs: dict[str, Any] = {
            "target_xyz": list(t_xyz),
            "arm": which,
            "axis": list(ax),
            "standoff_m": s_val,
            "overdrive_m": o_val,
            "contact_table": contact_table,
            "align_tol_m": tol,
            "max_linear_m": self._bounded(max_linear_m, "max_linear_m", 0.12),
            "max_steps": self._count(max_steps, "max_steps", 1),
            "reserve_steps": self._count(reserve_steps, "reserve_steps", 0),
        }
        if opening is not None:
            if isinstance(opening, bool):
                raise ValueError(f"opening must be in (0, 1], got {opening!r}")
            op_val = _floats((opening,), 1, "opening")[0]
            if not 0.0 < op_val <= 1.0:
                raise ValueError(f"opening must be in (0, 1], got {opening!r}")
            kwargs["opening"] = op_val
        if lift_dxyz is not None:
            lv = _floats(lift_dxyz, 3, "lift_dxyz")
            if math.sqrt(lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]) <= 1e-6:
                raise ValueError("lift_dxyz must be non-zero")
            kwargs["lift_dxyz"] = list(lv)
        return kwargs

    def _delta_grasp_result(
        self, kwargs: dict[str, Any], p: dict[str, Any], *, preview: bool
    ) -> DeltaGrasp:
        angles = p.get("final_angles") or {}
        open_pre = p.get("opening_after_preopen")
        closed_op = p.get("closed_opening")
        open_lift = p.get("opening_after_lift")
        width = p.get("width_m")
        return DeltaGrasp(
            arm=str(p.get("arm", kwargs["arm"])),
            status=str(p.get("status") or "rejected"),
            aligned=bool(p.get("aligned")),
            reached=bool(p.get("reached")),
            accepted=bool(p.get("accepted")),
            preview=bool(p.get("preview", preview)),
            stalled=bool(p.get("stalled")),
            table_contact_stalled=bool(p.get("table_contact_stalled")),
            stop_reason=p.get("stop_reason"),
            target_xyz=_tuple3(p.get("target_xyz", kwargs["target_xyz"])),
            axis=_tuple3(p.get("axis", kwargs["axis"])),
            standoff_m=float(p.get("standoff_m", kwargs["standoff_m"])),
            overdrive_m=float(p.get("overdrive_m", kwargs["overdrive_m"])),
            contact_table=bool(p.get("contact_table", kwargs["contact_table"])),
            transverse_error_before_m=float(p.get("transverse_error_before_m", math.nan)),
            transverse_error_after_m=float(p.get("transverse_error_after_m", math.nan)),
            opening_before=float(p.get("opening_before", math.nan)),
            opening_after_preopen=float(open_pre) if open_pre is not None else None,
            closed=bool(p.get("closed")),
            closed_opening=float(closed_op) if closed_op is not None else None,
            opening_after_lift=float(open_lift) if open_lift is not None else None,
            retained_closure=bool(p.get("retained_closure")),
            holding=bool(p.get("holding")),
            opening=float(p.get("opening", math.nan)),
            width_m=float(width) if width is not None else None,
            lift_status=str(p["lift_status"]) if p.get("lift_status") is not None else None,
            preopen_steps_used=int(p.get("preopen_steps_used", 0)),
            align_steps_used=int(p.get("align_steps_used", 0)),
            approach_steps_used=int(p.get("approach_steps_used", 0)),
            close_steps_used=int(p.get("close_steps_used", 0)),
            lift_steps_used=int(p.get("lift_steps_used", 0)),
            steps_used=int(p.get("steps_used", 0)),
            planned_steps=int(p.get("planned_steps", 0)),
            final_xyz=_tuple3(p.get("final_xyz")),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            truncated=self._latest_state().truncated,
        )

    def _align_feature_kwargs(
        self, feature, target, arm, axis, standoff_m, stroke_m, stroke_dxyz,
        align_tol_m, max_linear_m, max_steps, reserve_steps,
        stop_on_stall, stall_steps, allow_table_contact: bool = False,
    ) -> dict[str, Any]:
        which = _arm(arm)
        f_xyz = _extract_xyz(feature, "feature")
        t_xyz = _extract_xyz(target, "target")
        ax = _floats(axis, 3, "axis")
        if math.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]) <= 1e-9:
            raise ValueError("axis must be non-zero")
        if type(stop_on_stall) is not bool or type(allow_table_contact) is not bool:
            raise ValueError("stop_on_stall and allow_table_contact must be bool")
        tol = self._bounded(align_tol_m, "align_tol_m", 0.02)
        if tol < 0.0005:
            raise ValueError(f"align_tol_m must be in [0.0005, 0.02], got {align_tol_m!r}")
        kwargs: dict[str, Any] = {
            "feature_xyz": list(f_xyz),
            "target_xyz": list(t_xyz),
            "arm": which,
            "axis": list(ax),
            "align_tol_m": tol,
            "max_linear_m": self._bounded(max_linear_m, "max_linear_m", 0.12),
            "max_steps": self._count(max_steps, "max_steps", 1),
            "reserve_steps": self._count(reserve_steps, "reserve_steps", 0),
            "stop_on_stall": stop_on_stall,
            "stall_steps": self._count(stall_steps, "stall_steps", 1, 5),
            "allow_table_contact": allow_table_contact,
        }
        if standoff_m is not None:
            if isinstance(standoff_m, bool):
                raise ValueError(f"standoff_m must be a number >= 0, got {standoff_m!r}")
            s_val = _floats((standoff_m,), 1, "standoff_m")[0]
            if s_val < 0.0:
                raise ValueError(f"standoff_m must be >= 0, got {standoff_m!r}")
            kwargs["standoff_m"] = s_val
        if stroke_m is not None and stroke_dxyz is not None:
            raise ValueError("pass at most one of stroke_m or stroke_dxyz")
        if stroke_m is not None:
            if isinstance(stroke_m, bool):
                raise ValueError(f"stroke_m must be > 0, got {stroke_m!r}")
            st_val = _floats((stroke_m,), 1, "stroke_m")[0]
            if st_val <= 0.0:
                raise ValueError(f"stroke_m must be > 0, got {stroke_m!r}")
            kwargs["stroke_m"] = st_val
        elif stroke_dxyz is not None:
            sv = _floats(stroke_dxyz, 3, "stroke_dxyz")
            if math.sqrt(sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2]) <= 1e-6:
                raise ValueError("stroke_dxyz must be non-zero")
            kwargs["stroke_dxyz"] = list(sv)
        return kwargs

    def _align_feature_result(
        self, kwargs: dict[str, Any], p: dict[str, Any], *, preview: bool
    ) -> FeatureAlignment:
        angles = p.get("final_angles") or {}
        req_stroke = p.get("requested_stroke_m")
        ach_stroke = p.get("achieved_stroke_m")
        standoff = p.get("standoff_m")
        sb = p.get("suggested_backoff_dxyz")
        return FeatureAlignment(
            arm=str(p.get("arm", kwargs["arm"])),
            status=str(p.get("status") or "rejected"),
            aligned=bool(p.get("aligned")),
            reached=bool(p.get("reached")),
            accepted=bool(p.get("accepted")),
            preview=bool(p.get("preview", preview)),
            stalled=bool(p.get("stalled")),
            stop_reason=p.get("stop_reason"),
            feature_xyz=_tuple3(p.get("feature_xyz", kwargs["feature_xyz"])),
            target_xyz=_tuple3(p.get("target_xyz", kwargs["target_xyz"])),
            axis=_tuple3(p.get("axis", kwargs["axis"])),
            axial_gap_m=float(p.get("axial_gap_m", math.nan)),
            transverse_delta=_tuple3(p.get("transverse_delta")),
            planned_dxyz=_tuple3(p.get("planned_dxyz")),
            transverse_error_before_m=float(p.get("transverse_error_before_m", math.nan)),
            transverse_error_after_m=float(p.get("transverse_error_after_m", math.nan)),
            standoff_m=float(standoff) if standoff is not None else None,
            stroke_status=str(p["stroke_status"]) if p.get("stroke_status") is not None else None,
            requested_stroke_m=float(req_stroke) if req_stroke is not None else None,
            achieved_stroke_m=float(ach_stroke) if ach_stroke is not None else None,
            align_steps_used=int(p.get("align_steps_used", 0)),
            stroke_steps_used=int(p.get("stroke_steps_used", 0)),
            steps_used=int(p.get("steps_used", 0)),
            planned_steps=int(p.get("planned_steps", 0)),
            final_xyz=_tuple3(p.get("final_xyz")),
            pitch=float(angles.get("pitch", math.nan)),
            yaw=float(angles.get("yaw", math.nan)),
            roll=float(angles.get("roll", math.nan)),
            holding=bool(p.get("holding")),
            opening=float(p.get("opening", math.nan)),
            ik_error=str(p["ik_error"]) if p.get("ik_error") else None,
            note=str(p["note"]) if p.get("note") else None,
            truncated=self._latest_state().truncated,
            rim_contact_suspected=bool(p.get("rim_contact_suspected")),
            suggested_backoff_dxyz=_tuple3(sb) if sb is not None else None,
        )

    def _align_kwargs(self, goal, arm, frame, max_rounds, max_steps, reserve_steps,
                      stop_on_stall, crop_wrist, cameras, model, base_url,
                      stop_on_contact_complete, max_linear_m, max_angular_deg) -> dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if arm not in (*self.ARMS, "both"):
            raise ValueError(f"arm must be 'left', 'right' or 'both', got {arm!r}")
        if frame not in ("world", "tcp"):
            raise ValueError(f"frame must be 'world' or 'tcp', got {frame!r}")
        for name, flag in (("stop_on_stall", stop_on_stall), ("crop_wrist", crop_wrist),
                           ("stop_on_contact_complete", stop_on_contact_complete)):
            if type(flag) is not bool:
                raise ValueError(f"{name} must be bool")
        kwargs: dict[str, Any] = dict(
            goal=goal, arm=arm, frame=frame,
            max_rounds=self._count(max_rounds, "max_rounds", 1, 32),
            max_steps=self._count(max_steps, "max_steps", 1, 400),
            reserve_steps=self._count(reserve_steps, "reserve_steps", 0),
            stop_on_stall=stop_on_stall, crop_wrist=crop_wrist,
            stop_on_contact_complete=stop_on_contact_complete,
            max_linear_m=self._bounded(max_linear_m, "max_linear_m", 0.12),
            max_angular_deg=self._bounded(max_angular_deg, "max_angular_deg", 45.0))
        if cameras is not None:
            kwargs["cameras"] = [_camera(c, self.CAMERAS) for c in cameras]
        for name, value in (("model", model), ("base_url", base_url)):
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{name} must be a non-empty string or None")
                kwargs[name] = value
        return kwargs

    def _local_visual_align(self, kwargs: dict[str, Any], policy: Any) -> dict[str, Any]:
        """The visual delta loop in this process: a Python policy, one nudge per move."""

        from . import visual_delta as vd

        arm = kwargs["arm"]
        arms = self.ARMS if arm == "both" else (arm,)
        views = kwargs.get("cameras") or (
            list(self.CAMERAS) if arm == "both" else ["cam_head", f"cam_{arm}_wrist"])
        chosen = vd.as_policy(policy)
        last: dict[str, Any] = {}

        def poses(state: State) -> dict[str, dict[str, Any]]:
            return {name: {"xyz": list(state.arm(name).tcp_pos), "pitch": state.arm(name).pitch,
                           "yaw": state.arm(name).yaw, "roll": state.arm(name).roll,
                           "opening": state.arm(name).gripper} for name in arms}

        def observe(images: bool = True) -> dict[str, Any]:
            shots = []
            for camera in views if images else ():
                try:
                    picture = self.image(camera)
                except Exception:  # noqa: BLE001 - a missing view is left out
                    continue
                shots.append({"label": camera, "png": vd.encode_png(picture)})
                if kwargs["crop_wrist"] and camera.endswith("_wrist"):
                    from .hosts.robodojo_visual_delta import crop_center

                    shots.append({"label": f"{camera}_crop",
                                  "png": vd.encode_png(crop_center(picture))})
            return {"poses": poses(self.state()), "images": shots,
                    "remaining_steps": last.get("remaining_episode_steps")}

        def execute(specs: dict[str, dict[str, Any]], steps_left: int) -> dict[str, Any]:
            call: dict[str, Any] = dict(
                arm=arm, frame=kwargs["frame"], stop_on_stall=kwargs["stop_on_stall"],
                stall_steps=2, compact=True, max_steps=steps_left,
                reserve_steps=kwargs["reserve_steps"], max_linear_m=kwargs["max_linear_m"],
                max_angular_deg=kwargs["max_angular_deg"])
            if arm == "both":
                call.update({name: dict(spec) for name, spec in specs.items()})
            else:
                call.update(specs[arm])
            try:
                payload = self._stateful("nudge", call, lambda p: p)
            except ToolError as exc:
                return {"status": "rejected", "stop_reason": "invalid_delta", "steps_used": 0,
                        "ik_error": str(exc)}
            last.clear()
            last.update(payload)
            return payload

        out = vd.run_visual_delta_loop(
            goal=kwargs["goal"], arm=arm, frame=kwargs["frame"], policy=chosen,
            observe=observe, execute=execute, step_allowance=kwargs["max_steps"],
            max_rounds=kwargs["max_rounds"], max_linear_m=kwargs["max_linear_m"],
            max_angular_deg=kwargs["max_angular_deg"],
            stop_on_contact_complete=kwargs["stop_on_contact_complete"],
            model=kwargs.get("model"))
        state = self.state()
        out.update(local=True, reserve_steps=kwargs["reserve_steps"], cameras=list(views),
                   stalled=any((r.get("feedback") or {}).get("stalled") for r in out["rounds"]))
        if arm != "both":
            out.update(final_xyz=list(state.arm(arm).tcp_pos),
                       holding=bool(last.get("holding")) if last else False,
                       opening=state.arm(arm).gripper)
        return out

    def _align_result(self, p: dict[str, Any]) -> VisualAlign:
        return VisualAlign(
            goal=str(p.get("goal", "")),
            arm=str(p.get("arm", "")),
            status=str(p.get("status") or "rejected"),
            reached=bool(p.get("reached")),
            done=bool(p.get("done")),
            converged=bool(p.get("converged")),
            contact_complete=bool(p.get("contact_complete")),
            stalled=bool(p.get("stalled")),
            stop_reason=p.get("stop_reason"),
            rounds_used=int(p.get("rounds_used", 0)),
            steps_used=int(p.get("steps_used", 0)),
            step_allowance=int(p.get("step_allowance", 0)),
            final_xyz=_tuple3(p.get("final_xyz")),
            holding=bool(p.get("holding")),
            policy=str(p["policy"]) if p.get("policy") else None,
            rounds=list(p.get("rounds") or []),
            last_feedback=p.get("last_feedback"),
            policy_error=str(p["policy_error"]) if p.get("policy_error") else None,
            truncated=self._latest_state().truncated,
        )

    def _state_from_envelope(self, envelope: dict[str, Any]) -> State:
        summary = envelope.get("state") or {}
        raw = summary.get("state") or {}
        return State(
            step=int(envelope.get("step", -1)),
            task=str(envelope.get("task_language") or summary.get("instruction") or ""),
            left=_arm_state(raw, "left"),
            right=_arm_state(raw, "right"),
            truncated=bool(envelope.get("truncated")),
        )

    def _absorb(self, envelope: dict[str, Any] | None) -> None:
        """Latch ``truncated`` only.

        RPent records ``terminated=False`` on purpose (official success is read
        by its runner after the planner); nothing an envelope says may end
        this episode as a success either.
        """
        if envelope is None:
            return
        state = self._state_from_envelope(envelope)
        self._last_state = state
        if state.truncated:
            self._truncated = True

    def _summary_dict(self) -> dict[str, Any]:
        """Compact state for the per-turn tool feedback (host-facing)."""
        state = self.state()
        out: dict[str, Any] = {"step": state.step}
        for name in self.ARMS:
            arm = state.arm(name)
            out[f"{name}_tcp"] = [round(v, 4) for v in arm.tcp_pos]
            out[f"{name}_pitch_yaw_roll"] = [
                round(arm.pitch, 1),
                round(arm.yaw, 1),
                round(arm.roll, 1),
            ]
            out[f"{name}_gripper"] = round(arm.gripper, 3)
        out["truncated"] = state.truncated
        return out


ADAPTER = RobotAdapter(
    name="robodojo",
    robot_cls=RobodojoRobot,
    host=f"{__package__ or 'codeaspolicy'}.hosts.rpent_robodojo",
    knowledge="robodojo",
    blurb="a dual-arm ARX-X5 robot in the RoboDojo benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a frozen Pi0.5 VLA and a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
    services_nosam="",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "CameraRay",
    "DeltaGrasp",
    "FeatureAlignment",
    "Gripper",
    "HOME_MAX_ACTIONS",
    "HOME_TOL_RAD",
    "Home",
    "IKSolution",
    "MOVE_TOL_DEG",
    "MOVE_TOL_M",
    "Move",
    "Nudge",
    "ObjectPose",
    "Pick",
    "PixelPoint",
    "PlanePoint",
    "RobodojoRobot",
    "STATEFUL_TOOLS",
    "Segment",
    "State",
    "TCP_OFFSET",
    "TriangulatedPoint",
    "VLA_CHUNK_ACTIONS",
    "VisualAlign",
    "WorldPoint",
    "gripper_angles",
]
