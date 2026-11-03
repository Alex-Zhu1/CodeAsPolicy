# Copyright 2026 PyRUA-Lean Contributors.
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
is generated from them (see :mod:`pyrualean.prompt`), so keep them accurate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ._robot import RobotBase
from .coordination import CoordinationMixin
from .motion import MotionProfile, MotionWindow, resolve_motion
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

    tcp_pos: the grasp point [x, y, z] in the world frame, metres: the middle of
        the finger pads, 0.145 m out along the pointing direction from the
        end-effector link.  ``move_to`` drives this point.
    pitch / yaw / roll: the gripper's orientation, degrees (see the class
        docstring): ``pitch`` 0 points level and 90 straight down, ``yaw`` is
        the heading of the pointing direction on the table (0 away from the
        robot, +90 towards the robot's left), ``roll`` the turn of the fingers
        about the pointing direction (0 as at the start).  Pointing straight
        down, ``yaw`` alone says how the fingers lie and ``roll`` reads 0.
    eef_pos: position of the end-effector link (the wrist flange), metres.
    eef_quat: end-effector orientation as a unit quaternion [w, x, y, z].
    gripper: measured gripper opening, normalised: 1.0 = open, 0.0 = closed on
        nothing; closed on an object 0.002 + 0.109 x this reading is the
        object's width in metres.
        (Gripper COMMANDS use the opposite sign convention: see ``set_gripper``.)
    joints: the six arm joint angles, radians (the last turns the fingers about
        the pointing direction).
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
    """Snapshot of both arms at one recorded step.

    step: index of the recorded step (0 = initial scene).  A new step is
        recorded after every motion call.
    task: the task instruction given by the benchmark (authoritative).
    left / right: the two arms (``state.arm("left")`` picks one by name).
    truncated: the native-action budget is exhausted or the environment has
        ended the episode; no further motion is possible. This is not a task
        success verdict.
    """

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
    """Outcome of ``move_to``.

    target_xyz: the requested grasp-point position.
    start_xyz / final_xyz: where the grasp point was before and after.
    dist_to_target_m: distance from ``final_xyz`` to ``target_xyz``.
    orientation_error_deg: error against the selected orientation; this may be
        an equivalent half-turn or nearby solution described in note.
    pitch / yaw / roll: the orientation the gripper ended with, degrees.
    reached: within 0.01 m of target_xyz and 5 degrees of the selected target
        orientation, without a timed tracking interruption.
    status: completed, rejected (zero native actions), or partial (actions
        executed but not reached). This is motion status, not task success.
    planned_path: selected geometry, "line" or "joint", even if timing rejects
        it; None if no path was selected.
    path: executed geometry, "line" or "joint"; None when steps_used is zero.
        A non-None path does not mean that the whole path completed.
    stop_reason: None on completion; otherwise ik_path_rejected,
        unsupported_timed_trajectory, timed_path_exceeds_native_step_budget,
        trajectory_validation_failed, tracking_error, episode_native_step_budget
        or final_target_not_reached. Inspect this before choosing a correction.
    ik_error: geometric preflight detail (unreachable pose/path or table
        collision), otherwise None. It is not used for timing/budget refusals.
        An unreachable target may name nearby reachable alternatives.
    note: extra planning/execution detail, including timing refusal or selection
        of a nearby pose (within 2 cm / 8 degrees) or equivalent empty-gripper
        orientation. It is not evidence that a selected pose was reached.
    blocked: the measured TCP is over 5 mm from the planned endpoint after
        execution, with budget remaining. This is a tracking-lag heuristic,
        not a contact sensor or proof of an obstacle; inspect the actual scene.
    steps_used: native actions consumed.
    truncated: the budget ran out or the environment ended the episode.
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


@dataclass(frozen=True)
class Home:
    """Outcome of ``home``.

    arm: ``"left"``, ``"right"`` or ``"both"``, as requested.
    reached: every joint of the returned arm(s) ended within 0.01 rad of the
        angle it had when the episode started.
    max_joint_error_rad: the largest difference that remains.
    steps_used: native actions consumed.
    truncated: the budget ran out or the environment ended the episode.
    """

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

    commanded: ``"closed"`` or ``"open"``.
    opening: the opening once the fingers stopped (1.0 = open, 0.0 = closed on
        nothing).
    holding: after a close, the fingers stopped apart (an opening above 0.01,
        about 3 mm): something is between them.  It does not say what: check
        that the intended object moves with the arm.
    width_m: after a close, the width of what is held, across the fingers
        (0.0 when nothing is), else None.
    steps_used: native actions consumed.
    step / step_limit: native actions executed so far and the episode budget
        (None if the runtime did not report them).
    truncated: the budget ran out or the environment ended the episode.
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
    """Outcome of ``pi0_pick``.

    success: the grasp HEURISTIC fired for some arm: it descended at least
        0.06 m, then rose at least 0.04 m with its gripper below 0.55.  It is
        provisional and errs both ways: it says nothing about which object is
        held, and it stays False for an arm that grasped without first going
        down 0.06 m.  Read ``left`` / ``right`` and confirm from the wrist camera.
    instruction: the text the policy received.
    chunks_used: policy chunks executed (each up to 50 native actions; a chunk
        is cut short the moment the heuristic fires).
    left / right: what each arm did, as a dict of end-effector heights (metres)
        and measured gripper openings (1.0 = open, 0.0 = closed):
        ``start_eef_z``, ``min_eef_z`` (the lowest point reached),
        ``peak_eef_z`` (the highest point after that), ``peak_lift_m`` (peak -
        min), ``start_gripper``, ``final_gripper``.  The policy drives BOTH
        arms and picks the arm itself; inspect both.
    monitored_arm: the ``arm`` argument, echoed.
    budget_exhausted: the native-action budget ran out during the pick.
    truncated: same condition, as recorded in the state after the call.
    """

    success: bool
    instruction: str
    chunks_used: int
    left: dict[str, float]
    right: dict[str, float]
    monitored_arm: str
    budget_exhausted: bool
    truncated: bool


@dataclass(frozen=True, repr=False)
class Segment(PerceptionResult):
    """Outcome of ``segment``.

    found: SAM3 produced a mask scoring at least ``min_score``.
    centroid: (row, col) of the mask centroid (mean of the mask pixels, floored),
        or None.  Pass exactly this pixel, with the same camera, to
        ``back_project``.
    area_px: number of mask pixels (0 when not found).
    score: SAM3 confidence in [0, 1], None when not found.
    box: mask bounding box as reported by SAM3, or None.
    provenance: observation, image and calibration sources; backends without
        snapshot support report unverified provenance.  Reuse its
        ``observation_id`` when back-projecting
        ``centroid``; check ``robo.perception_is_current(result)`` before
        using the result for a new action.
    """

    found: bool
    centroid: tuple[int, int] | None
    area_px: int
    score: float | None
    box: Any
    camera: str
    provenance: dict[str, Any] | None = field(default=None, kw_only=True)


@dataclass(frozen=True, repr=False)
class WorldPoint(PerceptionResult):
    """Outcome of ``back_project``: the surface point seen at one pixel.

    xyz: world [x, y, z], metres.  It is a visible SURFACE point (the top of
        an object as the camera sees it), not the object's centre.
    depth_m: distance of that point along the camera's optical axis.
    provenance: observation, image and calibration sources.  A historical
        result keeps its original coordinates; ``robo.require_current(result)``
        rejects it after the scene or observation changes.
    """

    pixel: tuple[int, int]
    xyz: tuple[float, float, float]
    depth_m: float
    camera: str
    provenance: dict[str, Any] | None = field(default=None, kw_only=True)


@dataclass(frozen=True, repr=False)
class ObjectPose(PerceptionResult):
    """Outcome of ``locate``: an object measured from its mask and the depth image.

    found: SAM3 produced a mask scoring at least ``min_score``.
    center_xyz: [x, y, z] of the middle of the smallest rectangle around the
        object's footprint, at half its height above the table, metres.  A
        handle, a spout or a tail is part of the footprint and moves the
        middle away from the body's.  None when no 3D measurement was
        possible (see ``note``).
    top_z: height of the object's top surface (the highest points seen).
    height_m: ``top_z`` above the table surface (0.765 m).
    length_m / width_m: the long and the short side of the smallest rectangle
        around the footprint.
    long_axis_deg: direction of the long side on the table, degrees from +x
        (-90 .. 90); of a square, one of its sides.
    grasp_yaw: the ``yaw`` for ``move_to(..., pitch=90, yaw=grasp_yaw)``:
        pointing straight down, the fingers then close across the short side
        (a cube's: across two faces).
    score: SAM3 confidence.
    centroid: (row, col) of the mask in the image, as ``Segment.centroid``.
    area_px: mask pixels.
    camera: the camera measured from.
    note: why there is no 3D measurement (a flat object, no depth), else None.
    provenance: observation, image and calibration sources.  Inspect historical
        measurements freely; use ``robo.require_current(result)`` before
        depending on their coordinates for a new action.

    The camera sees the top and the near side.  The far side of an object that
    is round about an upright axis (a bottle, a cup) is filled in from the arc
    that is seen; the far side of anything else, and what other things or the
    robot's own gripper hide, is not measured, so such a footprint can be
    shorter than the object.  A held object is usually not found from the head
    camera.
    """

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


STATEFUL_TOOLS = frozenset({"move_pose", "grip", "home", "pi0_pick", "motion_execute"})


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
        schema="pyrualean.calibration.v1",
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


class RobodojoRobot(CoordinationMixin, RobotBase):
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
        Segment,
        WorldPoint,
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
        "truncated",
        "step",
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
        """Read the recorded state of both arms.

        Args:
            step: recorded step to read; ``-1`` is the latest, ``0`` the
                initial scene.  A new step is recorded after every motion call.
        Returns:
            State with ``left`` / ``right`` end-effector poses, measured
            gripper openings and joint angles, and the ``truncated`` flag.
        """
        return super().state(step)

    def image(self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1) -> Any:
        """Load a recorded camera image as a ``numpy`` uint8 array (480, 640, 3), RGB.

        Args:
            camera: ``"cam_head"`` (fixed view of the whole table: identify
                objects, destinations and the global layout),
                ``"cam_left_wrist"`` or ``"cam_right_wrist"`` (on the gripper:
                what that arm is about to grasp or is holding).
            resolution: there is one resolution per camera, ``"high"``.  Pixel
                coordinates are (row, col) with row 0 at the top and col 0 at
                the left.
            step: recorded step, ``-1`` = latest.  For pixels from a snapshot,
                use its ``step`` here and its ``observation_id`` with
                ``segment`` / ``back_project`` so both read the same capture.
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
        """Return the calibration of one camera at a recorded step as a dict.

        Keys: ``shape`` (image shape), ``intrinsic_matrix`` (3x3),
        ``extrinsic_matrix`` (4x4 camera-to-world; the camera looks down its
        own -z axis with +y up) and ``has_depth``.  The wrist cameras move
        with their arm.  ``observation_id`` pins the same calibration as a
        snapshot's image and perception result; it can name a historical
        capture.  With both ``step`` and ``observation_id``, they must agree.
        ``provenance`` records the capture and calibration source when the
        backend supports snapshots; it does not assert that a capture is current.
        """
        camera = _camera(camera, self.CAMERAS)
        snapshot, step = self._camera_snapshot(step, observation_id)
        payload = self._readonly(self.STATE_TOOL, {"step": step})
        cameras = (payload.get("state") or {}).get("cameras") or {}
        result = deepcopy(cameras.get(camera) or {})
        if snapshot is not None:
            result["provenance"] = _calibration_provenance(snapshot, camera)
        return result

    def show(self, camera: str = "cam_head") -> None:
        """Attach the current image of ``camera`` to the feedback of this turn.

        For normal on-demand feedback, call ``show("cam_head")``,
        ``show("cam_left_wrist")`` or ``show("cam_right_wrist")`` when *you*
        (the author) need to look at the scene before writing the next turn.
        Costs tokens: look only when a decision depends on it.
        """
        super().show(camera)

    def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Return the raw bytes of a recorded artifact (advanced).

        Names are ``cam_head.png``, ``cam_left_wrist.png`` and
        ``cam_right_wrist.png``.
        """
        return super().artifact(name, step=step)

    # -- perception ---------------------------------------------------------

    def segment(
        self,
        text_prompt: str,
        camera: str = "cam_head",
        min_score: float = 0.2,
        *,
        observation_id: str | None = None,
    ) -> Segment:
        """Segment an object in one observation of ``camera`` by a text prompt (SAM3).

        Args:
            text_prompt: what to find, e.g. ``"the green bottle"``.  Name one
                object by its visible properties.
            camera: ``"cam_head"`` (default), ``"cam_left_wrist"`` or
                ``"cam_right_wrist"``.
            min_score: masks scoring below this are reported as not found.
            observation_id: snapshot to measure; None pins the latest recorded
                observation.  An explicit historical snapshot uses its saved
                image and returns a historical result; it never reads a newer
                image instead.
        Returns:
            Segment; when ``found``, pass ``centroid`` (row, col) to
            ``back_project`` with the same camera and the result's
            ``provenance["observation_id"]`` to get a world point.  The
            mask is the best match for the text, not a guarantee of identity:
            check where it is against the image.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
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
        """Return the world point at one pixel, using one snapshot's image and calibration.

        Args:
            row / col: pixel, row 0 at the top, col 0 at the left (images are
                480 rows x 640 cols).
            camera: the camera the pixel was read from (default ``"cam_head"``).
            observation_id: the snapshot from which the pixel was read.  None
                pins the latest recorded observation.  Pass the source ID
                explicitly for an older image or ``Segment.centroid``.
        Returns:
            WorldPoint; ``xyz`` is where the camera ray meets the first
            surface.  A pixel on an object's edge or a thin part can land on
            the table or the background behind it: prefer ``Segment.centroid``.
            ``provenance`` records the source; use ``robo.require_current(result)``
            before depending on its coordinates for a new action.
        Raises:
            ToolError: pixel out of range, no valid depth there, or the
            camera calibration is missing.
        """
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
        """Find an object by a text prompt and measure it in 3D.

        SAM3 segments one snapshot's image of ``camera`` (as ``segment``); every
        pixel of the mask is then projected into the world through the depth
        image, and the points above the table give the object's centre, top,
        footprint and heading.  Nothing moves.

        Args:
            text_prompt: one object by its visible properties, e.g. ``"the
                small red cube on the left"``.
            camera: ``"cam_head"`` (default; the whole table),
                ``"cam_left_wrist"`` or ``"cam_right_wrist"`` (close up, from
                the gripper: re-measure before a precise grasp).
            min_score: masks scoring below this are reported as not found.
            observation_id: snapshot to measure; None pins the latest recorded
                observation.  Historical IDs use the recorded images, depth
                and calibration from that capture.
        Returns:
            ObjectPose.  For a top grasp: ``move_to([x, y, top_z + 0.08],
            pitch=90, yaw=grasp_yaw)`` above it, then down to the height to
            hold it at.  The mask is the best match for the text, not proof of
            identity: check it against the image.
            ``provenance`` links the measurement to its capture.  Use
            ``robo.require_current(result)`` before depending on coordinates;
            an old result does not become current when read again.
        Raises:
            ToolError: the camera image or the SAM3 service is unavailable.
        """
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
        """Run the frozen Pi0.5 policy closed loop until an arm has lifted something.

        The policy was trained on complete demonstrations of this benchmark's
        tasks, each under its task instruction: it is a whole-task policy, not
        a grasp skill (the name is RPent's).  It receives the three current
        camera images, both arms' joint state and an instruction, and drives
        BOTH arms for up to ``max_chunks`` chunks of up to 50 native actions
        each.  The call returns as soon as its grasp heuristic fires (an arm
        went down, came back up at least 0.04 m and has its gripper closed) or
        the budget runs out; called again, the policy carries on from the
        scene as it then is.

        Args:
            prompt: the instruction for the policy.  None (default) uses the
                benchmark's own task instruction (``robo.task``).  An override
                must name the intended object by what distinguishes it (not
                just "pick the bottle") and must not contain template markers
                such as ``<target>``.
            arm: recorded as ``monitored_arm`` only; the policy chooses the arm
                itself and the heuristic watches both.
            max_chunks: upper bound on policy chunks (>= 1).  Each chunk can
                spend 50 native actions of the episode's budget, so the
                default can spend 400.
        Returns:
            Pick; ``success`` is a provisional heuristic.  Look at the wrist
            camera of the arm that lifted (``Pick.left`` / ``Pick.right``) to
            confirm that the intended object is held.  A broad task
            instruction may fit several objects and promises no grasp order.
        Raises:
            ToolError: the policy service is unavailable or the prompt was
            rejected.
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

    def set_gripper(self, arm: str, gripper: int, opening: float | None = None) -> Gripper:
        """Close or open one gripper, without moving the arms.

        The command is repeated until the fingers stop (2 to 10 native
        actions).  After a close, ``holding`` tells whether something stopped
        the fingers apart and ``width_m`` how wide it is.  Later motions with
        ``robo.KEEP`` keep commanding what was commanded last, so a held
        object stays squeezed and a partial opening stays as it is.

        Args:
            arm: ``"left"`` or ``"right"``.
            gripper: ``robo.CLOSE`` (1) or ``robo.OPEN`` (-1).
            opening: with ``robo.OPEN`` only: open to this reading
                (0 < opening <= 1) instead of all the way, e.g. to reach
                between close neighbours.  Something 0.002 + 0.109 x opening
                metres wide then just fits between the fingers.
        Returns:
            Gripper.  ``holding`` is not proof that the INTENDED object is
            held: confirm it moves with the arm (wrist camera, or ``locate``
            it again after a lift).
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
        """Return an arm to the joint angles it had when the episode started.

        Near a recorded release, first withdraw along the gripper's negative
        pointing axis until 6 cm behind the release plane, keeping orientation.
        A partial withdrawal resumes from the measured pose. If this line is
        blocked or unreachable, return incomplete without swinging. This local
        guard does not detect objects along the rest of the home path.
        A grasp point lower than 1.0 m then goes straight up by up to 6 cm;
        after releasing a held object, it then follows a Cartesian line above
        the starting TCP at that cleared height before the joint return. This
        protection persists after finger retreat until homed or a new close command.
        Ordinary home keeps the original lift/joint route. An unreachable or
        incomplete staging line stops the call.
        Clear tall things yourself: there is no scene collision checking.
        The call returns when every joint is
        within 0.01 rad or after 25 actions of the swing. The gripper retains
        its last command; an arm that is not named holds its joints. Most
        tasks count as done only with both arms back where they started.

        Args:
            arm: ``"left"``, ``"right"`` or ``"both"`` (default).
        Returns:
            Home; test ``reached`` and inspect an incomplete return before
            deciding how to clear the path or retry. Between nearby operations,
            continue from a clear local pose; use home when clearance or the
            task's final reset requires it.
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
    host="pyrualean.hosts.rpent_robodojo",
    knowledge="robodojo",
    blurb="a dual-arm ARX-X5 robot in the RoboDojo benchmark",
    guides_subdir="",
    guide_names=(),
    show_examples=(
        "`robo.show('cam_head')`, `robo.show('cam_left_wrist')` or `robo.show('cam_right_wrist')`"
    ),
    services="a frozen Pi0.5 VLA and a SAM3 segmentation service",
    services_novla="a SAM3 segmentation service",
)


__all__ = [
    "ADAPTER",
    "ArmState",
    "Gripper",
    "HOME_MAX_ACTIONS",
    "HOME_TOL_RAD",
    "Home",
    "MOVE_TOL_DEG",
    "MOVE_TOL_M",
    "Move",
    "ObjectPose",
    "Pick",
    "RobodojoRobot",
    "STATEFUL_TOOLS",
    "Segment",
    "State",
    "TCP_OFFSET",
    "VLA_CHUNK_ACTIONS",
    "WorldPoint",
    "gripper_angles",
]
