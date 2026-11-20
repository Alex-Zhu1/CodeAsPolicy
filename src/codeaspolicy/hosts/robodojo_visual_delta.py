# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Guarded relative nudges and the closed visual-delta loop, for RoboDojo's host.

``nudge`` moves one arm (or both) by a small bounded relative delta of the grasp point,
in the world frame or the grasp (TCP) frame.  In the default compact mode the Cartesian
line is solved in 5 mm / 4 degree waypoints, split so no joint changes more than
``JOINT_STEP_RAD`` per native action, and each waypoint is ONE native action: no
rest-to-rest spline, no ramp, at most one settle hold.  A 1 cm press costs 2-3 actions
instead of a timed spline's dozens.  Every waypoint is solved and checked against the
joint limits and the table before the first action (a refused delta costs 0 actions).

Each action's measured progress is compared with the commanded progress.  With
``stop_on_stall`` a run of ``stall_steps`` actions that do not bring the grasp point
closer (against contact, a seated part, a hard stop) ends the nudge cleanly with
``status="stopped"``, ``stalled=True`` and ``stop_reason="contact_or_stall"``.  This is
a kinematic progress test, not a force sensor.

``visual_align_and_insert`` runs :func:`codeaspolicy.visual_delta.run_visual_delta_loop`
with views and proprioception from the live observation and ``nudge`` as executor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np

from .. import visual_delta as vd
from . import robodojo_kinematics as kin
from . import robodojo_tools as motion

ARMS = kin.ARMS
FRAMES = ("world", "tcp")
#: Default per-nudge limits and the hard ceiling an argument may raise them to.
MAX_LINEAR_M = 0.06
MAX_ANGULAR_DEG = 20.0
HARD_LINEAR_M = 0.12
HARD_ANGULAR_DEG = 45.0
#: Cartesian waypoint spacing of a compact nudge.
LINE_STEP_M = 0.005
LINE_STEP_DEG = 4.0
#: Finer waypoint spacing for tabletop-contact descent and gentle thin-part liftoff.
FINE_STEP_M = 0.003
#: Measured opening threshold below which held-part motion uses gentle 3 mm steps.
THIN_HELD_OPENING = 0.045
#: Maximum controlled fingertip overdrive below TABLE_Z when allow_table_contact=True.
TABLE_TOUCH_OVERDRIVE_M = 0.004
#: Pre-open tolerance in ``delta_grasp`` (skip pre-open in 0 steps when already within this).
PREOPEN_TOL = 0.05
#: A nudge has arrived within these of its target.
TOL_M = 0.002
TOL_DEG = 2.0
#: An action progresses when the grasp point closes at least this fraction of the
#: commanded progress; commanded progress below the minimums is not judged.
STALL_RATIO = 0.25
MIN_EXPECTED_M = 0.001
MIN_EXPECTED_DEG = 0.5
SETTLE_STEPS = 1
#: The closed loop's default per-round limits (finer than a free nudge).
ALIGN_MAX_LINEAR_M = 0.03
ALIGN_MAX_ANGULAR_DEG = 15.0
CROP_FRACTION = 0.5
#: Default transverse tolerance and sub-millimetre skip threshold for ``align_feature``.
ALIGN_TOL_M = 0.0025
ZERO_DELTA_M = 0.0005


def _rx(deg: float) -> np.ndarray:
    return kin._rot_x(math.radians(deg))


def _ry(deg: float) -> np.ndarray:
    return kin._rot_y(math.radians(deg))


def _rz(deg: float) -> np.ndarray:
    return kin._rot_z(math.radians(deg))


def _zyx_deg(rot: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees of ``rot = Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(rot[2, 0])))))
    yaw = math.degrees(math.atan2(float(rot[1, 0]), float(rot[0, 0])))
    roll = math.degrees(math.atan2(float(rot[2, 1]), float(rot[2, 2])))
    return yaw, pitch, roll


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a number")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return out


def _spec(raw: Any, where: str) -> dict[str, Any]:
    """A per-arm delta: dxyz (metres), dpitch / dyaw / droll (degrees), gripper."""

    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a dict with dxyz / dpitch / dyaw / droll / gripper")
    extra = sorted(set(raw) - {"dxyz", "dpitch", "dyaw", "droll", "gripper"})
    if extra:
        raise ValueError(f"{where} has unknown keys {extra}")
    dxyz = raw.get("dxyz", (0.0, 0.0, 0.0))
    try:
        dxyz = [_finite(v, f"{where}.dxyz") for v in dxyz]
    except TypeError as exc:
        raise ValueError(f"{where}.dxyz must be three numbers") from exc
    if len(dxyz) != 3:
        raise ValueError(f"{where}.dxyz must be three numbers")
    gripper = raw.get("gripper", 0)
    gripper = 0 if gripper is None else _finite(gripper, f"{where}.gripper")
    if gripper not in (-1.0, 0.0, 1.0):
        raise ValueError(f"{where}.gripper must be 1 (close), -1 (open) or 0 (keep)")
    return {"dxyz": dxyz, "dpitch": _finite(raw.get("dpitch", 0.0), f"{where}.dpitch"),
            "dyaw": _finite(raw.get("dyaw", 0.0), f"{where}.dyaw"),
            "droll": _finite(raw.get("droll", 0.0), f"{where}.droll"),
            "gripper": int(gripper)}


def _moves(spec: dict[str, Any]) -> bool:
    return (float(np.linalg.norm(spec["dxyz"])) > 1e-6
            or any(abs(spec[k]) > 1e-6 for k in ("dpitch", "dyaw", "droll")))


def target_pose(arm: str, q: Sequence[float], spec: dict[str, Any], frame: str) -> np.ndarray:
    """The grasp-point pose (4x4) a delta asks for, from joints ``q``.

    world: position + dxyz; angles = current gripper angles + deltas (degrees,
    ``robodojo_kinematics``), the rotation kept exactly when no angle changes.
    tcp: position + R @ dxyz; rotation R @ Rz(dyaw) @ Ry(dpitch) @ Rx(droll).
    """

    pose = motion._pose_of(arm, q)
    rot, pos = pose[:3, :3], pose[:3, 3]
    d = np.asarray(spec["dxyz"], float)
    out = np.eye(4)
    if frame == "world":
        out[:3, 3] = pos + d
        if any(abs(spec[k]) > 1e-9 for k in ("dpitch", "dyaw", "droll")):
            pitch, yaw, roll = kin.angles(rot)
            out[:3, :3] = kin.orientation(pitch + spec["dpitch"], yaw + spec["dyaw"],
                                          roll + spec["droll"])
        else:
            out[:3, :3] = rot
    else:
        out[:3, 3] = pos + rot @ d
        out[:3, :3] = rot @ _rz(spec["dyaw"]) @ _ry(spec["dpitch"]) @ _rx(spec["droll"])
    return out


def achieved_delta(start: np.ndarray, end: np.ndarray) -> dict[str, dict[str, Any]]:
    """The measured motion between two grasp-point poses, in both frames.

    world: dxyz metres and gripper-angle differences (degrees, wrapped; yaw/roll
    folded into one heading change near vertical, ``visual_delta.angle_delta``);
    tcp: dxyz in the start grasp frame and the relative rotation as
    ``Rz(dyaw) Ry(dpitch) Rx(droll)`` about that frame's axes.
    """

    d = end[:3, 3] - start[:3, 3]
    dpitch, dyaw, droll = vd.angle_delta(kin.angles(start[:3, :3]), kin.angles(end[:3, :3]))
    world = {"dxyz": [round(float(v), 4) for v in d], "dpitch": round(dpitch, 2),
             "dyaw": round(dyaw, 2), "droll": round(droll, 2)}
    yaw, pitch, roll = _zyx_deg(start[:3, :3].T @ end[:3, :3])
    local = start[:3, :3].T @ d
    tcp = {"dxyz": [round(float(v), 4) for v in local], "dpitch": round(pitch, 2),
           "dyaw": round(yaw, 2), "droll": round(roll, 2)}
    return {"world": world, "tcp": tcp}


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def _within_limits(q: Sequence[float]) -> bool:
    return all(lo - 1e-6 <= float(v) <= hi + 1e-6
               for v, (lo, hi) in zip(q, kin.JOINT_LIMITS, strict=True))


def _is_thin_held(primitives: Any, state: dict[str, Any], arm: str) -> bool:
    """Whether ``arm`` is holding a thin part (0 < opening <= THIN_HELD_OPENING)."""

    opening = motion._opening(state, arm)
    last = (getattr(primitives, "_grip_commands", None) or {}).get(arm)
    closed = last is None or last <= 0.05
    return bool(closed and 0.0 < opening <= THIN_HELD_OPENING)


def _other_arm(arm: str) -> str:
    return "left" if arm == "right" else "right"


def _safe_joints(state: Mapping[str, Any] | None, arm: str) -> list[float] | None:
    if not isinstance(state, Mapping):
        return None
    try:
        q = motion._joints(state, arm)
        return q if len(q) >= 6 else None
    except Exception:  # noqa: BLE001 - optional other-arm state
        return None


def _safe_opening(state: Mapping[str, Any] | None, arm: str, default: float = 0.5) -> float:
    if not isinstance(state, Mapping):
        return default
    try:
        return float(motion._opening(state, arm))
    except Exception:  # noqa: BLE001 - optional opening reading
        return default


def _check_inter_arm_pair(
    left_q: Sequence[float] | None,
    right_q: Sequence[float] | None,
    *,
    left_opening: float = 0.5,
    right_opening: float = 0.5,
) -> str | None:
    """Loose-coupled inter-arm collision check via ``kin.inter_arm_violations`` when available."""

    inter_fn = getattr(kin, "inter_arm_violations", None)
    if not callable(inter_fn) or left_q is None or right_q is None:
        return None
    bad = inter_fn(
        left_q,
        right_q,
        left_opening=float(left_opening),
        right_opening=float(right_opening),
    )
    if bad:
        labels = ", ".join(str(v) for v in bad)
        return f"the line causes an inter-arm collision ({labels})"
    return None


def _check_inter_arm_trajectory(
    start_left: Sequence[float] | None,
    start_right: Sequence[float] | None,
    pairs: Sequence[tuple[Sequence[float] | None, Sequence[float] | None]],
    *,
    left_opening: float = 0.5,
    right_opening: float = 0.5,
) -> str | None:
    """Inter-arm clearance check over a planned trajectory, allowing separating retreats.

    Matches :meth:`robodojo_intent.IntentController._check_inter_arm_plan`: when the
    starting pose is clear, any waypoint violation rejects the trajectory. When the
    starting pose already violates ``min_clearance_m``, the trajectory is allowed iff
    no waypoint worsens clearance (``dist >= start_dist - 0.001``) and the final
    waypoint either clears all violations or strictly increases clearance
    (``final_dist > start_dist + 0.002``).
    """

    inter_fn = getattr(kin, "inter_arm_violations", None)
    if not callable(inter_fn) or not pairs:
        return None
    lo, ro = float(left_opening), float(right_opening)
    start_violations: list[str] = []
    if start_left is not None and start_right is not None:
        start_violations = list(
            inter_fn(start_left, start_right, left_opening=lo, right_opening=ro)
        )
    min_dist_fn = getattr(
        kin, "inter_arm_min_distance", getattr(kin, "_inter_arm_min_distance", None)
    )
    start_dist: float | None = None
    if (
        start_violations
        and callable(min_dist_fn)
        and start_left is not None
        and start_right is not None
    ):
        start_dist = float(
            min_dist_fn(start_left, start_right, left_opening=lo, right_opening=ro)
        )
        final_left, final_right = pairs[-1]
        if final_left is not None and final_right is not None:
            final_violations = list(
                inter_fn(final_left, final_right, left_opening=lo, right_opening=ro)
            )
            if final_violations:
                final_dist = float(
                    min_dist_fn(final_left, final_right, left_opening=lo, right_opening=ro)
                )
                if final_dist <= start_dist + 0.002:
                    labels = ", ".join(str(v) for v in final_violations)
                    return f"the line causes an inter-arm collision ({labels})"

    for lq, rq in pairs:
        if lq is None or rq is None:
            continue
        bad = list(inter_fn(lq, rq, left_opening=lo, right_opening=ro))
        if not bad:
            continue
        if start_dist is None:
            labels = ", ".join(str(v) for v in bad)
            return f"the line causes an inter-arm collision ({labels})"
        dist = float(min_dist_fn(lq, rq, left_opening=lo, right_opening=ro))
        if dist < start_dist - 0.001:
            labels = ", ".join(str(v) for v in bad)
            return f"the line causes an inter-arm collision ({labels})"
    return None


def _plan_arm(
    arm: str,
    q: Sequence[float],
    spec: dict[str, Any],
    frame: str,
    step_m: float,
    step_deg: float,
    *,
    allow_table_contact: bool = False,
    other_q: Sequence[float] | None = None,
    opening: float = 0.5,
    other_opening: float = 0.5,
) -> tuple[np.ndarray, list, str | None]:
    """(target pose, one joint target per native action, why it cannot be done)."""

    target = target_pose(arm, q, spec, frame)
    if not _moves(spec):
        return target, [], None
    min_tcp_z = (
        kin.TABLE_Z + (kin.TCP_OFFSET - kin.TIP_OFFSET) - TABLE_TOUCH_OVERDRIVE_M
        if allow_table_contact
        else kin.TABLE_Z + 0.002
    )
    if target[2, 3] < min_tcp_z:
        return target, [], "the target grasp point is below the table top"
    ways, why = kin.line_path(
        arm,
        q,
        target,
        step_m=step_m,
        step_deg=step_deg,
        allow_table_touch=allow_table_contact,
    )
    if why is not None:
        return target, [], why
    steps = motion._followable(q, ways)
    for way in steps:
        if not _within_limits(way):
            return target, [], "the line needs a joint beyond its limits"
        bad = kin.clearance_violations(arm, way, allow_table_touch=allow_table_contact)
        if bad:
            return target, [], f"the line takes the arm into the table ({', '.join(bad)})"
    if other_q is not None and steps:
        start_left, start_right = (q, other_q) if arm == "left" else (other_q, q)
        lo, ro = (opening, other_opening) if arm == "left" else (other_opening, opening)
        pairs = [(way, other_q) if arm == "left" else (other_q, way) for way in steps]
        inter_why = _check_inter_arm_trajectory(
            start_left, start_right, pairs, left_opening=lo, right_opening=ro
        )
        if inter_why is not None:
            return target, [], inter_why
    return target, steps, None


def _gripper_state(primitives: Any, state: dict[str, Any], arm: str) -> dict[str, Any]:
    opening = motion._opening(state, arm)
    last = (getattr(primitives, "_grip_commands", None) or {}).get(arm)
    closed = last is not None and last <= 0.05
    holding = bool(closed and opening > motion.HOLD_MIN_OPENING)
    return {
        "opening": round(opening, 3),
        "gripper_commanded": None if last is None else ("closed" if closed else "open"),
        "holding": holding,
        "width_m": round(motion.GRIP_ZERO_M + motion.GRIP_SPAN_M * opening, 4)
        if holding else None,
    }


def _budget(primitives: Any) -> tuple[dict[str, Any], int]:
    getter = getattr(primitives.env, "get_status", None)
    status = getter() if callable(getter) else {}
    status = status if isinstance(status, dict) else {}
    try:
        remaining = int(status.get("step_limit", 10000)) - int(status.get("step", 0))
    except (TypeError, ValueError):
        remaining = 10000
    return status, remaining


def _progress(before: np.ndarray, after: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """(metres, degrees) by which ``after`` is closer to ``target`` than ``before``."""

    d0 = float(np.linalg.norm(target[:3, 3] - before[:3, 3]))
    d1 = float(np.linalg.norm(target[:3, 3] - after[:3, 3]))
    r0 = kin.rotation_error(before[:3, :3], target[:3, :3])
    r1 = kin.rotation_error(after[:3, :3], target[:3, :3])
    return d0 - d1, r0 - r1


def _stalled(measured: tuple[float, float], expected: tuple[float, float]) -> bool | None:
    """Whether one action failed to progress; None when too little was commanded."""

    judged = []
    if expected[0] >= MIN_EXPECTED_M:
        judged.append(measured[0] < STALL_RATIO * expected[0])
    if expected[1] >= MIN_EXPECTED_DEG:
        judged.append(measured[1] < STALL_RATIO * expected[1])
    if not judged:
        return None
    return all(judged)


# ---------------------------------------------------------------------------
# nudge
# ---------------------------------------------------------------------------


def _arguments(arm, dxyz, dpitch, dyaw, droll, gripper, left, right, frame, stall_steps,
               max_linear_m, max_angular_deg, max_steps, reserve_steps, step_m,
               settle_steps, tol_m, tol_deg) -> dict[str, dict[str, Any]]:
    if arm not in (*ARMS, "both"):
        raise ValueError("arm must be 'left', 'right' or 'both'")
    if frame not in FRAMES:
        raise ValueError("frame must be 'world' or 'tcp'")
    if arm == "both":
        if left is None and right is None:
            raise ValueError("arm='both' takes left={...} and/or right={...} deltas")
        specs = {name: _spec(raw, name) for name, raw in (("left", left), ("right", right))
                 if raw is not None}
    else:
        if left is not None or right is not None:
            raise ValueError("left= / right= deltas go with arm='both'")
        specs = {arm: _spec(dict(dxyz=dxyz, dpitch=dpitch, dyaw=dyaw, droll=droll,
                                 gripper=gripper), arm)}
    if type(stall_steps) is not int or not 1 <= stall_steps <= 5:
        raise ValueError("stall_steps must be an integer between 1 and 5")
    if not 0 < _finite(max_linear_m, "max_linear_m") <= HARD_LINEAR_M:
        raise ValueError(f"max_linear_m must be in (0, {HARD_LINEAR_M}]")
    if not 0 < _finite(max_angular_deg, "max_angular_deg") <= HARD_ANGULAR_DEG:
        raise ValueError(f"max_angular_deg must be in (0, {HARD_ANGULAR_DEG}]")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("max_steps must be a positive integer or None")
    if type(reserve_steps) is not int or reserve_steps < 0:
        raise ValueError("reserve_steps must be a non-negative integer")
    if not 0.002 <= _finite(step_m, "step_m") <= 0.02:
        raise ValueError("step_m must be between 0.002 and 0.02 m")
    if type(settle_steps) is not int or not 0 <= settle_steps <= 3:
        raise ValueError("settle_steps must be an integer between 0 and 3")
    if not 0.0005 <= _finite(tol_m, "tol_m") <= 0.02:
        raise ValueError("tol_m must be between 0.0005 and 0.02 m")
    if not 0.1 <= _finite(tol_deg, "tol_deg") <= 10.0:
        raise ValueError("tol_deg must be between 0.1 and 10 degrees")
    return specs


def _arm_report(primitives, state, arm, spec, start, target, frame) -> dict[str, Any]:
    q = motion._joints(state, arm)
    now = motion._pose_of(arm, q)
    summary = motion._pose_summary(arm, q)
    report = {
        "start_xyz": [round(float(v), 4) for v in start[:3, 3]],
        "target_xyz": [round(float(v), 4) for v in target[:3, 3]],
        "final_xyz": summary["xyz"],
        "final_angles": {k: summary[k] for k in ("pitch", "yaw", "roll")},
        "dist_to_target_m": round(float(np.linalg.norm(target[:3, 3] - now[:3, 3])), 4),
        "orientation_error_deg": round(kin.rotation_error(now[:3, :3], target[:3, :3]), 2),
        "requested_delta": {k: spec[k] for k in ("dxyz", "dpitch", "dyaw", "droll")},
        "gripper_command": spec["gripper"],
        "achieved_delta": achieved_delta(start, now),
        "frame": frame,
    }
    report.update(_gripper_state(primitives, state, arm))
    return report


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
    """Copy a single arm's report to the top level (typed results, ledger summary)."""

    arms = result.get("arms") or {}
    if len(arms) == 1:
        (info,) = arms.values()
        for key in ("start_xyz", "target_xyz", "final_xyz", "final_angles",
                    "dist_to_target_m", "orientation_error_deg", "achieved_delta",
                    "opening", "holding", "width_m", "gripper_commanded"):
            if key in info:
                result[key] = info[key]
    return result


def nudge(
    primitives: Any,
    start_joints: dict[str, list[float]],
    arm: str = "right",
    dxyz: Sequence[float] = (0.0, 0.0, 0.0),
    dpitch: float = 0.0,
    dyaw: float = 0.0,
    droll: float = 0.0,
    gripper: float = 0,
    *,
    left: dict | None = None,
    right: dict | None = None,
    frame: str = "world",
    compact: bool = True,
    stop_on_stall: bool = True,
    independent_stall: bool = False,
    stall_steps: int = 2,
    max_linear_m: float = MAX_LINEAR_M,
    max_angular_deg: float = MAX_ANGULAR_DEG,
    max_steps: int | None = None,
    reserve_steps: int = 0,
    step_m: float = LINE_STEP_M,
    settle_steps: int = SETTLE_STEPS,
    tol_m: float = TOL_M,
    tol_deg: float = TOL_DEG,
    allow_table_contact: bool = False,
    _arm_allow_table_contact: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Move one arm (or both) by a bounded relative delta of the grasp point.

    ``dxyz`` metres, ``dpitch`` / ``dyaw`` / ``droll`` degrees, ``gripper`` in
    RoboDojo's command convention (1 close, -1 open, 0 keep).  ``arm="both"`` takes
    ``left`` / ``right`` dicts with the same fields.  With ``allow_table_contact=True``
    only the fingertip floor is relaxed down to ``TABLE_Z - 0.004`` (arm joints stay
    strictly clear of the table) and waypoint spacing tightens to 3 mm so guarded
    downward contact stalls cleanly on the tabletop.  With ``independent_stall=True``
    (and ``stop_on_stall=True``), a stalled arm freezes at its measured joints while
    the other arm finishes its remaining waypoints and settle step.
    """

    try:
        specs = _arguments(arm, dxyz, dpitch, dyaw, droll, gripper, left, right, frame,
                           stall_steps, max_linear_m, max_angular_deg, max_steps,
                           reserve_steps, step_m, settle_steps, tol_m, tol_deg)
    except ValueError as exc:
        return {"error": f"nudge: {exc}"}
    if (not isinstance(compact, bool) or not isinstance(stop_on_stall, bool)
            or not isinstance(independent_stall, bool)
            or not isinstance(allow_table_contact, bool)):
        return {
            "error": "nudge: compact, stop_on_stall, independent_stall and "
                     "allow_table_contact must be booleans"
        }
    per_arm_table = {
        a: bool(_arm_allow_table_contact.get(a, allow_table_contact))
        if _arm_allow_table_contact is not None
        else allow_table_contact
        for a in specs
    }
    any_table_contact = any(per_arm_table.values())
    if any_table_contact and not compact:
        return {"error": "nudge: allow_table_contact=True requires compact=True"}
    state = motion._obs(primitives)["state"]
    q0 = {a: motion._joints(state, a) for a in specs}
    start = {a: motion._pose_of(a, q0[a]) for a in specs}
    result: dict[str, Any] = {
        "arm": arm, "frame": frame, "compact": compact, "status": "rejected",
        "reached": False, "stalled": False, "stop_reason": None, "steps_used": 0,
        "planned_steps": 0, "ik_error": None, "recoverable": True,
    }

    def finish(state_now, targets):
        result["arms"] = {a: _arm_report(primitives, state_now, a, specs[a], start[a],
                                         targets[a], frame) for a in specs}
        result["remaining_episode_steps"] = _budget(primitives)[1]
        return _flatten(result)

    for name, spec in specs.items():
        linear = float(np.linalg.norm(spec["dxyz"]))
        turn = max(abs(spec[k]) for k in ("dpitch", "dyaw", "droll"))
        if linear > max_linear_m + 1e-9 or turn > max_angular_deg + 1e-9:
            result.update(stop_reason="delta_exceeds_limit", note=(
                f"{name}: {linear:.3f} m / {turn:.1f} deg exceeds the per-nudge limit "
                f"{max_linear_m:.3f} m / {max_angular_deg:.1f} deg; split it or move_to"))
            return finish(state, {a: start[a] for a in specs})
    if not compact:
        return _legacy(primitives, start_joints, arm, specs, frame, result, finish, state)
    thin_held = any(_is_thin_held(primitives, state, a) and _moves(specs[a]) for a in specs)
    has_linear = any(float(np.linalg.norm(specs[a]["dxyz"])) > 1e-6 for a in specs)
    effective_step_m = (
        min(step_m, FINE_STEP_M)
        if (any_table_contact or (thin_held and has_linear))
        else step_m
    )
    if any_table_contact:
        step_deg = LINE_STEP_DEG * effective_step_m / LINE_STEP_M
    elif not has_linear:
        step_deg = 6.0
    else:
        step_deg = LINE_STEP_DEG * min(step_m, LINE_STEP_M) / LINE_STEP_M
    q_all = {a: _safe_joints(state, a) for a in ARMS}
    open_all = {a: _safe_opening(state, a) for a in ARMS}
    effective_open = {
        a: max(open_all[a], 1.0) if (a in specs and specs[a]["gripper"] < 0) else open_all[a]
        for a in ARMS
    }
    plans = {
        a: _plan_arm(
            a,
            q0[a],
            specs[a],
            frame,
            effective_step_m,
            step_deg,
            allow_table_contact=per_arm_table[a],
            other_q=None if arm == "both" else q_all[_other_arm(a)],
            opening=effective_open[a],
            other_opening=effective_open[_other_arm(a)],
        )
        for a in specs
    }
    targets = {a: plan[0] for a, plan in plans.items()}
    refused = [f"{a}: {plan[2]}" for a, plan in plans.items() if plan[2]]
    if refused:
        result.update(stop_reason="ik_path_rejected", ik_error="; ".join(refused))
        return finish(state, targets)
    ways = {a: plan[1] for a, plan in plans.items()}
    planned = max(len(w) for w in ways.values())
    if arm == "both" and planned > 0:
        left_ways = ways.get("left") or []
        right_ways = ways.get("right") or []
        pairs = [
            (
                left_ways[min(idx, len(left_ways) - 1)] if left_ways else q_all["left"],
                right_ways[min(idx, len(right_ways) - 1)] if right_ways else q_all["right"],
            )
            for idx in range(planned)
        ]
        inter_why = _check_inter_arm_trajectory(
            q_all["left"],
            q_all["right"],
            pairs,
            left_opening=effective_open["left"],
            right_opening=effective_open["right"],
        )
        if inter_why is not None:
            result.update(stop_reason="ik_path_rejected", ik_error=f"both: {inter_why}")
            return finish(state, targets)
    grips = {a: motion._command(specs[a]["gripper"]) for a in specs}
    result["planned_steps"] = planned
    status, remaining = _budget(primitives)
    allowed = remaining - reserve_steps
    if max_steps is not None:
        allowed = min(allowed, max_steps)
    if planned == 0 and all(g is None for g in grips.values()):
        result.update(status="completed", reached=True, stop_reason=None,
                      note="zero delta: nothing to do")
        out = finish(state, targets)
        for a in specs:
            out["arms"][a]["stalled"] = False
            out["arms"][a]["reached"] = True
        return out
    if allowed <= 0:
        result.update(stop_reason="episode_native_step_budget" if remaining <= 0
                      else "reserve_steps_exhausted")
        return finish(state, targets)
    return _execute(primitives, state, status, specs, start, targets, ways, grips, planned,
                    allowed, stop_on_stall, independent_stall, stall_steps, effective_step_m,
                    settle_steps, tol_m, tol_deg, result, finish)


def _execute(primitives, state, status, specs, start, targets, ways, grips, planned, allowed,
             stop_on_stall, independent_stall, stall_steps, step_m, settle_steps, tol_m, tol_deg,
             result, finish) -> dict[str, Any]:
    """Dispatch planned waypoints one native action each, watching measured progress."""

    moving = [a for a in specs if ways[a]]
    measured = {a: start[a] for a in specs}
    commanded = {a: start[a] for a in specs}
    counters = {a: 0 for a in specs}
    openings = {a: motion._opening(state, a) for a in specs}
    open_change = {a: math.inf for a in specs}
    last_cmd = {a: list(ways[a][-1]) if ways[a] else motion._joints(state, a) for a in specs}
    steps = 0
    stop: str | None = None
    stalled_arm: str | None = None
    stall_seen = False
    indep = bool(independent_stall and stop_on_stall)
    stalled_arms: set[str] = set()
    frozen_q: dict[str, list[float]] = {}

    def act(cmd):
        nonlocal state, status, steps
        primitives._check_cancelled()
        state, status = motion._act(primitives, state, cmd, grips)
        steps += 1
        for a in specs:
            now = motion._opening(state, a)
            open_change[a], openings[a] = abs(now - openings[a]), now

    def judge(arm, expected_pose, expected=None):
        nonlocal stalled_arm, stall_seen
        now = motion._pose_of(arm, motion._joints(state, arm))
        if expected is None:
            expected = _progress(commanded[arm], expected_pose, targets[arm])
        verdict = _stalled(_progress(measured[arm], now, targets[arm]), expected)
        counters[arm] = counters[arm] + 1 if verdict else (0 if verdict is False
                                                             else counters[arm])
        measured[arm] = now
        if expected_pose is not None:
            commanded[arm] = expected_pose
        if counters[arm] >= stall_steps:
            stall_seen = True
            stalled_arm = stalled_arm or arm
            return True
        return False

    def error(arm):
        now = measured[arm]
        return (float(np.linalg.norm(targets[arm][:3, 3] - now[:3, 3])),
                kin.rotation_error(now[:3, :3], targets[arm][:3, :3]))

    for index in range(planned):
        if steps >= allowed:
            stop = "step_budget"
            break
        if motion._spent(status):
            break
        cmd = {}
        for a in specs:
            if a in frozen_q:
                cmd[a] = list(frozen_q[a])
            elif ways[a]:
                cmd[a] = list(ways[a][min(index, len(ways[a]) - 1)])
            elif a not in moving and grips[a] is None:
                cmd[a] = motion._joints(state, a)
            else:
                cmd[a] = last_cmd[a]
        act(cmd)
        if indep:
            for a in moving:
                if a in stalled_arms:
                    measured[a] = motion._pose_of(a, motion._joints(state, a))
                elif index < len(ways[a]):
                    if judge(a, motion._pose_of(a, cmd[a])):
                        stalled_arms.add(a)
                        frozen_q[a] = motion._joints(state, a)
                else:
                    measured[a] = motion._pose_of(a, motion._joints(state, a))
            if moving and all(
                (a in stalled_arms)
                or (
                    index >= len(ways[a]) - 1
                    and error(a)[0] <= tol_m
                    and error(a)[1] <= tol_deg
                )
                for a in moving
            ):
                break
        else:
            hit = [judge(a, motion._pose_of(a, cmd[a])) for a in moving if index < len(ways[a])]
            if any(hit) and stop_on_stall:
                stop = "contact_or_stall"
                break
    # An arm whose own path ended early kept holding: judge its pose as measured now.
    measured.update({a: motion._pose_of(a, motion._joints(state, a)) for a in moving})
    settles = 0
    if indep:
        active_unsettled = [
            a for a in moving
            if a not in stalled_arms and (error(a)[0] > tol_m or error(a)[1] > tol_deg)
        ]
        while (stop is None and active_unsettled and settles < settle_steps
               and not motion._spent(status) and steps < allowed):
            before = {a: error(a) for a in active_unsettled}
            settle_cmd = {
                a: list(frozen_q[a]) if a in frozen_q else last_cmd[a]
                for a in specs
            }
            act(settle_cmd)
            settles += 1
            for a in active_unsettled:
                if judge(a, None, (min(before[a][0], step_m), min(before[a][1], LINE_STEP_DEG))):
                    stalled_arms.add(a)
                    frozen_q[a] = motion._joints(state, a)
            measured.update({a: motion._pose_of(a, motion._joints(state, a)) for a in moving})
            active_unsettled = [
                a for a in moving
                if a not in stalled_arms and (error(a)[0] > tol_m or error(a)[1] > tol_deg)
            ]
    else:
        while (stop is None and moving and settles < settle_steps and not motion._spent(status)
               and steps < allowed
               and any(error(a)[0] > tol_m or error(a)[1] > tol_deg for a in moving)):
            before = {a: error(a) for a in moving}
            act({a: last_cmd[a] for a in specs})
            settles += 1
            hit = [judge(a, None, (min(before[a][0], step_m), min(before[a][1], LINE_STEP_DEG)))
                   for a in moving]
            if any(hit) and stop_on_stall:
                stop = "contact_or_stall"
    if any(g is not None for g in grips.values()):
        # Finish the gripper stroke; after a stall, hold where the arm stopped.
        hold = {
            a: list(frozen_q[a])
            if a in frozen_q
            else (motion._joints(state, a) if stop == "contact_or_stall" else last_cmd[a])
            for a in specs
        }
        for _ in range(motion.GRIP_MAX_ACTIONS):
            if steps >= allowed or motion._spent(status):
                break
            if steps >= 2 and all(open_change[a] < motion.GRIP_STILL
                                  for a in specs if grips[a] is not None):
                break
            act(hold)
    measured.update({a: motion._pose_of(a, motion._joints(state, a)) for a in moving})
    arm_reached = {
        a: (error(a)[0] <= tol_m and error(a)[1] <= tol_deg) if a in moving else True
        for a in specs
    }
    reached = all(arm_reached[a] for a in moving)
    if indep:
        if moving and not reached and stop is None and not stalled_arms and steps >= allowed:
            stop = "step_budget"
        result.update(
            steps_used=steps,
            stalled=bool(stalled_arms),
            recoverable=False,
            settle_steps=settles,
        )
        if reached:
            result.update(status="completed", reached=True, stop_reason=None)
            if stalled_arms:
                result["note"] = "progress stalled near the target, within tolerance"
        elif stalled_arms:
            result.update(
                status="stopped",
                reached=False,
                stop_reason="contact_or_stall",
                stalled_arm=sorted(stalled_arms)[0],
                note="measured progress stopped before the target (contact, a seated "
                     "part or a hard stop); no force is measured",
            )
        elif motion._spent(status):
            result.update(status="partial", stop_reason=motion._episode_stop_reason(status))
        else:
            result.update(status="partial", stop_reason=stop or "final_target_not_reached")
    else:
        if moving and not reached and stop is None and steps >= allowed:
            stop = "step_budget"
        result.update(steps_used=steps, stalled=stall_seen and stop == "contact_or_stall",
                      recoverable=False, settle_steps=settles)
        if stall_seen and not stop_on_stall:
            result["stall_detected"] = True
        if reached:
            result.update(status="completed", reached=True, stop_reason=None)
            if stall_seen:
                result["note"] = "progress stalled near the target, within tolerance"
        elif stop == "contact_or_stall":
            result.update(status="stopped", stop_reason="contact_or_stall", stalled_arm=stalled_arm,
                          note="measured progress stopped before the target (contact, a seated "
                               "part or a hard stop); no force is measured")
        elif motion._spent(status):
            result.update(status="partial", stop_reason=motion._episode_stop_reason(status))
        else:
            result.update(status="partial", stop_reason=stop or "final_target_not_reached")
    out = finish(state, targets)
    for a in specs:
        out["arms"][a]["stalled"] = (
            (a in stalled_arms)
            if indep
            else bool(stall_seen and stop == "contact_or_stall" and a == stalled_arm)
        )
        out["arms"][a]["reached"] = arm_reached[a]
    return out


def _legacy(primitives, start_joints, arm, specs, frame, result, finish, state):
    """compact=False: the same target through ``move_pose`` (timed/spline path)."""

    if arm == "both" or len(specs) != 1:
        return {"error": "nudge: compact=False moves a single arm; use compact=True for both"}
    q = motion._joints(state, arm)
    spec = specs[arm]
    target = target_pose(arm, q, spec, frame)
    pitch, yaw, roll = kin.angles(target[:3, :3])
    angles = {}
    if any(abs(spec[k]) > 1e-9 for k in ("dpitch", "dyaw", "droll")):
        angles = dict(pitch=pitch, yaw=yaw, roll=roll)
    moved = motion.move_pose(primitives, start_joints, [float(v) for v in target[:3, 3]],
                             arm=arm, gripper=spec["gripper"], path="line", **angles)
    if "error" in moved:
        return {"error": f"nudge: {moved['error']}"}
    reached = bool(moved.get("reached"))
    status = moved.get("status", "rejected")
    result.update(status=status if status != "partial" else "partial", reached=reached,
                  steps_used=int(moved.get("steps_used", 0)), stop_reason=moved.get("stop_reason"),
                  ik_error=moved.get("ik_error"), recoverable=bool(moved.get("recoverable")),
                  planned_path=moved.get("planned_path"), trajectory=moved.get("trajectory"))
    state = motion._obs(primitives)["state"]
    return finish(state, {arm: target})


# ---------------------------------------------------------------------------
# the closed visual-delta loop
# ---------------------------------------------------------------------------

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def default_cameras(arm: str) -> tuple[str, ...]:
    return CAMERAS if arm == "both" else ("cam_head", f"cam_{arm}_wrist")


def crop_center(image: Any, fraction: float = CROP_FRACTION) -> np.ndarray:
    """The central ``fraction`` of an image, resized back to the full size (2x zoom)."""

    from PIL import Image

    array = np.asarray(image)[..., :3].astype(np.uint8)
    h, w = array.shape[:2]
    ch, cw = max(1, round(h * fraction)), max(1, round(w * fraction))
    top, left = (h - ch) // 2, (w - cw) // 2
    crop = Image.fromarray(array[top:top + ch, left:left + cw])
    return np.asarray(crop.resize((w, h), Image.BICUBIC))


def camera_images(obs: dict[str, Any], cameras: Sequence[str], crop_wrist: bool
                  ) -> list[dict[str, Any]]:
    """Labelled PNGs of the named views, plus zoomed wrist crops when asked."""

    out = []
    vision = obs.get("vision") or {}
    for name in cameras:
        color = (vision.get(name) or {}).get("color")
        if color is None:
            continue
        out.append({"label": name, "png": vd.encode_png(color)})
        if crop_wrist and name.endswith("_wrist"):
            out.append({"label": f"{name}_crop", "png": vd.encode_png(crop_center(color))})
    return out


def arm_poses(state: dict[str, Any], arms: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for name in arms:
        summary = motion._pose_summary(name, motion._joints(state, name))
        summary["opening"] = motion._opening(state, name)
        out[name] = summary
    return out


has_configured_policy = vd.has_configured_policy


def policy_status(
    primitives: Any = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Metadata describing whether a visual-delta VLM policy is configured."""

    return {
        "protocol": vd.PROTOCOL,
        "visual_delta_policy_configured": has_configured_policy(primitives, env=env),
    }


def resolve_policy(primitives: Any, policy: Any = None, model: str | None = None,
                   base_url: str | None = None) -> Any:
    """An explicit policy, else the host's configured one, else the OpenAI client."""

    if policy is not None:
        return vd.as_policy(policy)
    configured = getattr(primitives, "_visual_delta_policy", None)
    if configured is not None:
        return vd.as_policy(configured)
    return vd.OpenAIVisualDeltaClient(model=model, base_url=base_url)


def set_policy(primitives: Any, policy: Any) -> None:
    """Install a host-side default policy (a server client, a local delta model)."""

    primitives._visual_delta_policy = None if policy is None else vd.as_policy(policy)


def visual_align_and_insert(
    primitives: Any,
    start_joints: dict[str, list[float]],
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
    max_linear_m: float = ALIGN_MAX_LINEAR_M,
    max_angular_deg: float = ALIGN_MAX_ANGULAR_DEG,
    stall_steps: int = 2,
) -> dict[str, Any]:
    """Closed-loop last-mile alignment: views + state -> delta policy -> guarded nudge.

    At most ``max_rounds`` policy turns and ``min(max_steps, remaining episode steps
    - reserve_steps)`` native actions.  Stops early on ``"done": true``, an all-zero
    delta (converged), a contact stall when ``stop_on_contact_complete``, repeated
    invalid answers, a policy error or rounds that cannot move.
    """

    if not isinstance(goal, str) or not goal.strip():
        return {"error": "visual_align_and_insert: goal must be a non-empty string"}
    if arm not in (*ARMS, "both"):
        return {"error": "visual_align_and_insert: arm must be 'left', 'right' or 'both'"}
    if frame not in FRAMES:
        return {"error": "visual_align_and_insert: frame must be 'world' or 'tcp'"}
    for name, value, low, high in (("max_rounds", max_rounds, 1, 32),
                                   ("max_steps", max_steps, 1, 400),
                                   ("reserve_steps", reserve_steps, 0, 10000),
                                   ("stall_steps", stall_steps, 1, 5)):
        if type(value) is not int or not low <= value <= high:
            return {"error": f"visual_align_and_insert: {name} must be an integer in "
                             f"[{low}, {high}]"}
    for name, value in (("stop_on_stall", stop_on_stall), ("crop_wrist", crop_wrist),
                        ("stop_on_contact_complete", stop_on_contact_complete)):
        if not isinstance(value, bool):
            return {"error": f"visual_align_and_insert: {name} must be a boolean"}
    try:
        if not 0 < _finite(max_linear_m, "max_linear_m") <= HARD_LINEAR_M:
            raise ValueError(f"max_linear_m must be in (0, {HARD_LINEAR_M}]")
        if not 0 < _finite(max_angular_deg, "max_angular_deg") <= HARD_ANGULAR_DEG:
            raise ValueError(f"max_angular_deg must be in (0, {HARD_ANGULAR_DEG}]")
    except ValueError as exc:
        return {"error": f"visual_align_and_insert: {exc}"}
    views = tuple(cameras) if cameras is not None else default_cameras(arm)
    unknown = [c for c in views if c not in CAMERAS]
    if unknown:
        return {"error": f"visual_align_and_insert: unknown cameras {unknown}; use {CAMERAS}"}
    arms = ARMS if arm == "both" else (arm,)
    _, remaining = _budget(primitives)
    allowance = min(max_steps, remaining - reserve_steps)
    state = motion._obs(primitives)["state"]
    base: dict[str, Any] = {
        "protocol": vd.PROTOCOL,
        "goal": goal,
        "arm": arm,
        "frame": frame,
        "max_rounds": max_rounds,
        "reserve_steps": reserve_steps,
        "step_allowance": max(0, allowance),
        "cameras": list(views),
        "visual_delta_policy_configured": has_configured_policy(primitives),
    }
    if allowance <= 0:
        base.update(status="rejected", reached=False, done=False, steps_used=0,
                    rounds_used=0, rounds=[], recoverable=True,
                    stop_reason="episode_native_step_budget" if remaining <= 0
                    else "reserve_steps_exhausted",
                    final_state=vd.format_state(arm_poses(state, arms), arm))
        return base
    try:
        chosen = resolve_policy(primitives, policy, model, base_url)
    except (TypeError, ValueError) as exc:
        return {"error": f"visual_align_and_insert: {exc}"}

    def observe(images: bool = True) -> dict[str, Any]:
        obs = motion._obs(primitives)
        return {"poses": arm_poses(obs["state"], arms),
                "images": camera_images(obs, views, crop_wrist) if images else [],
                "remaining_steps": _budget(primitives)[1]}

    def execute(specs: dict[str, dict[str, Any]], steps_left: int) -> dict[str, Any]:
        common = dict(frame=frame, stop_on_stall=stop_on_stall, stall_steps=stall_steps,
                      max_linear_m=max_linear_m, max_angular_deg=max_angular_deg,
                      max_steps=steps_left)
        if arm == "both":
            out = nudge(primitives, start_joints, "both", left=specs.get("left"),
                        right=specs.get("right"), **common)
        else:
            out = nudge(primitives, start_joints, arm, **specs[arm], **common)
        if "error" in out:
            return {"status": "rejected", "stop_reason": "invalid_delta", "steps_used": 0,
                    "ik_error": out["error"]}
        return out

    loop = vd.run_visual_delta_loop(
        goal=goal, arm=arm, frame=frame, policy=chosen, observe=observe, execute=execute,
        step_allowance=allowance, max_rounds=max_rounds, max_linear_m=max_linear_m,
        max_angular_deg=max_angular_deg, stop_on_contact_complete=stop_on_contact_complete,
        model=model or getattr(chosen, "model", None))
    base.update(loop)
    state = motion._obs(primitives)["state"]
    base["arms"] = {a: dict(motion._pose_summary(a, motion._joints(state, a)),
                            **_gripper_state(primitives, state, a)) for a in arms}
    if arm != "both":
        info = base["arms"][arm]
        base.update(final_xyz=info["xyz"],
                    final_angles={k: info[k] for k in ("pitch", "yaw", "roll")},
                    holding=info["holding"], opening=info["opening"])
    base["stalled"] = any((r.get("feedback") or {}).get("stalled") for r in base["rounds"])
    base["recoverable"] = base["steps_used"] == 0
    base["remaining_episode_steps"] = _budget(primitives)[1]
    return base


visual_servo = visual_align_and_insert


# ---------------------------------------------------------------------------
# pre-contact feature-to-target alignment
# ---------------------------------------------------------------------------


def _vec3(raw: Any, where: str) -> np.ndarray:
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{where} must be three numbers")
    try:
        vals = [_finite(v, where) for v in raw]
    except TypeError as exc:
        raise ValueError(f"{where} must be three numbers") from exc
    if len(vals) != 3:
        raise ValueError(f"{where} must be three numbers")
    return np.asarray(vals, dtype=float)


def _wrist_relief_hint(
    arm: str,
    desired_pose: np.ndarray,
    q_seed: Sequence[float],
    *,
    allow_table_contact: bool = False,
    other_q: Sequence[float] | None = None,
    opening: float = 0.5,
    other_opening: float = 0.5,
) -> str | None:
    """Probe small wrist pitch/yaw reliefs at ``desired_pose`` (0 sim steps) for IK hints."""

    pitch0, yaw0, roll0 = kin.angles(desired_pose[:3, :3])
    candidates = (
        *(("dpitch", dp, 0.0) for dp in (8.0, -8.0, 15.0, -15.0)),
        *(("dyaw", 0.0, dy) for dy in (15.0, -15.0)),
    )
    feasible: list[str] = []
    for label, dp, dy in candidates:
        cand = desired_pose.copy()
        cand[:3, :3] = kin.orientation(pitch0 + dp, yaw0 + dy, roll0)
        try:
            sol = kin.best_solution(
                arm,
                cand,
                q_seed,
                allow_table_touch=allow_table_contact,
                other_arm_q=other_q,
                opening=opening,
                other_opening=other_opening,
            )
        except Exception:  # noqa: BLE001 - best-effort diagnostic hint
            continue
        if sol.ok and not sol.collides:
            delta = dp if label == "dpitch" else dy
            feasible.append(f"{label}={delta:+.0f}°")
            if len(feasible) >= 2:
                break
    if not feasible:
        return None
    return (
        f"{arm}: target TCP is reachable with wrist relief ({', '.join(feasible)}); "
        "re-stage hover orientation/height with robo.ik() before calling align_feature."
    )


def _sync_pairs(
    start_l: Sequence[float] | None,
    start_r: Sequence[float] | None,
    ways_l: Sequence[Sequence[float]],
    ways_r: Sequence[Sequence[float]],
) -> list[tuple[Sequence[float] | None, Sequence[float] | None]]:
    n = max(len(ways_l), len(ways_r))
    if n == 0:
        return []
    return [
        (
            ways_l[min(i, len(ways_l) - 1)] if ways_l else start_l,
            ways_r[min(i, len(ways_r) - 1)] if ways_r else start_r,
        )
        for i in range(n)
    ]


def _align_arm_spec(
    raw: Any,
    where: str,
    *,
    default_axis: np.ndarray,
    default_standoff: float | None,
    default_stroke_m: float | None,
    default_stroke_dxyz: Sequence[float] | None,
    default_tol: float,
    default_allow_table: bool,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a dict with feature_xyz/feature and target_xyz/target")
    extra = sorted(
        set(raw)
        - {
            "feature_xyz", "feature", "target_xyz", "target", "axis",
            "standoff_m", "stroke_m", "stroke_dxyz", "align_tol_m", "allow_table_contact",
        }
    )
    if extra:
        raise ValueError(f"{where} has unknown keys {extra}")
    f_raw = raw.get("feature_xyz") if raw.get("feature_xyz") is not None else raw.get("feature")
    t_raw = raw.get("target_xyz") if raw.get("target_xyz") is not None else raw.get("target")
    if f_raw is None or t_raw is None:
        raise ValueError(f"{where} requires feature_xyz/feature and target_xyz/target")
    f_xyz = _vec3(f_raw, f"{where}.feature_xyz")
    t_xyz = _vec3(t_raw, f"{where}.target_xyz")
    if "axis" in raw and raw["axis"] is not None:
        axis_raw = _vec3(raw["axis"], f"{where}.axis")
        axis_norm = float(np.linalg.norm(axis_raw))
        if axis_norm <= 1e-9:
            raise ValueError(f"{where}.axis must be non-zero")
        u_axis = axis_raw / axis_norm
    else:
        u_axis = default_axis.copy()
    standoff_raw = raw["standoff_m"] if "standoff_m" in raw else default_standoff
    standoff_f: float | None = None
    if standoff_raw is not None:
        standoff_f = _finite(standoff_raw, f"{where}.standoff_m")
        if standoff_f < 0.0:
            raise ValueError(f"{where}.standoff_m must be >= 0")
    if "stroke_m" in raw or "stroke_dxyz" in raw:
        sm_raw, sd_raw = raw.get("stroke_m"), raw.get("stroke_dxyz")
    else:
        sm_raw, sd_raw = default_stroke_m, default_stroke_dxyz
    if sm_raw is not None and sd_raw is not None:
        raise ValueError(f"{where}: pass at most one of stroke_m or stroke_dxyz")
    stroke_vec: np.ndarray | None = None
    stroke_unit: np.ndarray | None = None
    requested_stroke_m: float | None = None
    if sm_raw is not None:
        sf = _finite(sm_raw, f"{where}.stroke_m")
        if sf <= 0.0:
            raise ValueError(f"{where}.stroke_m must be > 0")
        stroke_vec, stroke_unit, requested_stroke_m = u_axis * sf, u_axis, sf
    elif sd_raw is not None:
        stroke_vec = _vec3(sd_raw, f"{where}.stroke_dxyz")
        requested_stroke_m = float(np.linalg.norm(stroke_vec))
        if requested_stroke_m <= 1e-6:
            raise ValueError(f"{where}.stroke_dxyz must be non-zero")
        stroke_unit = stroke_vec / requested_stroke_m
    tol_f = _finite(raw.get("align_tol_m", default_tol), f"{where}.align_tol_m")
    if not 0.0005 <= tol_f <= 0.02:
        raise ValueError(f"{where}.align_tol_m must be between 0.0005 and 0.02 m")
    allow_t = raw.get("allow_table_contact", default_allow_table)
    if not isinstance(allow_t, bool):
        raise ValueError(f"{where}.allow_table_contact must be a boolean")
    raw_diff = t_xyz - f_xyz
    axial_gap_m = float(np.dot(raw_diff, u_axis))
    transverse_delta = raw_diff - axial_gap_m * u_axis
    transverse_err_before = float(np.linalg.norm(transverse_delta))
    planned_dxyz = (
        transverse_delta.copy() if standoff_f is None else (t_xyz - u_axis * standoff_f) - f_xyz
    )
    align_norm = float(np.linalg.norm(planned_dxyz))
    return {
        "f_xyz": f_xyz, "t_xyz": t_xyz, "u_axis": u_axis, "standoff_f": standoff_f,
        "stroke_vec": stroke_vec, "stroke_unit": stroke_unit,
        "requested_stroke_m": requested_stroke_m, "tol_f": tol_f,
        "allow_table_contact": allow_t, "axial_gap_m": axial_gap_m,
        "transverse_delta": transverse_delta, "transverse_error_before_m": transverse_err_before,
        "planned_dxyz": planned_dxyz, "align_norm": align_norm,
        "skip_align": align_norm <= ZERO_DELTA_M,
    }


def _align_feature_both(
    primitives: Any,
    start_joints: dict[str, list[float]],
    *,
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    u_axis_default: np.ndarray,
    standoff_f: float | None,
    stroke_m: float | None,
    stroke_dxyz: Sequence[float] | None,
    independent_stall: bool,
    tol_f: float,
    max_lin_f: float,
    max_steps: int,
    reserve_steps: int,
    stop_on_stall: bool,
    stall_steps: int,
    allow_table_contact: bool,
    preview: bool,
) -> dict[str, Any]:
    try:
        if left is None or right is None:
            raise ValueError("arm='both' requires both left={...} and right={...}")
        cfgs = {
            a: _align_arm_spec(
                raw, a, default_axis=u_axis_default, default_standoff=standoff_f,
                default_stroke_m=stroke_m, default_stroke_dxyz=stroke_dxyz,
                default_tol=tol_f, default_allow_table=allow_table_contact,
            )
            for a, raw in (("left", left), ("right", right))
        }
    except ValueError as exc:
        return {"error": f"align_feature: {exc}"}

    state0 = motion._obs(primitives)["state"]
    q0 = {a: motion._joints(state0, a) for a in ARMS}
    tcp_before = {a: motion._pose_of(a, q0[a])[:3, 3] for a in ARMS}
    open0 = {a: _safe_opening(state0, a) for a in ARMS}
    any_stroke = any(cfgs[a]["stroke_vec"] is not None for a in ARMS)
    req_strokes = [
        cfgs[a]["requested_stroke_m"]
        for a in ARMS if cfgs[a]["requested_stroke_m"] is not None
    ]
    arm_rep: dict[str, dict[str, Any]] = {}
    for a in ARMS:
        c = cfgs[a]
        arm_rep[a] = {
            "status": "rejected", "aligned": False, "reached": False, "stalled": False,
            "stop_reason": None,
            "feature_xyz": [round(float(v), 5) for v in c["f_xyz"]],
            "target_xyz": [round(float(v), 5) for v in c["t_xyz"]],
            "axis": [round(float(v), 6) for v in c["u_axis"]],
            "axial_gap_m": round(c["axial_gap_m"], 5),
            "transverse_delta": [round(float(v), 5) for v in c["transverse_delta"]],
            "planned_dxyz": [round(float(v), 5) for v in c["planned_dxyz"]],
            "transverse_error_before_m": round(c["transverse_error_before_m"], 5),
            "transverse_error_after_m": round(c["transverse_error_before_m"], 5),
            "standoff_m": round(c["standoff_f"], 5) if c["standoff_f"] is not None else None,
            "stroke_status": None,
            "requested_stroke_m": round(c["requested_stroke_m"], 5)
            if c["requested_stroke_m"] is not None else None,
            "achieved_stroke_m": None,
            "rim_contact_suspected": False,
            "suggested_backoff_dxyz": None,
        }
    result: dict[str, Any] = {
        "arm": "both", "status": "rejected", "aligned": False, "reached": False,
        "accepted": False, "preview": preview, "stalled": False, "stop_reason": None,
        "feature_xyz": [0.0, 0.0, 0.0], "target_xyz": [0.0, 0.0, 0.0],
        "axis": [round(float(v), 6) for v in u_axis_default],
        "axial_gap_m": round(max(abs(cfgs[a]["axial_gap_m"]) for a in ARMS), 5),
        "transverse_delta": [0.0, 0.0, 0.0], "planned_dxyz": [0.0, 0.0, 0.0],
        "transverse_error_before_m": round(max(cfgs[a]["transverse_error_before_m"] for a in ARMS), 5),
        "transverse_error_after_m": round(max(cfgs[a]["transverse_error_before_m"] for a in ARMS), 5),
        "standoff_m": round(standoff_f, 5) if standoff_f is not None else None,
        "stroke_status": None,
        "requested_stroke_m": round(max(req_strokes), 5) if req_strokes else None,
        "achieved_stroke_m": None, "rim_contact_suspected": False,
        "suggested_backoff_dxyz": None, "align_steps_used": 0, "stroke_steps_used": 0,
        "steps_used": 0, "planned_steps": 0, "ik_error": None, "note": None,
        "recoverable": True,
    }

    def finish(state_now: dict[str, Any]) -> dict[str, Any]:
        for a in ARMS:
            summary = motion._pose_summary(a, motion._joints(state_now, a))
            arm_rep[a]["final_xyz"] = summary["xyz"]
            arm_rep[a]["final_angles"] = {k: summary[k] for k in ("pitch", "yaw", "roll")}
            arm_rep[a].update(_gripper_state(primitives, state_now, a))
        result["arms"] = {"left": arm_rep["left"], "right": arm_rep["right"]}
        result["final_xyz"] = [0.0, 0.0, 0.0]
        result["final_angles"] = {"pitch": math.nan, "yaw": math.nan, "roll": math.nan}
        result["holding"] = all(arm_rep[a]["holding"] for a in ARMS)
        result["opening"] = round(min(arm_rep[a]["opening"] for a in ARMS), 3)
        result["transverse_error_after_m"] = round(
            max(arm_rep[a]["transverse_error_after_m"] for a in ARMS), 5
        )
        result["rim_contact_suspected"] = any(arm_rep[a]["rim_contact_suspected"] for a in ARMS)
        result["suggested_backoff_dxyz"] = next(
            (arm_rep[a]["suggested_backoff_dxyz"] for a in ARMS
             if arm_rep[a]["suggested_backoff_dxyz"] is not None),
            None,
        )
        result["remaining_episode_steps"] = _budget(primitives)[1]
        return result

    for a in ARMS:
        c = cfgs[a]
        if not c["skip_align"] and c["align_norm"] > max_lin_f + 1e-9:
            result.update(
                stop_reason="delta_exceeds_limit",
                stroke_status="rejected" if any_stroke else None,
                note=f"{a}: alignment delta {c['align_norm']:.4f} m exceeds max_linear_m "
                     f"{max_lin_f:.4f} m; re-stage closer before align_feature",
            )
            return finish(state0)
        if (c["stroke_vec"] is not None and c["requested_stroke_m"] is not None
                and c["requested_stroke_m"] > max_lin_f + 1e-9):
            result.update(
                stop_reason="delta_exceeds_limit", stroke_status="rejected",
                note=f"{a}: stroke {c['requested_stroke_m']:.4f} m exceeds max_linear_m "
                     f"{max_lin_f:.4f} m",
            )
            return finish(state0)

    any_allow_table = any(cfgs[a]["allow_table_contact"] for a in ARMS)
    per_arm_table = {a: cfgs[a]["allow_table_contact"] for a in ARMS}
    any_thin_moving = any(
        _is_thin_held(primitives, state0, a)
        and (not cfgs[a]["skip_align"] or cfgs[a]["stroke_vec"] is not None)
        for a in ARMS
    )
    plan_step_m = FINE_STEP_M if (any_allow_table or any_thin_moving) else LINE_STEP_M
    plan_step_deg = LINE_STEP_DEG * plan_step_m / LINE_STEP_M if any_allow_table else LINE_STEP_DEG

    align_ways: dict[str, list[list[float]]] = {}
    q_after_align_plan: dict[str, list[float]] = {}
    for a in ARMS:
        c = cfgs[a]
        if c["skip_align"]:
            align_ways[a], q_after_align_plan[a] = [], q0[a]
            continue
        align_target, ways_a, why_a = _plan_arm(
            a, q0[a], _spec({"dxyz": c["planned_dxyz"].tolist()}, a), "world",
            plan_step_m, plan_step_deg, allow_table_contact=c["allow_table_contact"],
            other_q=None, opening=open0[a], other_opening=open0[_other_arm(a)],
        )
        if why_a is not None:
            hint = _wrist_relief_hint(
                a, align_target, q0[a], allow_table_contact=c["allow_table_contact"],
                other_q=q0[_other_arm(a)], opening=open0[a], other_opening=open0[_other_arm(a)],
            )
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"{a}: {why_a}",
                stroke_status="rejected" if any_stroke else None,
            )
            if hint:
                result["note"] = hint
            return finish(state0)
        align_ways[a] = ways_a
        q_after_align_plan[a] = ways_a[-1] if ways_a else q0[a]

    align_pairs = _sync_pairs(q0["left"], q0["right"], align_ways["left"], align_ways["right"])
    if align_pairs:
        inter_why = _check_inter_arm_trajectory(
            q0["left"], q0["right"], align_pairs,
            left_opening=open0["left"], right_opening=open0["right"],
        )
        if inter_why is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"both align: {inter_why}",
                stroke_status="rejected" if any_stroke else None,
            )
            return finish(state0)

    stroke_ways: dict[str, list[list[float]]] = {}
    for a in ARMS:
        c = cfgs[a]
        if c["stroke_vec"] is None:
            stroke_ways[a] = []
            continue
        stroke_target, ways_a, why_a = _plan_arm(
            a, q_after_align_plan[a], _spec({"dxyz": c["stroke_vec"].tolist()}, a), "world",
            plan_step_m, plan_step_deg, allow_table_contact=c["allow_table_contact"],
            other_q=None, opening=open0[a], other_opening=open0[_other_arm(a)],
        )
        if why_a is not None:
            hint = _wrist_relief_hint(
                a, stroke_target, q_after_align_plan[a],
                allow_table_contact=c["allow_table_contact"],
                other_q=q_after_align_plan[_other_arm(a)],
                opening=open0[a], other_opening=open0[_other_arm(a)],
            )
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"{a} stroke: {why_a}",
                stroke_status="rejected",
            )
            if hint:
                result["note"] = hint
            return finish(state0)
        stroke_ways[a] = ways_a

    stroke_pairs = _sync_pairs(
        q_after_align_plan["left"], q_after_align_plan["right"],
        stroke_ways["left"], stroke_ways["right"],
    )
    if stroke_pairs:
        inter_why = _check_inter_arm_trajectory(
            q_after_align_plan["left"], q_after_align_plan["right"], stroke_pairs,
            left_opening=open0["left"], right_opening=open0["right"],
        )
        if inter_why is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"both stroke: {inter_why}",
                stroke_status="rejected",
            )
            return finish(state0)

    planned_steps = max(len(align_ways[a]) for a in ARMS) + max(len(stroke_ways[a]) for a in ARMS)
    result["planned_steps"] = planned_steps
    _, remaining = _budget(primitives)
    allowed = min(max_steps, remaining - reserve_steps)
    if planned_steps > 0 and allowed <= 0:
        result.update(
            stop_reason="episode_native_step_budget" if remaining <= 0 else "reserve_steps_exhausted",
            stroke_status="rejected" if any_stroke else None,
        )
        return finish(state0)
    if preview:
        if planned_steps > 0 and planned_steps > allowed:
            result.update(
                status="rejected", accepted=False, stop_reason="step_budget",
                stroke_status="rejected" if any_stroke else None,
                note=f"planned {planned_steps} steps exceeds allowance {allowed}",
            )
            return finish(state0)
        for a in ARMS:
            c = cfgs[a]
            arm_rep[a].update(
                status="planned",
                aligned=bool(c["skip_align"] and c["transverse_error_before_m"] <= c["tol_f"]),
                reached=False,
                stroke_status="planned" if c["stroke_vec"] is not None else None,
            )
        result.update(
            status="planned", accepted=True,
            aligned=all(arm_rep[a]["aligned"] for a in ARMS),
            reached=False, stroke_status="planned" if any_stroke else None,
        )
        return finish(state0)

    result["accepted"] = True
    if all(cfgs[a]["skip_align"] for a in ARMS):
        align_steps, align_stalled, align_stop_reason = 0, False, None
        align_arm_stalled = {a: False for a in ARMS}
        state_mid = state0
        tcp_after_align = dict(tcp_before)
    else:
        align_res = nudge(
            primitives, start_joints, "both",
            left={"dxyz": [0.0, 0.0, 0.0] if cfgs["left"]["skip_align"] else cfgs["left"]["planned_dxyz"].tolist()},
            right={"dxyz": [0.0, 0.0, 0.0] if cfgs["right"]["skip_align"] else cfgs["right"]["planned_dxyz"].tolist()},
            frame="world", compact=True, stop_on_stall=stop_on_stall, independent_stall=False,
            stall_steps=stall_steps, max_linear_m=max_lin_f, max_steps=max_steps,
            reserve_steps=reserve_steps, tol_m=min(TOL_M, min(cfgs[a]["tol_f"] for a in ARMS)),
            allow_table_contact=any_allow_table, _arm_allow_table_contact=per_arm_table,
        )
        if "error" in align_res:
            return {"error": f"align_feature: {align_res['error']}"}
        align_steps = int(align_res.get("steps_used", 0))
        align_stalled = bool(align_res.get("stalled"))
        align_stop_reason = align_res.get("stop_reason")
        align_arm_stalled = {
            a: bool((align_res.get("arms") or {}).get(a, {}).get("stalled")) for a in ARMS
        }
        if align_res.get("ik_error"):
            result["ik_error"] = align_res["ik_error"]
        state_mid = motion._obs(primitives)["state"]
        tcp_after_align = {a: motion._pose_of(a, motion._joints(state_mid, a))[:3, 3] for a in ARMS}

    for a in ARMS:
        c = cfgs[a]
        rem_diff = c["t_xyz"] - (c["f_xyz"] + (tcp_after_align[a] - tcp_before[a]))
        axial_after = float(np.dot(rem_diff, c["u_axis"]))
        trans_err_after = float(np.linalg.norm(rem_diff - axial_after * c["u_axis"]))
        standoff_ok = (
            True if c["standoff_f"] is None
            else abs(axial_after - c["standoff_f"]) <= max(c["tol_f"], TOL_M)
        )
        arm_rep[a]["aligned"] = bool(trans_err_after <= c["tol_f"] and standoff_ok)
        arm_rep[a]["transverse_error_after_m"] = round(trans_err_after, 5)
        arm_rep[a]["stalled"] = align_arm_stalled[a]

    both_aligned = all(arm_rep[a]["aligned"] for a in ARMS)
    result.update(aligned=both_aligned, align_steps_used=align_steps)

    if not any_stroke:
        for a in ARMS:
            if arm_rep[a]["aligned"]:
                arm_rep[a].update(status="completed", reached=True, stop_reason=None)
            elif arm_rep[a]["stalled"]:
                arm_rep[a].update(status="stopped", reached=False, stop_reason="contact_or_stall")
            else:
                arm_rep[a].update(
                    status="partial" if align_steps > 0 else "rejected", reached=False,
                    stop_reason=align_stop_reason or "alignment_tolerance_not_met",
                )
        result.update(
            steps_used=align_steps, stroke_steps_used=0, stalled=align_stalled,
            recoverable=(align_steps == 0),
        )
        if both_aligned:
            result.update(status="completed", reached=True, stop_reason=None)
            if all(cfgs[a]["skip_align"] for a in ARMS):
                result["note"] = "already aligned within 0.5 mm: 0 alignment steps"
        elif align_stalled:
            result.update(
                status="stopped", reached=False, stop_reason="contact_or_stall",
                note="alignment stalled before reaching transverse tolerance",
            )
        else:
            result.update(
                status="partial" if align_steps > 0 else "rejected", reached=False,
                stop_reason=align_stop_reason or "alignment_tolerance_not_met",
            )
        return finish(state_mid)

    if not both_aligned:
        for a in ARMS:
            arm_rep[a]["stroke_status"] = "skipped" if cfgs[a]["stroke_vec"] is not None else None
            arm_rep[a]["achieved_stroke_m"] = 0.0 if cfgs[a]["stroke_vec"] is not None else None
            arm_rep[a]["reached"] = False
            arm_rep[a]["status"] = (
                "stopped" if arm_rep[a]["stalled"]
                else ("partial" if align_steps > 0 else "rejected")
            )
        result.update(
            steps_used=align_steps, stroke_steps_used=0, stroke_status="skipped",
            achieved_stroke_m=0.0, stalled=align_stalled, recoverable=(align_steps == 0),
            reached=False,
            status="stopped" if align_stalled else ("partial" if align_steps > 0 else "rejected"),
            stop_reason="contact_or_stall" if align_stalled
            else (align_stop_reason or "alignment_tolerance_not_met"),
            note="alignment stalled; engagement stroke skipped" if align_stalled
            else "alignment residual exceeded align_tol_m; engagement stroke skipped",
        )
        return finish(state_mid)

    stroke_budget = max_steps - align_steps
    if stroke_budget <= 0:
        for a in ARMS:
            if cfgs[a]["stroke_vec"] is not None:
                arm_rep[a].update(
                    stroke_status="rejected", achieved_stroke_m=0.0, reached=False,
                    status="partial" if align_steps > 0 else "rejected", stop_reason="step_budget",
                )
        result.update(
            steps_used=align_steps, stroke_steps_used=0, stroke_status="rejected",
            achieved_stroke_m=0.0, stalled=False, recoverable=(align_steps == 0),
            reached=False, status="partial" if align_steps > 0 else "rejected",
            stop_reason="step_budget", note="step budget exhausted before engagement stroke",
        )
        return finish(state_mid)

    stroke_res = nudge(
        primitives, start_joints, "both",
        left={"dxyz": [0.0, 0.0, 0.0] if cfgs["left"]["stroke_vec"] is None else cfgs["left"]["stroke_vec"].tolist()},
        right={"dxyz": [0.0, 0.0, 0.0] if cfgs["right"]["stroke_vec"] is None else cfgs["right"]["stroke_vec"].tolist()},
        frame="world", compact=True, stop_on_stall=stop_on_stall,
        independent_stall=independent_stall, stall_steps=stall_steps, max_linear_m=max_lin_f,
        max_steps=stroke_budget, reserve_steps=reserve_steps,
        allow_table_contact=any_allow_table, _arm_allow_table_contact=per_arm_table,
    )
    if "error" in stroke_res:
        return {"error": f"align_feature: {stroke_res['error']}"}
    stroke_steps = int(stroke_res.get("steps_used", 0))
    total_steps = align_steps + stroke_steps
    state_end = motion._obs(primitives)["state"]
    tcp_after_stroke = {a: motion._pose_of(a, motion._joints(state_end, a))[:3, 3] for a in ARMS}
    stroke_stalled = bool(stroke_res.get("stalled"))
    stroke_arms_info = stroke_res.get("arms") or {}
    achieved_list: list[float] = []

    for a in ARMS:
        c = cfgs[a]
        if c["stroke_vec"] is None:
            arm_rep[a].update(
                stroke_status=None, achieved_stroke_m=None, stalled=False, reached=True,
                status="completed", stop_reason=None, rim_contact_suspected=False,
                suggested_backoff_dxyz=None,
            )
            continue
        assert c["stroke_unit"] is not None
        ach = float(np.dot(tcp_after_stroke[a] - tcp_after_align[a], c["stroke_unit"]))
        achieved_list.append(ach)
        a_info = stroke_arms_info.get(a) or {}
        a_reached = bool(a_info.get("reached"))
        a_stalled = bool(a_info.get("stalled"))
        arm_rep[a].update(achieved_stroke_m=round(ach, 5), stalled=a_stalled, reached=a_reached)
        if a_reached:
            arm_rep[a].update(
                stroke_status="completed", status="completed", stop_reason=None,
                rim_contact_suspected=False, suggested_backoff_dxyz=None,
            )
        elif a_stalled or (not independent_stall and stroke_stalled):
            rim = bool(
                c["requested_stroke_m"] is not None
                and c["requested_stroke_m"] > 0.0
                and ach < 0.7 * c["requested_stroke_m"]
            )
            arm_rep[a].update(
                stroke_status="stopped", status="stopped", stop_reason="contact_or_stall",
                rim_contact_suspected=rim,
                suggested_backoff_dxyz=(
                    [round(float(-v), 5) for v in c["stroke_unit"] * 0.004] if rim else None
                ),
            )
        else:
            arm_rep[a].update(
                stroke_status=str(stroke_res.get("status") or "partial"),
                status="partial" if total_steps > 0 else "rejected",
                stop_reason=stroke_res.get("stop_reason") or "stroke_not_reached",
                rim_contact_suspected=False, suggested_backoff_dxyz=None,
            )

    all_reached = all(arm_rep[a]["reached"] for a in ARMS)
    result.update(
        steps_used=total_steps, stroke_steps_used=stroke_steps,
        achieved_stroke_m=round(min(achieved_list), 5) if achieved_list else None,
        stalled=stroke_stalled, recoverable=(total_steps == 0),
    )
    if stroke_res.get("ik_error"):
        result["ik_error"] = stroke_res["ik_error"]
    if all_reached:
        result.update(status="completed", reached=True, stroke_status="completed", stop_reason=None)
    elif stroke_stalled:
        result.update(
            status="stopped", reached=False, stroke_status="stopped",
            stop_reason="contact_or_stall",
            note=stroke_res.get("note") or "engagement stroke stopped on contact or stall",
        )
    else:
        result.update(
            status="partial" if total_steps > 0 else "rejected", reached=False,
            stroke_status=str(stroke_res.get("status") or "partial"),
            stop_reason=stroke_res.get("stop_reason") or "stroke_not_reached",
        )
        if stroke_res.get("note"):
            result["note"] = stroke_res["note"]
    return finish(state_end)


def align_feature(
    primitives: Any,
    start_joints: dict[str, list[float]],
    feature_xyz: Sequence[float] | None = None,
    target_xyz: Sequence[float] | None = None,
    arm: str = "right",
    *,
    axis: Sequence[float] = (0.0, 0.0, -1.0),
    standoff_m: float | None = None,
    stroke_m: float | None = None,
    stroke_dxyz: Sequence[float] | None = None,
    left: Mapping[str, Any] | None = None,
    right: Mapping[str, Any] | None = None,
    independent_stall: bool = True,
    align_tol_m: float = ALIGN_TOL_M,
    max_linear_m: float = MAX_LINEAR_M,
    max_steps: int = 15,
    reserve_steps: int = 20,
    stop_on_stall: bool = True,
    stall_steps: int = 2,
    allow_table_contact: bool = False,
    preview: bool = False,
) -> dict[str, Any]:
    """Zero transverse offset from a held distal feature to a target, with optional stroke.

    Projects ``target_xyz - feature_xyz`` perpendicular to the unit engagement ``axis``
    (plus an optional axial ``standoff_m`` upstream along ``-axis``), preflights reach,
    table clearance and budget, executes a compact guarded ``nudge`` for alignment,
    measures residual ``transverse_error_after_m`` from the actual TCP shift, and when
    ``aligned`` (within ``align_tol_m``) optionally executes a guarded engagement stroke
    (``stroke_m`` along ``+axis`` or explicit ``stroke_dxyz``). With ``arm="both"``,
    pass ``left={...}`` and ``right={...}`` to align and stroke both arms simultaneously.
    """

    try:
        if arm not in (*ARMS, "both"):
            raise ValueError("arm must be 'left', 'right' or 'both'")
        axis_raw = _vec3(axis, "axis")
        axis_norm = float(np.linalg.norm(axis_raw))
        if axis_norm <= 1e-9:
            raise ValueError("axis must be non-zero")
        u_axis = axis_raw / axis_norm
        standoff_f: float | None = None
        if standoff_m is not None:
            standoff_f = _finite(standoff_m, "standoff_m")
            if standoff_f < 0.0:
                raise ValueError("standoff_m must be >= 0")
        if stroke_m is not None and stroke_dxyz is not None:
            raise ValueError("pass at most one of stroke_m or stroke_dxyz")
        stroke_vec: np.ndarray | None = None
        stroke_unit: np.ndarray | None = None
        requested_stroke_m: float | None = None
        if stroke_m is not None:
            stroke_f = _finite(stroke_m, "stroke_m")
            if stroke_f <= 0.0:
                raise ValueError("stroke_m must be > 0")
            stroke_vec = u_axis * stroke_f
            stroke_unit = u_axis
            requested_stroke_m = stroke_f
        elif stroke_dxyz is not None:
            stroke_vec = _vec3(stroke_dxyz, "stroke_dxyz")
            requested_stroke_m = float(np.linalg.norm(stroke_vec))
            if requested_stroke_m <= 1e-6:
                raise ValueError("stroke_dxyz must be non-zero")
            stroke_unit = stroke_vec / requested_stroke_m
        tol_f = _finite(align_tol_m, "align_tol_m")
        if not 0.0005 <= tol_f <= 0.02:
            raise ValueError("align_tol_m must be between 0.0005 and 0.02 m")
        max_lin_f = _finite(max_linear_m, "max_linear_m")
        if not 0.0 < max_lin_f <= HARD_LINEAR_M:
            raise ValueError(f"max_linear_m must be in (0, {HARD_LINEAR_M}]")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(reserve_steps) is not int or reserve_steps < 0:
            raise ValueError("reserve_steps must be a non-negative integer")
        if type(stall_steps) is not int or not 1 <= stall_steps <= 5:
            raise ValueError("stall_steps must be an integer between 1 and 5")
        if (not isinstance(stop_on_stall, bool) or not isinstance(preview, bool)
                or not isinstance(allow_table_contact, bool)
                or not isinstance(independent_stall, bool)):
            raise ValueError(
                "stop_on_stall, independent_stall, allow_table_contact and preview must be booleans"
            )
        if arm == "both":
            if feature_xyz is not None or target_xyz is not None:
                raise ValueError("feature_xyz/target_xyz must be None when arm='both'")
        else:
            if left is not None or right is not None:
                raise ValueError("left= / right= go with arm='both'")
            if feature_xyz is None or target_xyz is None:
                raise ValueError("feature_xyz and target_xyz are required")
            f_xyz = _vec3(feature_xyz, "feature_xyz")
            t_xyz = _vec3(target_xyz, "target_xyz")
    except ValueError as exc:
        return {"error": f"align_feature: {exc}"}

    if arm == "both":
        return _align_feature_both(
            primitives, start_joints, left=left, right=right, u_axis_default=u_axis,
            standoff_f=standoff_f, stroke_m=stroke_m, stroke_dxyz=stroke_dxyz,
            independent_stall=independent_stall, tol_f=tol_f, max_lin_f=max_lin_f,
            max_steps=max_steps, reserve_steps=reserve_steps, stop_on_stall=stop_on_stall,
            stall_steps=stall_steps, allow_table_contact=allow_table_contact, preview=preview,
        )

    raw_diff = t_xyz - f_xyz
    axial_gap_m = float(np.dot(raw_diff, u_axis))
    transverse_delta = raw_diff - axial_gap_m * u_axis
    transverse_error_before_m = float(np.linalg.norm(transverse_delta))
    if standoff_f is None:
        planned_dxyz = transverse_delta.copy()
    else:
        desired_feature = t_xyz - u_axis * standoff_f
        planned_dxyz = desired_feature - f_xyz
    align_norm = float(np.linalg.norm(planned_dxyz))
    skip_align = align_norm <= ZERO_DELTA_M

    state0 = motion._obs(primitives)["state"]
    q0 = motion._joints(state0, arm)
    tcp_before = motion._pose_of(arm, q0)[:3, 3]

    result: dict[str, Any] = {
        "arm": arm,
        "status": "rejected",
        "aligned": False,
        "reached": False,
        "accepted": False,
        "preview": preview,
        "stalled": False,
        "stop_reason": None,
        "feature_xyz": [round(float(v), 5) for v in f_xyz],
        "target_xyz": [round(float(v), 5) for v in t_xyz],
        "axis": [round(float(v), 6) for v in u_axis],
        "axial_gap_m": round(axial_gap_m, 5),
        "transverse_delta": [round(float(v), 5) for v in transverse_delta],
        "planned_dxyz": [round(float(v), 5) for v in planned_dxyz],
        "transverse_error_before_m": round(transverse_error_before_m, 5),
        "transverse_error_after_m": round(transverse_error_before_m, 5),
        "standoff_m": round(standoff_f, 5) if standoff_f is not None else None,
        "stroke_status": None,
        "requested_stroke_m": round(requested_stroke_m, 5)
        if requested_stroke_m is not None else None,
        "achieved_stroke_m": None,
        "rim_contact_suspected": False,
        "suggested_backoff_dxyz": None,
        "align_steps_used": 0,
        "stroke_steps_used": 0,
        "steps_used": 0,
        "planned_steps": 0,
        "ik_error": None,
        "note": None,
        "recoverable": True,
    }

    def finish(state_now: dict[str, Any]) -> dict[str, Any]:
        q_now = motion._joints(state_now, arm)
        summary = motion._pose_summary(arm, q_now)
        result["final_xyz"] = summary["xyz"]
        result["final_angles"] = {k: summary[k] for k in ("pitch", "yaw", "roll")}
        result.update(_gripper_state(primitives, state_now, arm))
        result["remaining_episode_steps"] = _budget(primitives)[1]
        result["arms"] = {
            arm: {
                k: result[k]
                for k in (
                    "status", "aligned", "reached", "stalled", "stop_reason",
                    "feature_xyz", "target_xyz", "axis", "axial_gap_m",
                    "transverse_delta", "planned_dxyz",
                    "transverse_error_before_m", "transverse_error_after_m",
                    "standoff_m", "stroke_status", "requested_stroke_m",
                    "achieved_stroke_m", "rim_contact_suspected",
                    "suggested_backoff_dxyz", "final_xyz", "final_angles",
                    "opening", "gripper_commanded", "holding", "width_m",
                )
                if k in result
            }
        }
        return result

    if not skip_align and align_norm > max_lin_f + 1e-9:
        result.update(
            stop_reason="delta_exceeds_limit",
            stroke_status="rejected" if stroke_vec is not None else None,
            note=f"{arm}: alignment delta {align_norm:.4f} m exceeds max_linear_m "
                 f"{max_lin_f:.4f} m; re-stage closer before align_feature",
        )
        return finish(state0)

    if stroke_vec is not None and requested_stroke_m is not None and requested_stroke_m > max_lin_f + 1e-9:
        result.update(
            stop_reason="delta_exceeds_limit",
            stroke_status="rejected",
            note=f"{arm}: stroke {requested_stroke_m:.4f} m exceeds max_linear_m "
                 f"{max_lin_f:.4f} m",
        )
        return finish(state0)

    plan_step_m = (
        FINE_STEP_M if (allow_table_contact or _is_thin_held(primitives, state0, arm))
        else LINE_STEP_M
    )
    plan_step_deg = (
        LINE_STEP_DEG * plan_step_m / LINE_STEP_M if allow_table_contact else LINE_STEP_DEG
    )
    other_arm = _other_arm(arm)
    other_q0 = _safe_joints(state0, other_arm)
    opening_arm0 = _safe_opening(state0, arm)
    opening_other0 = _safe_opening(state0, other_arm)

    if skip_align:
        align_ways: list[list[float]] = []
        q_after_align_plan = q0
    else:
        align_spec = _spec({"dxyz": planned_dxyz.tolist()}, arm)
        align_target, align_ways, align_why = _plan_arm(
            arm, q0, align_spec, "world", plan_step_m, plan_step_deg,
            allow_table_contact=allow_table_contact,
            other_q=other_q0,
            opening=opening_arm0,
            other_opening=opening_other0,
        )
        if align_why is not None:
            hint = _wrist_relief_hint(
                arm, align_target, q0,
                allow_table_contact=allow_table_contact,
                other_q=other_q0, opening=opening_arm0, other_opening=opening_other0,
            )
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm}: {align_why}",
                stroke_status="rejected" if stroke_vec is not None else None,
            )
            if hint:
                result["note"] = f"{result['note']} {hint}" if result.get("note") else hint
            return finish(state0)
        q_after_align_plan = align_ways[-1] if align_ways else q0

    if stroke_vec is not None:
        stroke_spec = _spec({"dxyz": stroke_vec.tolist()}, arm)
        stroke_target, stroke_ways, stroke_why = _plan_arm(
            arm, q_after_align_plan, stroke_spec, "world", plan_step_m, plan_step_deg,
            allow_table_contact=allow_table_contact,
            other_q=other_q0,
            opening=opening_arm0,
            other_opening=opening_other0,
        )
        if stroke_why is not None:
            hint = _wrist_relief_hint(
                arm, stroke_target, q_after_align_plan,
                allow_table_contact=allow_table_contact,
                other_q=other_q0, opening=opening_arm0, other_opening=opening_other0,
            )
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm} stroke: {stroke_why}",
                stroke_status="rejected",
            )
            if hint:
                result["note"] = f"{result['note']} {hint}" if result.get("note") else hint
            return finish(state0)
    else:
        stroke_ways = []

    planned_steps = len(align_ways) + len(stroke_ways)
    result["planned_steps"] = planned_steps
    _, remaining = _budget(primitives)
    allowed = min(max_steps, remaining - reserve_steps)

    if planned_steps > 0 and allowed <= 0:
        result.update(
            stop_reason="episode_native_step_budget" if remaining <= 0
            else "reserve_steps_exhausted",
            stroke_status="rejected" if stroke_vec is not None else None,
        )
        return finish(state0)

    if preview:
        if planned_steps > 0 and planned_steps > allowed:
            result.update(
                status="rejected",
                accepted=False,
                stop_reason="step_budget",
                stroke_status="rejected" if stroke_vec is not None else None,
                note=f"planned {planned_steps} steps exceeds allowance {allowed}",
            )
            return finish(state0)
        result.update(
            status="planned",
            accepted=True,
            aligned=bool(skip_align and transverse_error_before_m <= tol_f),
            reached=False,
            stroke_status="planned" if stroke_vec is not None else None,
        )
        return finish(state0)

    result["accepted"] = True
    if skip_align:
        align_steps = 0
        align_stalled = False
        align_stop_reason: str | None = None
        state_mid = state0
        tcp_after_align = tcp_before
    else:
        align_res = nudge(
            primitives, start_joints, arm, dxyz=planned_dxyz.tolist(), frame="world",
            compact=True, stop_on_stall=stop_on_stall, stall_steps=stall_steps,
            max_linear_m=max_lin_f, max_steps=max_steps, reserve_steps=reserve_steps,
            tol_m=min(TOL_M, tol_f), allow_table_contact=allow_table_contact,
        )
        if "error" in align_res:
            return {"error": f"align_feature: {align_res['error']}"}
        align_steps = int(align_res.get("steps_used", 0))
        align_stalled = bool(align_res.get("stalled"))
        align_stop_reason = align_res.get("stop_reason")
        if align_res.get("ik_error"):
            result["ik_error"] = align_res["ik_error"]
        state_mid = motion._obs(primitives)["state"]
        tcp_after_align = motion._pose_of(arm, motion._joints(state_mid, arm))[:3, 3]

    actual_align_shift = tcp_after_align - tcp_before
    feature_after = f_xyz + actual_align_shift
    rem_diff = t_xyz - feature_after
    axial_gap_after = float(np.dot(rem_diff, u_axis))
    rem_transverse = rem_diff - axial_gap_after * u_axis
    transverse_error_after = float(np.linalg.norm(rem_transverse))
    standoff_ok = (True if standoff_f is None
                   else abs(axial_gap_after - standoff_f) <= max(tol_f, TOL_M))
    aligned = bool(transverse_error_after <= tol_f and standoff_ok)
    result.update(
        aligned=aligned,
        transverse_error_after_m=round(transverse_error_after, 5),
        align_steps_used=align_steps,
    )

    if stroke_vec is None:
        result.update(
            steps_used=align_steps,
            stroke_steps_used=0,
            stalled=align_stalled,
            recoverable=(align_steps == 0),
        )
        if aligned:
            result.update(status="completed", reached=True, stop_reason=None)
            if skip_align:
                result["note"] = "already aligned within 0.5 mm: 0 alignment steps"
        elif align_stalled:
            result.update(
                status="stopped",
                reached=False,
                stop_reason="contact_or_stall",
                note="alignment stalled before reaching transverse tolerance",
            )
        else:
            result.update(
                status="partial" if align_steps > 0 else "rejected",
                reached=False,
                stop_reason=align_stop_reason or "alignment_tolerance_not_met",
            )
        return finish(state_mid)

    if not aligned:
        result.update(
            steps_used=align_steps,
            stroke_steps_used=0,
            stroke_status="skipped",
            achieved_stroke_m=0.0,
            stalled=align_stalled,
            recoverable=(align_steps == 0),
            reached=False,
        )
        if align_stalled:
            result.update(
                status="stopped",
                stop_reason="contact_or_stall",
                note="alignment stalled; engagement stroke skipped",
            )
        else:
            result.update(
                status="partial" if align_steps > 0 else "rejected",
                stop_reason=align_stop_reason or "alignment_tolerance_not_met",
                note="alignment residual exceeded align_tol_m; engagement stroke skipped",
            )
        return finish(state_mid)

    stroke_budget = max_steps - align_steps
    if stroke_budget <= 0:
        result.update(
            steps_used=align_steps,
            stroke_steps_used=0,
            stroke_status="rejected",
            achieved_stroke_m=0.0,
            stalled=False,
            recoverable=(align_steps == 0),
            reached=False,
            status="partial" if align_steps > 0 else "rejected",
            stop_reason="step_budget",
            note="step budget exhausted before engagement stroke",
        )
        return finish(state_mid)

    stroke_res = nudge(
        primitives, start_joints, arm, dxyz=stroke_vec.tolist(), frame="world",
        compact=True, stop_on_stall=stop_on_stall, stall_steps=stall_steps,
        max_linear_m=max_lin_f, max_steps=stroke_budget, reserve_steps=reserve_steps,
        allow_table_contact=allow_table_contact,
    )
    if "error" in stroke_res:
        return {"error": f"align_feature: {stroke_res['error']}"}
    stroke_steps = int(stroke_res.get("steps_used", 0))
    total_steps = align_steps + stroke_steps
    state_end = motion._obs(primitives)["state"]
    tcp_after_stroke = motion._pose_of(arm, motion._joints(state_end, arm))[:3, 3]
    assert stroke_unit is not None
    achieved_stroke = float(np.dot(tcp_after_stroke - tcp_after_align, stroke_unit))
    stroke_stalled = bool(stroke_res.get("stalled"))
    stroke_reached = bool(stroke_res.get("reached"))
    stroke_status = str(stroke_res.get("status") or "rejected")
    result.update(
        steps_used=total_steps,
        stroke_steps_used=stroke_steps,
        stroke_status=stroke_status,
        achieved_stroke_m=round(achieved_stroke, 5),
        stalled=stroke_stalled,
        recoverable=(total_steps == 0),
    )
    if stroke_res.get("ik_error"):
        result["ik_error"] = stroke_res["ik_error"]
    if stroke_reached:
        result.update(status="completed", reached=True, stop_reason=None)
    elif stroke_stalled:
        rim_suspected = bool(
            requested_stroke_m is not None
            and requested_stroke_m > 0.0
            and achieved_stroke < 0.7 * requested_stroke_m
        )
        suggested_backoff = (
            [round(float(-v), 5) for v in stroke_unit * 0.004]
            if rim_suspected
            else None
        )
        result.update(
            status="stopped",
            reached=False,
            stop_reason="contact_or_stall",
            rim_contact_suspected=rim_suspected,
            suggested_backoff_dxyz=suggested_backoff,
            note=stroke_res.get("note") or "engagement stroke stopped on contact or stall",
        )
    else:
        result.update(
            status="partial" if total_steps > 0 else "rejected",
            reached=False,
            stop_reason=stroke_res.get("stop_reason") or "stroke_not_reached",
        )
        if stroke_res.get("note"):
            result["note"] = stroke_res["note"]
    return finish(state_end)


def preview_align_feature(
    primitives: Any, start_joints: dict[str, list[float]], **kwargs: Any
) -> dict[str, Any]:
    """Read-only preflight for ``align_feature`` without moving any arm or gripper."""

    result = align_feature(primitives, start_joints, **kwargs, preview=True)
    if "error" not in result:
        result["preview"] = True
    return result


preview_align_feature._readonly = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# fine delta grasp & in-air handoff primitive
# ---------------------------------------------------------------------------


def _grip_to(
    primitives: Any,
    arm: str,
    command: float,
    max_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], int, float]:
    """Drive one gripper to ``command`` (0.0 close, (0..1] open) within ``max_steps``."""

    state = motion._obs(primitives)["state"]
    hold = {a: motion._joints(state, a) for a in ARMS}
    before = motion._opening(state, arm)
    steps = 0
    status: dict[str, Any] = {}
    last = before
    limit = min(motion.GRIP_MAX_ACTIONS, max(0, max_steps))
    for _ in range(limit):
        primitives._check_cancelled()
        state, status = motion._act(primitives, state, hold, {arm: float(command)})
        steps += 1
        now = motion._opening(state, arm)
        if steps >= 2 and abs(now - last) < motion.GRIP_STILL:
            break
        last = now
        if motion._spent(status):
            break
    return state, status, steps, motion._opening(state, arm)


def _grip_to_many(
    primitives: Any,
    commands: Mapping[str, float],
    max_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, float]]:
    """Drive one or both grippers simultaneously to ``commands`` within ``max_steps``."""

    state = motion._obs(primitives)["state"]
    hold = {a: motion._joints(state, a) for a in ARMS}
    active = {a: float(cmd) for a, cmd in commands.items()}
    steps = 0
    status: dict[str, Any] = {}
    last = {a: motion._opening(state, a) for a in active}
    limit = min(motion.GRIP_MAX_ACTIONS, max(0, max_steps))
    for _ in range(limit):
        primitives._check_cancelled()
        state, status = motion._act(primitives, state, hold, active)
        steps += 1
        now = {a: motion._opening(state, a) for a in active}
        if steps >= 2 and all(abs(now[a] - last[a]) < motion.GRIP_STILL for a in active):
            break
        last = now
        if motion._spent(status):
            break
    return state, status, steps, {a: motion._opening(state, a) for a in ARMS}


def _delta_grasp_arm_spec(
    raw: Any,
    where: str,
    *,
    default_axis: np.ndarray,
    default_standoff: float,
    default_overdrive: float,
    default_opening: float | None,
    default_contact_table: bool,
    default_lift_dxyz: Sequence[float] | None,
    default_tol: float,
) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        extra = sorted(
            set(raw)
            - {
                "target_xyz", "target", "axis", "standoff_m", "overdrive_m",
                "opening", "contact_table", "lift_dxyz", "align_tol_m",
            }
        )
        if extra:
            raise ValueError(f"{where} has unknown keys {extra}")
        t_raw = raw.get("target_xyz") if raw.get("target_xyz") is not None else raw.get("target")
        if t_raw is None:
            raise ValueError(f"{where} requires target_xyz/target")
        t_xyz = _vec3(t_raw, f"{where}.target_xyz")
        if "axis" in raw and raw["axis"] is not None:
            axis_raw = _vec3(raw["axis"], f"{where}.axis")
            axis_norm = float(np.linalg.norm(axis_raw))
            if axis_norm <= 1e-9:
                raise ValueError(f"{where}.axis must be non-zero")
            u_axis = axis_raw / axis_norm
        else:
            u_axis = default_axis.copy()
        standoff_f = _finite(raw.get("standoff_m", default_standoff), f"{where}.standoff_m")
        overdrive_f = _finite(raw.get("overdrive_m", default_overdrive), f"{where}.overdrive_m")
        op_raw = raw["opening"] if "opening" in raw else default_opening
        contact_t = raw.get("contact_table", default_contact_table)
        lift_raw = raw["lift_dxyz"] if "lift_dxyz" in raw else default_lift_dxyz
        tol_f = _finite(raw.get("align_tol_m", default_tol), f"{where}.align_tol_m")
    else:
        t_xyz = _vec3(raw, f"{where}")
        u_axis = default_axis.copy()
        standoff_f = default_standoff
        overdrive_f = default_overdrive
        op_raw = default_opening
        contact_t = default_contact_table
        lift_raw = default_lift_dxyz
        tol_f = default_tol
    if standoff_f < 0.0:
        raise ValueError(f"{where}.standoff_m must be >= 0")
    if overdrive_f < 0.0:
        raise ValueError(f"{where}.overdrive_m must be >= 0")
    opening_f: float | None = None
    if op_raw is not None:
        opening_f = _finite(op_raw, f"{where}.opening")
        if not 0.0 < opening_f <= 1.0:
            raise ValueError(f"{where}.opening must be in (0, 1]")
    if not isinstance(contact_t, bool):
        raise ValueError(f"{where}.contact_table must be a boolean")
    lift_vec: np.ndarray | None = None
    lift_norm = 0.0
    if lift_raw is not None:
        lift_vec = _vec3(lift_raw, f"{where}.lift_dxyz")
        lift_norm = float(np.linalg.norm(lift_vec))
        if lift_norm <= 1e-6:
            raise ValueError(f"{where}.lift_dxyz must be non-zero")
    if not 0.0005 <= tol_f <= 0.02:
        raise ValueError(f"{where}.align_tol_m must be between 0.0005 and 0.02 m")
    return {
        "t_xyz": t_xyz, "u_axis": u_axis, "standoff_f": standoff_f,
        "overdrive_f": overdrive_f, "opening_f": opening_f,
        "contact_table": contact_t, "lift_vec": lift_vec, "lift_norm": lift_norm,
        "tol_f": tol_f,
    }


def _delta_grasp_both(
    primitives: Any,
    start_joints: dict[str, list[float]],
    *,
    left: Mapping[str, Any] | Sequence[float] | None,
    right: Mapping[str, Any] | Sequence[float] | None,
    u_axis_default: np.ndarray,
    standoff_f: float,
    overdrive_f: float,
    opening_f: float | None,
    contact_table: bool,
    lift_dxyz: Sequence[float] | None,
    tol_f: float,
    max_lin_f: float,
    max_steps: int,
    reserve_steps: int,
    stall_steps: int,
    preview: bool,
) -> dict[str, Any]:
    try:
        if left is None or right is None:
            raise ValueError("arm='both' requires both left= and right= targets")
        cfgs = {
            a: _delta_grasp_arm_spec(
                raw, a, default_axis=u_axis_default, default_standoff=standoff_f,
                default_overdrive=overdrive_f, default_opening=opening_f,
                default_contact_table=contact_table, default_lift_dxyz=lift_dxyz,
                default_tol=tol_f,
            )
            for a, raw in (("left", left), ("right", right))
        }
    except ValueError as exc:
        return {"error": f"delta_grasp: {exc}"}

    state0 = motion._obs(primitives)["state"]
    q0 = {a: motion._joints(state0, a) for a in ARMS}
    pose0 = {a: motion._pose_of(a, q0[a]) for a in ARMS}
    tcp_before = {a: pose0[a][:3, 3] for a in ARMS}
    opening_before = {a: motion._opening(state0, a) for a in ARMS}
    grip_cmds = getattr(primitives, "_grip_commands", None) or {}
    any_contact = any(cfgs[a]["contact_table"] for a in ARMS)
    per_arm_table = {a: cfgs[a]["contact_table"] for a in ARMS}
    any_lift = any(cfgs[a]["lift_vec"] is not None for a in ARMS)

    needs_preopen: dict[str, bool] = {}
    transverse_delta: dict[str, np.ndarray] = {}
    transverse_err_before: dict[str, float] = {}
    skip_align: dict[str, bool] = {}
    desired_tcp: dict[str, np.ndarray] = {}
    axial_plan_vec: dict[str, np.ndarray] = {}
    axial_plan_norm: dict[str, float] = {}
    skip_approach: dict[str, bool] = {}

    for a in ARMS:
        c = cfgs[a]
        last_cmd = grip_cmds.get(a)
        needs_preopen[a] = bool(
            c["opening_f"] is not None
            and (
                abs(opening_before[a] - c["opening_f"]) > PREOPEN_TOL
                or (last_cmd is not None and last_cmd <= 0.05)
            )
        )
        raw_diff = c["t_xyz"] - tcp_before[a]
        axial_gap = float(np.dot(raw_diff, c["u_axis"]))
        perp = raw_diff - axial_gap * c["u_axis"]
        transverse_delta[a] = perp
        transverse_err_before[a] = float(np.linalg.norm(perp))
        skip_align[a] = transverse_err_before[a] <= ZERO_DELTA_M
        dtcp = c["t_xyz"] - c["u_axis"] * c["standoff_f"] + c["u_axis"] * c["overdrive_f"]
        if c["contact_table"] and c["u_axis"][2] < -1e-6 and c["t_xyz"][2] >= kin.TABLE_Z - 0.005:
            tip_delta_z = float(pose0[a][2, 0]) * (kin.TIP_OFFSET - kin.TCP_OFFSET)
            min_safe_tcp_z = kin.TABLE_Z - TABLE_TOUCH_OVERDRIVE_M + 0.0005 - tip_delta_z
            if dtcp[2] < min_safe_tcp_z:
                pullback = (min_safe_tcp_z - float(dtcp[2])) / (-float(c["u_axis"][2]))
                dtcp = dtcp - pullback * c["u_axis"]
        desired_tcp[a] = dtcp
        tcp_mid_plan = tcp_before[a] + (np.zeros(3) if skip_align[a] else perp)
        ax_dist = float(np.dot(dtcp - tcp_mid_plan, c["u_axis"]))
        axial_plan_vec[a] = ax_dist * c["u_axis"]
        axial_plan_norm[a] = abs(ax_dist)
        skip_approach[a] = axial_plan_norm[a] <= ZERO_DELTA_M

    arm_rep: dict[str, dict[str, Any]] = {}
    for a in ARMS:
        arm_rep[a] = {
            "status": "rejected", "aligned": False, "reached": False,
            "table_contact_stalled": False,
            "target_xyz": [round(float(v), 5) for v in cfgs[a]["t_xyz"]],
            "desired_tcp_xyz": [round(float(v), 5) for v in desired_tcp[a]],
            "final_xyz": [0.0, 0.0, 0.0],
            "final_angles": {"pitch": math.nan, "yaw": math.nan, "roll": math.nan},
            "transverse_error_before_m": round(transverse_err_before[a], 5),
            "transverse_error_after_m": round(transverse_err_before[a], 5),
            "opening_before": round(opening_before[a], 4),
            "opening_after_preopen": None,
            "closed": False, "closed_opening": None, "opening_after_lift": None,
            "retained_closure": False, "holding": False,
            "opening": round(opening_before[a], 3), "width_m": None, "lift_status": None,
        }

    result: dict[str, Any] = {
        "arm": "both", "status": "rejected", "aligned": False, "reached": False,
        "accepted": False, "preview": preview, "stalled": False,
        "table_contact_stalled": False, "stop_reason": None,
        "target_xyz": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, -1.0],
        "standoff_m": round(standoff_f, 5), "overdrive_m": round(overdrive_f, 5),
        "contact_table": any_contact,
        "transverse_error_before_m": round(max(transverse_err_before.values()), 5),
        "transverse_error_after_m": round(max(transverse_err_before.values()), 5),
        "opening_before": round(min(opening_before.values()), 4),
        "opening_after_preopen": None, "closed": False, "closed_opening": None,
        "opening_after_lift": None, "retained_closure": False, "lift_status": None,
        "lift_step_m": None, "preopen_steps_used": 0, "align_steps_used": 0,
        "approach_steps_used": 0, "close_steps_used": 0, "lift_steps_used": 0,
        "steps_used": 0, "planned_steps": 0, "ik_error": None, "note": None,
        "recoverable": True,
    }

    def finish(state_now: dict[str, Any]) -> dict[str, Any]:
        for a in ARMS:
            summary = motion._pose_summary(a, motion._joints(state_now, a))
            arm_rep[a]["final_xyz"] = summary["xyz"]
            arm_rep[a]["final_angles"] = {k: summary[k] for k in ("pitch", "yaw", "roll")}
            arm_rep[a].update(_gripper_state(primitives, state_now, a))
        result["arms"] = {"left": arm_rep["left"], "right": arm_rep["right"]}
        result["aligned"] = all(arm_rep[a]["aligned"] for a in ARMS)
        result["table_contact_stalled"] = any(arm_rep[a]["table_contact_stalled"] for a in ARMS)
        result["closed"] = all(arm_rep[a]["closed"] for a in ARMS)
        result["retained_closure"] = all(arm_rep[a]["retained_closure"] for a in ARMS)
        result["holding"] = all(arm_rep[a]["holding"] for a in ARMS)
        result["opening"] = round(min(arm_rep[a]["opening"] for a in ARMS), 3)
        pre_ops = [
            arm_rep[a]["opening_after_preopen"]
            for a in ARMS if arm_rep[a]["opening_after_preopen"] is not None
        ]
        result["opening_after_preopen"] = min(pre_ops) if len(pre_ops) == 2 else None
        cl_ops = [
            arm_rep[a]["closed_opening"]
            for a in ARMS if arm_rep[a]["closed_opening"] is not None
        ]
        result["closed_opening"] = min(cl_ops) if len(cl_ops) == 2 else None
        lift_ops = [
            arm_rep[a]["opening_after_lift"]
            for a in ARMS if arm_rep[a]["opening_after_lift"] is not None
        ]
        result["opening_after_lift"] = min(lift_ops) if len(lift_ops) == 2 else None
        widths = [
            arm_rep[a]["width_m"]
            for a in ARMS if arm_rep[a]["holding"] and arm_rep[a]["width_m"] is not None
        ]
        result["width_m"] = round(min(widths), 4) if widths else None
        result["transverse_error_after_m"] = round(
            max(arm_rep[a]["transverse_error_after_m"] for a in ARMS), 5
        )
        result["final_xyz"] = [0.0, 0.0, 0.0]
        result["final_angles"] = {"pitch": math.nan, "yaw": math.nan, "roll": math.nan}
        result["remaining_episode_steps"] = _budget(primitives)[1]
        return result

    for a in ARMS:
        c = cfgs[a]
        if not skip_align[a] and transverse_err_before[a] > max_lin_f + 1e-9:
            result.update(
                stop_reason="delta_exceeds_limit",
                lift_status="rejected" if any_lift else None,
                note=f"{a}: transverse delta {transverse_err_before[a]:.4f} m exceeds "
                     f"max_linear_m {max_lin_f:.4f} m; re-stage closer before delta_grasp",
            )
            return finish(state0)
        if not skip_approach[a] and axial_plan_norm[a] > max_lin_f + 1e-9:
            result.update(
                stop_reason="delta_exceeds_limit",
                lift_status="rejected" if any_lift else None,
                note=f"{a}: axial approach {axial_plan_norm[a]:.4f} m exceeds "
                     f"max_linear_m {max_lin_f:.4f} m; re-stage closer before delta_grasp",
            )
            return finish(state0)
        if c["lift_vec"] is not None and c["lift_norm"] > max_lin_f + 1e-9:
            result.update(
                stop_reason="delta_exceeds_limit", lift_status="rejected",
                note=f"{a}: lift_dxyz {c['lift_norm']:.4f} m exceeds max_linear_m {max_lin_f:.4f} m",
            )
            return finish(state0)

    move_opening = {
        a: cfgs[a]["opening_f"] if cfgs[a]["opening_f"] is not None else opening_before[a]
        for a in ARMS
    }
    lift_opening = {a: min(move_opening[a], 0.1) for a in ARMS}
    stage_step_m = FINE_STEP_M if any_contact else LINE_STEP_M
    stage_step_deg = LINE_STEP_DEG * stage_step_m / LINE_STEP_M

    # 0-step preflight: Stage 2 (align)
    align_ways: dict[str, list[list[float]]] = {}
    q_after_align: dict[str, list[float]] = {}
    for a in ARMS:
        if skip_align[a]:
            align_ways[a], q_after_align[a] = [], q0[a]
            continue
        _, ways_a, why_a = _plan_arm(
            a, q0[a], _spec({"dxyz": transverse_delta[a].tolist()}, a), "world",
            stage_step_m, stage_step_deg, allow_table_contact=cfgs[a]["contact_table"],
            other_q=None, opening=move_opening[a], other_opening=move_opening[_other_arm(a)],
        )
        if why_a is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"{a} align: {why_a}",
                lift_status="rejected" if any_lift else None,
            )
            return finish(state0)
        align_ways[a] = ways_a
        q_after_align[a] = ways_a[-1] if ways_a else q0[a]
    align_pairs = _sync_pairs(q0["left"], q0["right"], align_ways["left"], align_ways["right"])
    if align_pairs:
        inter_why = _check_inter_arm_trajectory(
            q0["left"], q0["right"], align_pairs,
            left_opening=move_opening["left"], right_opening=move_opening["right"],
        )
        if inter_why is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"both align: {inter_why}",
                lift_status="rejected" if any_lift else None,
            )
            return finish(state0)

    # 0-step preflight: Stage 3 (approach)
    approach_ways: dict[str, list[list[float]]] = {}
    q_after_approach: dict[str, list[float]] = {}
    for a in ARMS:
        if skip_approach[a]:
            approach_ways[a], q_after_approach[a] = [], q_after_align[a]
            continue
        _, ways_a, why_a = _plan_arm(
            a, q_after_align[a], _spec({"dxyz": axial_plan_vec[a].tolist()}, a), "world",
            stage_step_m, stage_step_deg, allow_table_contact=cfgs[a]["contact_table"],
            other_q=None, opening=move_opening[a], other_opening=move_opening[_other_arm(a)],
        )
        if why_a is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"{a} approach: {why_a}",
                lift_status="rejected" if any_lift else None,
            )
            return finish(state0)
        approach_ways[a] = ways_a
        q_after_approach[a] = ways_a[-1] if ways_a else q_after_align[a]
    approach_pairs = _sync_pairs(
        q_after_align["left"], q_after_align["right"],
        approach_ways["left"], approach_ways["right"],
    )
    if approach_pairs:
        inter_why = _check_inter_arm_trajectory(
            q_after_align["left"], q_after_align["right"], approach_pairs,
            left_opening=move_opening["left"], right_opening=move_opening["right"],
        )
        if inter_why is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"both approach: {inter_why}",
                lift_status="rejected" if any_lift else None,
            )
            return finish(state0)

    # 0-step preflight: Stage 5 (lift)
    lift_ways: dict[str, list[list[float]]] = {}
    for a in ARMS:
        if cfgs[a]["lift_vec"] is None:
            lift_ways[a] = []
            continue
        _, ways_a, why_a = _plan_arm(
            a, q_after_approach[a], _spec({"dxyz": cfgs[a]["lift_vec"].tolist()}, a), "world",
            stage_step_m, stage_step_deg, allow_table_contact=cfgs[a]["contact_table"],
            other_q=None, opening=lift_opening[a], other_opening=lift_opening[_other_arm(a)],
        )
        if why_a is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"{a} lift: {why_a}",
                lift_status="rejected",
            )
            return finish(state0)
        lift_ways[a] = ways_a
    lift_pairs = _sync_pairs(
        q_after_approach["left"], q_after_approach["right"],
        lift_ways["left"], lift_ways["right"],
    )
    if lift_pairs:
        inter_why = _check_inter_arm_trajectory(
            q_after_approach["left"], q_after_approach["right"], lift_pairs,
            left_opening=lift_opening["left"], right_opening=lift_opening["right"],
        )
        if inter_why is not None:
            result.update(
                stop_reason="ik_path_rejected", ik_error=f"both lift: {inter_why}",
                lift_status="rejected",
            )
            return finish(state0)

    est_preopen = 2 if any(needs_preopen.values()) else 0
    planned_steps = (
        est_preopen
        + max(len(align_ways[a]) for a in ARMS)
        + max(len(approach_ways[a]) for a in ARMS)
        + 2
        + max(len(lift_ways[a]) for a in ARMS)
    )
    result["planned_steps"] = planned_steps
    _, remaining = _budget(primitives)
    allowed = min(max_steps, remaining - reserve_steps)
    if allowed <= 0:
        result.update(
            stop_reason="episode_native_step_budget" if remaining <= 0 else "reserve_steps_exhausted",
            lift_status="rejected" if any_lift else None,
        )
        return finish(state0)
    if preview:
        if planned_steps > allowed:
            result.update(
                status="rejected", accepted=False, stop_reason="step_budget",
                lift_status="rejected" if any_lift else None,
                note=f"planned {planned_steps} steps exceeds allowance {allowed}",
            )
            return finish(state0)
        for a in ARMS:
            arm_rep[a].update(
                status="planned",
                aligned=bool(skip_align[a] and transverse_err_before[a] <= cfgs[a]["tol_f"]),
                reached=False,
                lift_status="planned" if cfgs[a]["lift_vec"] is not None else None,
            )
        result.update(
            status="planned", accepted=True, reached=False,
            lift_status="planned" if any_lift else None,
        )
        return finish(state0)

    result["accepted"] = True
    steps_used = 0
    state_cur = state0
    status_cur: dict[str, Any] = {}

    def budget_left() -> int:
        _, ep_rem = _budget(primitives)
        return min(max_steps - steps_used, ep_rem - reserve_steps)

    # Stage 1: synchronized pre-open
    preopen_cmds = {
        a: cfgs[a]["opening_f"]
        for a in ARMS if needs_preopen[a] and cfgs[a]["opening_f"] is not None
    }
    if preopen_cmds:
        left_b = budget_left()
        if left_b <= 0:
            result.update(
                steps_used=steps_used, recoverable=True, status="rejected",
                stop_reason="step_budget", lift_status="skipped" if any_lift else None,
            )
            return finish(state_cur)
        state_cur, status_cur, pre_steps, open_after = _grip_to_many(
            primitives, preopen_cmds, left_b
        )
        steps_used += pre_steps
        result["preopen_steps_used"] = pre_steps
        for a in ARMS:
            arm_rep[a]["opening_after_preopen"] = round(open_after[a], 4)
        if motion._spent(status_cur) or budget_left() <= 0:
            result.update(
                steps_used=steps_used, recoverable=(steps_used == 0),
                status="partial" if steps_used > 0 else "rejected",
                stop_reason=motion._episode_stop_reason(status_cur)
                if motion._spent(status_cur) else "step_budget",
                lift_status="skipped" if any_lift else None,
            )
            return finish(state_cur)
    else:
        for a in ARMS:
            arm_rep[a]["opening_after_preopen"] = round(opening_before[a], 4)

    # Stage 2: synchronized transverse align (independent_stall=False)
    perp_cur: dict[str, np.ndarray] = {}
    need_align_arm: dict[str, bool] = {}
    for a in ARMS:
        tcp_c = motion._pose_of(a, motion._joints(state_cur, a))[:3, 3]
        diff_c = cfgs[a]["t_xyz"] - tcp_c
        perp_c = diff_c - float(np.dot(diff_c, cfgs[a]["u_axis"])) * cfgs[a]["u_axis"]
        perp_cur[a] = perp_c
        need_align_arm[a] = float(np.linalg.norm(perp_c)) > ZERO_DELTA_M

    if not any(need_align_arm.values()):
        align_steps, align_stalled, align_stop_reason = 0, False, None
    else:
        left_b = budget_left()
        if left_b <= 0:
            result.update(
                steps_used=steps_used, recoverable=(steps_used == 0),
                status="partial" if steps_used > 0 else "rejected",
                stop_reason="step_budget", lift_status="skipped" if any_lift else None,
            )
            return finish(state_cur)
        align_res = nudge(
            primitives, start_joints, "both",
            left={"dxyz": perp_cur["left"].tolist() if need_align_arm["left"] else [0.0, 0.0, 0.0]},
            right={"dxyz": perp_cur["right"].tolist() if need_align_arm["right"] else [0.0, 0.0, 0.0]},
            frame="world", compact=True, stop_on_stall=True, independent_stall=False,
            stall_steps=stall_steps, max_linear_m=max_lin_f, max_steps=left_b,
            reserve_steps=reserve_steps, step_m=stage_step_m,
            tol_m=min(TOL_M, min(cfgs[a]["tol_f"] for a in ARMS)),
            allow_table_contact=any_contact, _arm_allow_table_contact=per_arm_table,
        )
        if "error" in align_res:
            return {"error": f"delta_grasp: {align_res['error']}"}
        align_steps = int(align_res.get("steps_used", 0))
        steps_used += align_steps
        align_stalled = bool(align_res.get("stalled"))
        align_stop_reason = align_res.get("stop_reason")
        if align_res.get("ik_error"):
            result["ik_error"] = align_res["ik_error"]
        state_cur = motion._obs(primitives)["state"]

    tcp_after_align: dict[str, np.ndarray] = {}
    for a in ARMS:
        tcp_a = motion._pose_of(a, motion._joints(state_cur, a))[:3, 3]
        tcp_after_align[a] = tcp_a
        rem = cfgs[a]["t_xyz"] - tcp_a
        rem_p = rem - float(np.dot(rem, cfgs[a]["u_axis"])) * cfgs[a]["u_axis"]
        err_a = float(np.linalg.norm(rem_p))
        arm_rep[a]["aligned"] = bool(err_a <= cfgs[a]["tol_f"])
        arm_rep[a]["transverse_error_after_m"] = round(err_a, 5)
    result["align_steps_used"] = align_steps
    if not all(arm_rep[a]["aligned"] for a in ARMS):
        for a in ARMS:
            arm_rep[a]["lift_status"] = "skipped" if cfgs[a]["lift_vec"] is not None else None
            arm_rep[a]["status"] = (
                "completed" if arm_rep[a]["aligned"]
                else ("stopped" if align_stalled else ("partial" if steps_used > 0 else "rejected"))
            )
        result.update(
            steps_used=steps_used, stalled=align_stalled, recoverable=(steps_used == 0),
            reached=False, lift_status="skipped" if any_lift else None,
            status="stopped" if align_stalled else ("partial" if steps_used > 0 else "rejected"),
            stop_reason="contact_or_stall" if align_stalled
            else (align_stop_reason or "alignment_tolerance_not_met"),
            note="transverse alignment did not meet align_tol_m; aborted before closing",
        )
        return finish(state_cur)

    # Stage 3: synchronized guarded axial approach (independent_stall=True)
    axial_rem_vec: dict[str, np.ndarray] = {}
    need_app_arm: dict[str, bool] = {}
    for a in ARMS:
        dist_a = float(np.dot(desired_tcp[a] - tcp_after_align[a], cfgs[a]["u_axis"]))
        axial_rem_vec[a] = dist_a * cfgs[a]["u_axis"]
        need_app_arm[a] = abs(dist_a) > ZERO_DELTA_M

    if not any(need_app_arm.values()):
        approach_steps = 0
        app_stop_reason: str | None = None
        app_arms_info: dict[str, Any] = {
            a: {"reached": True, "stalled": False} for a in ARMS
        }
    else:
        left_b = budget_left()
        if left_b <= 0:
            result.update(
                steps_used=steps_used, recoverable=(steps_used == 0),
                status="partial" if steps_used > 0 else "rejected",
                stop_reason="step_budget", lift_status="skipped" if any_lift else None,
            )
            return finish(state_cur)
        app_res = nudge(
            primitives, start_joints, "both",
            left={"dxyz": axial_rem_vec["left"].tolist() if need_app_arm["left"] else [0.0, 0.0, 0.0]},
            right={"dxyz": axial_rem_vec["right"].tolist() if need_app_arm["right"] else [0.0, 0.0, 0.0]},
            frame="world", compact=True, stop_on_stall=True, independent_stall=True,
            stall_steps=stall_steps, max_linear_m=max_lin_f, max_steps=left_b,
            reserve_steps=reserve_steps, step_m=stage_step_m,
            allow_table_contact=any_contact, _arm_allow_table_contact=per_arm_table,
        )
        if "error" in app_res:
            return {"error": f"delta_grasp: {app_res['error']}"}
        approach_steps = int(app_res.get("steps_used", 0))
        steps_used += approach_steps
        app_stop_reason = app_res.get("stop_reason")
        app_arms_info = app_res.get("arms") or {}
        if app_res.get("ik_error"):
            result["ik_error"] = app_res["ik_error"]
        state_cur = motion._obs(primitives)["state"]

    result["approach_steps_used"] = approach_steps
    seated_ok: dict[str, bool] = {}
    unplanned_stall: dict[str, bool] = {}
    for a in ARMS:
        c = cfgs[a]
        a_info = app_arms_info.get(a) or {}
        a_reached = True if not need_app_arm[a] else bool(a_info.get("reached"))
        a_stalled = False if not need_app_arm[a] else bool(a_info.get("stalled"))
        end_seated = False
        end_table_seated = False
        if (
            not a_reached
            and not a_stalled
            and app_stop_reason in (None, "final_target_not_reached", "contact_or_stall")
        ):
            q_app = motion._joints(state_cur, a)
            fr_app = kin.frames(a, q_app)
            tcp_app = motion._pose_of(a, q_app)[:3, 3]
            tip_z = float(
                (fr_app[-1][:3, 3] + fr_app[-1][:3, 0] * max(kin.TCP_OFFSET, kin.TIP_OFFSET))[2]
            )
            rem_app = desired_tcp[a] - tcp_app
            ax_err = abs(float(np.dot(rem_app, c["u_axis"])))
            perp_err = float(
                np.linalg.norm(rem_app - float(np.dot(rem_app, c["u_axis"])) * c["u_axis"])
            )
            end_seated = (
                ax_err <= max(0.012, c["overdrive_f"] + 0.010)
                and perp_err <= c["tol_f"] + 0.002
            )
            end_table_seated = (
                c["contact_table"]
                and c["u_axis"][2] < -0.3
                and tip_z <= kin.TABLE_Z + 0.020
                and end_seated
            )
        tcs = bool(c["contact_table"] and (a_stalled or end_table_seated))
        arm_rep[a]["table_contact_stalled"] = tcs
        seated_ok[a] = bool(a_reached or tcs or end_seated)
        unplanned_stall[a] = bool(a_stalled and not tcs)

    if not all(seated_ok.values()):
        any_unplanned = any(unplanned_stall.values())
        for a in ARMS:
            arm_rep[a]["lift_status"] = "skipped" if cfgs[a]["lift_vec"] is not None else None
            arm_rep[a]["status"] = (
                "completed" if seated_ok[a]
                else ("stopped" if unplanned_stall[a] else ("partial" if steps_used > 0 else "rejected"))
            )
        result.update(
            steps_used=steps_used, stalled=any_unplanned, recoverable=(steps_used == 0),
            reached=False, lift_status="skipped" if any_lift else None,
            status="stopped" if any_unplanned else ("partial" if steps_used > 0 else "rejected"),
            stop_reason="contact_or_stall" if any_unplanned
            else (app_stop_reason or "approach_not_reached"),
            note="axial approach did not reach target; aborted before closing",
        )
        return finish(state_cur)

    # Stage 4: synchronized close
    left_b = budget_left()
    ep_status, _ = _budget(primitives)
    if left_b <= 0 or motion._spent(ep_status):
        result.update(
            steps_used=steps_used, recoverable=(steps_used == 0), reached=False,
            lift_status="skipped" if any_lift else None,
            status="partial" if steps_used > 0 else "rejected",
            stop_reason=motion._episode_stop_reason(ep_status)
            if motion._spent(ep_status) else "step_budget",
            note="step budget exhausted before closing gripper",
        )
        return finish(state_cur)

    state_cur, status_cur, close_steps, closed_openings = _grip_to_many(
        primitives, {"left": 0.0, "right": 0.0}, left_b
    )
    steps_used += close_steps
    result["close_steps_used"] = close_steps
    for a in ARMS:
        arm_rep[a]["closed"] = close_steps > 0
        arm_rep[a]["closed_opening"] = round(closed_openings[a], 4)

    if not any_lift:
        for a in ARMS:
            arm_rep[a]["retained_closure"] = bool(motion.retained_closure(0.0, closed_openings[a]))
            arm_rep[a]["reached"] = close_steps > 0
            arm_rep[a]["status"] = (
                "completed" if close_steps > 0 else ("partial" if steps_used > 0 else "rejected")
            )
        result.update(steps_used=steps_used, recoverable=(steps_used == 0), stalled=False)
        if close_steps > 0:
            result.update(status="completed", reached=True, stop_reason=None)
            if any(arm_rep[a]["table_contact_stalled"] for a in ARMS):
                result["note"] = "tabletop contact seated fingertips before closing"
        else:
            result.update(
                status="partial" if steps_used > 0 else "rejected", reached=False,
                stop_reason=motion._episode_stop_reason(status_cur) or "step_budget",
            )
        return finish(state_cur)

    # Stage 5: synchronized guarded liftoff (independent_stall=True)
    left_b = budget_left()
    if left_b <= 0 or motion._spent(status_cur):
        for a in ARMS:
            arm_rep[a]["retained_closure"] = bool(motion.retained_closure(0.0, closed_openings[a]))
            arm_rep[a]["lift_status"] = "rejected" if cfgs[a]["lift_vec"] is not None else None
            arm_rep[a]["status"] = "partial"
        result.update(
            steps_used=steps_used, recoverable=(steps_used == 0), reached=False,
            lift_status="rejected", status="partial",
            stop_reason=motion._episode_stop_reason(status_cur)
            if motion._spent(status_cur) else "step_budget",
            note="step budget exhausted before post-close liftoff",
        )
        return finish(state_cur)

    lift_step_m = (
        FINE_STEP_M
        if any(cfgs[a]["contact_table"] or closed_openings[a] <= THIN_HELD_OPENING for a in ARMS)
        else LINE_STEP_M
    )
    result["lift_step_m"] = lift_step_m
    lift_res = nudge(
        primitives, start_joints, "both",
        left={"dxyz": cfgs["left"]["lift_vec"].tolist() if cfgs["left"]["lift_vec"] is not None else [0.0, 0.0, 0.0]},
        right={"dxyz": cfgs["right"]["lift_vec"].tolist() if cfgs["right"]["lift_vec"] is not None else [0.0, 0.0, 0.0]},
        frame="world", compact=True, stop_on_stall=True, independent_stall=True,
        stall_steps=stall_steps, max_linear_m=max_lin_f, max_steps=left_b,
        reserve_steps=reserve_steps, step_m=lift_step_m,
        allow_table_contact=any_contact, _arm_allow_table_contact=per_arm_table,
    )
    if "error" in lift_res:
        return {"error": f"delta_grasp: {lift_res['error']}"}
    lift_steps = int(lift_res.get("steps_used", 0))
    steps_used += lift_steps
    state_end = motion._obs(primitives)["state"]
    lift_arms_info = lift_res.get("arms") or {}
    arm_lift_stalled: dict[str, bool] = {}
    for a in ARMS:
        op_after = motion._opening(state_end, a)
        arm_rep[a]["opening_after_lift"] = round(op_after, 4)
        arm_rep[a]["retained_closure"] = bool(
            motion.retained_closure(0.0, op_after, closed_openings[a])
        )
        if cfgs[a]["lift_vec"] is None:
            arm_rep[a].update(lift_status=None, reached=True, status="completed")
            arm_lift_stalled[a] = False
        else:
            a_info = lift_arms_info.get(a) or {}
            a_reached = bool(a_info.get("reached"))
            a_stalled = bool(a_info.get("stalled"))
            arm_lift_stalled[a] = a_stalled
            a_status = (
                "completed" if a_reached
                else ("stopped" if a_stalled else str(lift_res.get("status") or "partial"))
            )
            arm_rep[a].update(lift_status=a_status, reached=a_reached, status=a_status)

    all_reached = all(arm_rep[a]["reached"] for a in ARMS)
    any_lift_stalled = any(arm_lift_stalled.values())
    top_lift_status = (
        "completed" if all_reached
        else ("stopped" if any_lift_stalled else str(lift_res.get("status") or "partial"))
    )
    result.update(
        steps_used=steps_used, lift_steps_used=lift_steps, lift_status=top_lift_status,
        stalled=any_lift_stalled, recoverable=(steps_used == 0),
    )
    if lift_res.get("ik_error"):
        result["ik_error"] = lift_res["ik_error"]
    if all_reached:
        result.update(status="completed", reached=True, stop_reason=None)
        if any(arm_rep[a]["table_contact_stalled"] for a in ARMS):
            result["note"] = "tabletop contact seated fingertips; closed and lifted cleanly"
    elif any_lift_stalled:
        result.update(
            status="stopped", reached=False, stop_reason="contact_or_stall",
            note=lift_res.get("note") or "post-close liftoff stopped on contact or stall",
        )
    else:
        result.update(
            status="partial" if steps_used > 0 else "rejected", reached=False,
            stop_reason=lift_res.get("stop_reason") or "lift_not_reached",
        )
        if lift_res.get("note"):
            result["note"] = lift_res["note"]
    return finish(state_end)


def delta_grasp(
    primitives: Any,
    start_joints: dict[str, list[float]],
    target_xyz: Sequence[float] | None = None,
    arm: str = "right",
    *,
    axis: Sequence[float] = (0.0, 0.0, -1.0),
    standoff_m: float = 0.0,
    overdrive_m: float = 0.0,
    opening: float | None = None,
    contact_table: bool = False,
    lift_dxyz: Sequence[float] | None = None,
    left: Mapping[str, Any] | Sequence[float] | None = None,
    right: Mapping[str, Any] | Sequence[float] | None = None,
    align_tol_m: float = ALIGN_TOL_M,
    max_linear_m: float = MAX_LINEAR_M,
    max_steps: int = 36,
    reserve_steps: int = 20,
    stall_steps: int = 2,
    preview: bool = False,
) -> dict[str, Any]:
    """Compact last-3cm grasp / handoff: pre-open -> align -> approach/table-touch -> close -> lift.

    Stages executed sequentially once preflight passes (0 steps on refusal):
    1. Optional pre-open to ``opening`` in ``(0, 1]`` (skipped in 0 steps if already within 0.05).
    2. Transverse alignment perpendicular to ``axis`` via guarded compact ``nudge`` (skipped in
       0 steps if <= 0.5 mm; aborts before closing if residual transverse error > ``align_tol_m``).
    3. Axial approach along ``+axis`` to ``target_xyz - axis * standoff_m + axis * overdrive_m``
       (clamped to the safe fingertip table-touch floor when ``contact_table=True`` and
       ``target_xyz`` is at/above the tabletop). On ``contact_table=True``, a stall against the
       tabletop is expected seating, not an abort.
    4. Gripper close (``command=0.0``), recording ``closed_opening``, ``holding``, ``width_m``.
    5. Optional guarded ``lift_dxyz`` liftoff, automatically using 3 mm micro-steps when
       ``closed_opening <= 0.045`` to avoid breaking thin-part static friction.
    With ``arm="both"``, pass ``left`` and ``right`` targets/dicts to execute all five stages
    synchronously on one clock with independent tabletop seating per arm.
    """

    try:
        if arm not in (*ARMS, "both"):
            raise ValueError("arm must be 'left', 'right' or 'both'")
        axis_raw = _vec3(axis, "axis")
        axis_norm = float(np.linalg.norm(axis_raw))
        if axis_norm <= 1e-9:
            raise ValueError("axis must be non-zero")
        u_axis = axis_raw / axis_norm
        standoff_f = _finite(standoff_m, "standoff_m")
        if standoff_f < 0.0:
            raise ValueError("standoff_m must be >= 0")
        overdrive_f = _finite(overdrive_m, "overdrive_m")
        if overdrive_f < 0.0:
            raise ValueError("overdrive_m must be >= 0")
        opening_f: float | None = None
        if opening is not None:
            opening_f = _finite(opening, "opening")
            if not 0.0 < opening_f <= 1.0:
                raise ValueError("opening must be in (0, 1]")
        if not isinstance(contact_table, bool):
            raise ValueError("contact_table must be a boolean")
        lift_vec: np.ndarray | None = None
        lift_norm = 0.0
        if lift_dxyz is not None:
            lift_vec = _vec3(lift_dxyz, "lift_dxyz")
            lift_norm = float(np.linalg.norm(lift_vec))
            if lift_norm <= 1e-6:
                raise ValueError("lift_dxyz must be non-zero")
        tol_f = _finite(align_tol_m, "align_tol_m")
        if not 0.0005 <= tol_f <= 0.02:
            raise ValueError("align_tol_m must be between 0.0005 and 0.02 m")
        max_lin_f = _finite(max_linear_m, "max_linear_m")
        if not 0.0 < max_lin_f <= HARD_LINEAR_M:
            raise ValueError(f"max_linear_m must be in (0, {HARD_LINEAR_M}]")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(reserve_steps) is not int or reserve_steps < 0:
            raise ValueError("reserve_steps must be a non-negative integer")
        if type(stall_steps) is not int or not 1 <= stall_steps <= 5:
            raise ValueError("stall_steps must be an integer between 1 and 5")
        if not isinstance(preview, bool):
            raise ValueError("preview must be a boolean")
        if arm == "both":
            if target_xyz is not None:
                raise ValueError("target_xyz must be None when arm='both'")
        else:
            if left is not None or right is not None:
                raise ValueError("left= / right= go with arm='both'")
            if target_xyz is None:
                raise ValueError("target_xyz is required for single-arm delta_grasp")
            t_xyz = _vec3(target_xyz, "target_xyz")
    except ValueError as exc:
        return {"error": f"delta_grasp: {exc}"}

    if arm == "both":
        return _delta_grasp_both(
            primitives, start_joints, left=left, right=right, u_axis_default=u_axis,
            standoff_f=standoff_f, overdrive_f=overdrive_f, opening_f=opening_f,
            contact_table=contact_table, lift_dxyz=lift_dxyz, tol_f=tol_f,
            max_lin_f=max_lin_f, max_steps=max_steps, reserve_steps=reserve_steps,
            stall_steps=stall_steps, preview=preview,
        )

    state0 = motion._obs(primitives)["state"]
    q0 = motion._joints(state0, arm)
    pose0 = motion._pose_of(arm, q0)
    tcp_before = pose0[:3, 3]
    opening_before = motion._opening(state0, arm)
    last_grip_cmd = (getattr(primitives, "_grip_commands", None) or {}).get(arm)
    needs_preopen = bool(
        opening_f is not None
        and (
            abs(opening_before - opening_f) > PREOPEN_TOL
            or (last_grip_cmd is not None and last_grip_cmd <= 0.05)
        )
    )

    raw_diff = t_xyz - tcp_before
    axial_gap_m = float(np.dot(raw_diff, u_axis))
    transverse_delta = raw_diff - axial_gap_m * u_axis
    transverse_error_before_m = float(np.linalg.norm(transverse_delta))
    skip_align = transverse_error_before_m <= ZERO_DELTA_M

    desired_tcp = t_xyz - u_axis * standoff_f + u_axis * overdrive_f
    if contact_table and u_axis[2] < -1e-6 and t_xyz[2] >= kin.TABLE_Z - 0.005:
        tip_delta_z = float(pose0[2, 0]) * (kin.TIP_OFFSET - kin.TCP_OFFSET)
        min_safe_tcp_z = kin.TABLE_Z - TABLE_TOUCH_OVERDRIVE_M + 0.0005 - tip_delta_z
        if desired_tcp[2] < min_safe_tcp_z:
            pullback = (min_safe_tcp_z - float(desired_tcp[2])) / (-float(u_axis[2]))
            desired_tcp = desired_tcp - pullback * u_axis

    tcp_after_align_plan = tcp_before + (np.zeros(3) if skip_align else transverse_delta)
    axial_plan_dist = float(np.dot(desired_tcp - tcp_after_align_plan, u_axis))
    axial_plan_vec = axial_plan_dist * u_axis
    axial_plan_norm = abs(axial_plan_dist)
    skip_approach = axial_plan_norm <= ZERO_DELTA_M

    result: dict[str, Any] = {
        "arm": arm,
        "status": "rejected",
        "aligned": False,
        "reached": False,
        "accepted": False,
        "preview": preview,
        "stalled": False,
        "table_contact_stalled": False,
        "stop_reason": None,
        "target_xyz": [round(float(v), 5) for v in t_xyz],
        "desired_tcp_xyz": [round(float(v), 5) for v in desired_tcp],
        "axis": [round(float(v), 6) for v in u_axis],
        "standoff_m": round(standoff_f, 5),
        "overdrive_m": round(overdrive_f, 5),
        "contact_table": contact_table,
        "transverse_error_before_m": round(transverse_error_before_m, 5),
        "transverse_error_after_m": round(transverse_error_before_m, 5),
        "opening_before": round(opening_before, 4),
        "opening_after_preopen": None,
        "closed": False,
        "closed_opening": None,
        "opening_after_lift": None,
        "retained_closure": False,
        "lift_status": None,
        "lift_step_m": None,
        "preopen_steps_used": 0,
        "align_steps_used": 0,
        "approach_steps_used": 0,
        "close_steps_used": 0,
        "lift_steps_used": 0,
        "steps_used": 0,
        "planned_steps": 0,
        "ik_error": None,
        "note": None,
        "recoverable": True,
    }

    def finish(state_now: dict[str, Any]) -> dict[str, Any]:
        q_now = motion._joints(state_now, arm)
        summary = motion._pose_summary(arm, q_now)
        result["final_xyz"] = summary["xyz"]
        result["final_angles"] = {k: summary[k] for k in ("pitch", "yaw", "roll")}
        result.update(_gripper_state(primitives, state_now, arm))
        result["remaining_episode_steps"] = _budget(primitives)[1]
        result["arms"] = {
            arm: {
                k: result[k]
                for k in (
                    "status", "aligned", "reached", "table_contact_stalled",
                    "target_xyz", "desired_tcp_xyz", "final_xyz", "final_angles",
                    "transverse_error_before_m", "transverse_error_after_m",
                    "opening_before", "opening_after_preopen", "closed",
                    "closed_opening", "opening_after_lift", "retained_closure",
                    "holding", "opening", "width_m", "lift_status",
                )
                if k in result
            }
        }
        return result

    if not skip_align and transverse_error_before_m > max_lin_f + 1e-9:
        result.update(
            stop_reason="delta_exceeds_limit",
            lift_status="rejected" if lift_vec is not None else None,
            note=f"{arm}: transverse delta {transverse_error_before_m:.4f} m exceeds "
                 f"max_linear_m {max_lin_f:.4f} m; re-stage closer before delta_grasp",
        )
        return finish(state0)

    if not skip_approach and axial_plan_norm > max_lin_f + 1e-9:
        result.update(
            stop_reason="delta_exceeds_limit",
            lift_status="rejected" if lift_vec is not None else None,
            note=f"{arm}: axial approach {axial_plan_norm:.4f} m exceeds "
                 f"max_linear_m {max_lin_f:.4f} m; re-stage closer before delta_grasp",
        )
        return finish(state0)

    if lift_vec is not None and lift_norm > max_lin_f + 1e-9:
        result.update(
            stop_reason="delta_exceeds_limit",
            lift_status="rejected",
            note=f"{arm}: lift_dxyz {lift_norm:.4f} m exceeds max_linear_m {max_lin_f:.4f} m",
        )
        return finish(state0)

    other_arm = _other_arm(arm)
    other_q0 = _safe_joints(state0, other_arm)
    opening_other0 = _safe_opening(state0, other_arm)
    move_opening = opening_f if opening_f is not None else opening_before
    lift_opening = min(move_opening, 0.1)

    align_step_m = FINE_STEP_M if contact_table else LINE_STEP_M
    align_step_deg = LINE_STEP_DEG * align_step_m / LINE_STEP_M
    if skip_align:
        align_ways: list[list[float]] = []
        q_after_align_plan = q0
    else:
        align_spec = _spec({"dxyz": transverse_delta.tolist()}, arm)
        _, align_ways, align_why = _plan_arm(
            arm, q0, align_spec, "world", align_step_m, align_step_deg,
            allow_table_contact=contact_table,
            other_q=other_q0,
            opening=move_opening,
            other_opening=opening_other0,
        )
        if align_why is not None:
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm} align: {align_why}",
                lift_status="rejected" if lift_vec is not None else None,
            )
            return finish(state0)
        q_after_align_plan = align_ways[-1] if align_ways else q0

    approach_step_m = FINE_STEP_M if contact_table else LINE_STEP_M
    approach_step_deg = LINE_STEP_DEG * approach_step_m / LINE_STEP_M
    if skip_approach:
        approach_ways: list[list[float]] = []
        q_after_approach_plan = q_after_align_plan
    else:
        approach_spec = _spec({"dxyz": axial_plan_vec.tolist()}, arm)
        _, approach_ways, approach_why = _plan_arm(
            arm, q_after_align_plan, approach_spec, "world", approach_step_m, approach_step_deg,
            allow_table_contact=contact_table,
            other_q=other_q0,
            opening=move_opening,
            other_opening=opening_other0,
        )
        if approach_why is not None:
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm} approach: {approach_why}",
                lift_status="rejected" if lift_vec is not None else None,
            )
            return finish(state0)
        q_after_approach_plan = approach_ways[-1] if approach_ways else q_after_align_plan

    if lift_vec is not None:
        lift_plan_step_m = FINE_STEP_M if contact_table else LINE_STEP_M
        lift_plan_step_deg = LINE_STEP_DEG * lift_plan_step_m / LINE_STEP_M
        lift_spec = _spec({"dxyz": lift_vec.tolist()}, arm)
        _, lift_ways, lift_why = _plan_arm(
            arm, q_after_approach_plan, lift_spec, "world", lift_plan_step_m, lift_plan_step_deg,
            allow_table_contact=contact_table,
            other_q=other_q0,
            opening=lift_opening,
            other_opening=opening_other0,
        )
        if lift_why is not None:
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm} lift: {lift_why}",
                lift_status="rejected",
            )
            return finish(state0)
    else:
        lift_ways = []

    est_preopen_steps = 2 if needs_preopen else 0
    est_close_steps = 2
    planned_steps = (
        est_preopen_steps + len(align_ways) + len(approach_ways) + est_close_steps + len(lift_ways)
    )
    result["planned_steps"] = planned_steps
    _, remaining = _budget(primitives)
    allowed = min(max_steps, remaining - reserve_steps)

    if allowed <= 0:
        result.update(
            stop_reason="episode_native_step_budget" if remaining <= 0
            else "reserve_steps_exhausted",
            lift_status="rejected" if lift_vec is not None else None,
        )
        return finish(state0)

    if preview:
        if planned_steps > allowed:
            result.update(
                status="rejected",
                accepted=False,
                stop_reason="step_budget",
                lift_status="rejected" if lift_vec is not None else None,
                note=f"planned {planned_steps} steps exceeds allowance {allowed}",
            )
            return finish(state0)
        result.update(
            status="planned",
            accepted=True,
            aligned=bool(skip_align and transverse_error_before_m <= tol_f),
            reached=False,
            lift_status="planned" if lift_vec is not None else None,
        )
        return finish(state0)

    result["accepted"] = True
    steps_used = 0
    state_cur = state0
    status_cur: dict[str, Any] = {}

    def budget_left() -> int:
        _, ep_rem = _budget(primitives)
        return min(max_steps - steps_used, ep_rem - reserve_steps)

    # Stage 1: optional pre-open
    if needs_preopen and opening_f is not None:
        left = budget_left()
        if left <= 0:
            result.update(
                steps_used=steps_used,
                recoverable=True,
                status="rejected",
                stop_reason="step_budget",
                lift_status="skipped" if lift_vec is not None else None,
            )
            return finish(state_cur)
        state_cur, status_cur, pre_steps, open_after = _grip_to(primitives, arm, opening_f, left)
        steps_used += pre_steps
        result["preopen_steps_used"] = pre_steps
        result["opening_after_preopen"] = round(open_after, 4)
        if motion._spent(status_cur) or budget_left() <= 0:
            result.update(
                steps_used=steps_used,
                recoverable=(steps_used == 0),
                status="partial" if steps_used > 0 else "rejected",
                stop_reason=motion._episode_stop_reason(status_cur) if motion._spent(status_cur)
                else "step_budget",
                lift_status="skipped" if lift_vec is not None else None,
            )
            return finish(state_cur)
    else:
        result["opening_after_preopen"] = round(opening_before, 4)

    # Stage 2: transverse alignment
    tcp_cur = motion._pose_of(arm, motion._joints(state_cur, arm))[:3, 3]
    diff_cur = t_xyz - tcp_cur
    perp_cur = diff_cur - float(np.dot(diff_cur, u_axis)) * u_axis
    perp_norm = float(np.linalg.norm(perp_cur))
    if perp_norm <= ZERO_DELTA_M:
        align_steps = 0
        align_stalled = False
        align_stop_reason: str | None = None
        tcp_after_align = tcp_cur
    else:
        left = budget_left()
        if left <= 0:
            result.update(
                steps_used=steps_used,
                recoverable=(steps_used == 0),
                status="partial" if steps_used > 0 else "rejected",
                stop_reason="step_budget",
                lift_status="skipped" if lift_vec is not None else None,
            )
            return finish(state_cur)
        align_res = nudge(
            primitives, start_joints, arm, dxyz=perp_cur.tolist(), frame="world",
            compact=True, stop_on_stall=True, stall_steps=stall_steps,
            max_linear_m=max_lin_f, max_steps=left, reserve_steps=reserve_steps,
            step_m=align_step_m, tol_m=min(TOL_M, tol_f),
            allow_table_contact=contact_table,
        )
        if "error" in align_res:
            return {"error": f"delta_grasp: {align_res['error']}"}
        align_steps = int(align_res.get("steps_used", 0))
        steps_used += align_steps
        align_stalled = bool(align_res.get("stalled"))
        align_stop_reason = align_res.get("stop_reason")
        if align_res.get("ik_error"):
            result["ik_error"] = align_res["ik_error"]
        state_cur = motion._obs(primitives)["state"]
        tcp_after_align = motion._pose_of(arm, motion._joints(state_cur, arm))[:3, 3]

    rem_diff = t_xyz - tcp_after_align
    rem_perp = rem_diff - float(np.dot(rem_diff, u_axis)) * u_axis
    transverse_error_after = float(np.linalg.norm(rem_perp))
    aligned = bool(transverse_error_after <= tol_f)
    result.update(
        aligned=aligned,
        transverse_error_after_m=round(transverse_error_after, 5),
        align_steps_used=align_steps,
    )
    if not aligned:
        result.update(
            steps_used=steps_used,
            stalled=align_stalled,
            recoverable=(steps_used == 0),
            reached=False,
            lift_status="skipped" if lift_vec is not None else None,
            status="stopped" if align_stalled else ("partial" if steps_used > 0 else "rejected"),
            stop_reason="contact_or_stall" if align_stalled
            else (align_stop_reason or "alignment_tolerance_not_met"),
            note="transverse alignment did not meet align_tol_m; aborted before closing",
        )
        return finish(state_cur)

    # Stage 3: guarded axial approach / table-touch
    axial_rem_dist = float(np.dot(desired_tcp - tcp_after_align, u_axis))
    axial_rem_vec = axial_rem_dist * u_axis
    if abs(axial_rem_dist) <= ZERO_DELTA_M:
        approach_steps = 0
        approach_stalled = False
        approach_reached = True
        approach_stop_reason: str | None = None
    else:
        left = budget_left()
        if left <= 0:
            result.update(
                steps_used=steps_used,
                recoverable=(steps_used == 0),
                status="partial" if steps_used > 0 else "rejected",
                stop_reason="step_budget",
                lift_status="skipped" if lift_vec is not None else None,
            )
            return finish(state_cur)
        app_res = nudge(
            primitives, start_joints, arm, dxyz=axial_rem_vec.tolist(), frame="world",
            compact=True, stop_on_stall=True, stall_steps=stall_steps,
            max_linear_m=max_lin_f, max_steps=left, reserve_steps=reserve_steps,
            step_m=approach_step_m, allow_table_contact=contact_table,
        )
        if "error" in app_res:
            return {"error": f"delta_grasp: {app_res['error']}"}
        approach_steps = int(app_res.get("steps_used", 0))
        steps_used += approach_steps
        approach_stalled = bool(app_res.get("stalled"))
        approach_reached = bool(app_res.get("reached"))
        approach_stop_reason = app_res.get("stop_reason")
        if app_res.get("ik_error"):
            result["ik_error"] = app_res["ik_error"]
        state_cur = motion._obs(primitives)["state"]

    result["approach_steps_used"] = approach_steps
    end_of_stroke_seated = False
    end_of_stroke_table_seated = False
    if (
        not approach_reached
        and not approach_stalled
        and approach_stop_reason == "final_target_not_reached"
    ):
        q_after_app = motion._joints(state_cur, arm)
        fr_after_app = kin.frames(arm, q_after_app)
        tcp_after_app = motion._pose_of(arm, q_after_app)[:3, 3]
        tip_after_z = float(
            (fr_after_app[-1][:3, 3] + fr_after_app[-1][:3, 0] * max(kin.TCP_OFFSET, kin.TIP_OFFSET))[2]
        )
        rem_after_app = desired_tcp - tcp_after_app
        axial_err_after = abs(float(np.dot(rem_after_app, u_axis)))
        perp_err_after = float(np.linalg.norm(rem_after_app - float(np.dot(rem_after_app, u_axis)) * u_axis))
        end_of_stroke_seated = (
            axial_err_after <= max(0.012, overdrive_f + 0.010)
            and perp_err_after <= tol_f + 0.002
        )
        end_of_stroke_table_seated = (
            contact_table
            and u_axis[2] < -0.3
            and tip_after_z <= kin.TABLE_Z + 0.020
            and end_of_stroke_seated
        )
    table_contact_stalled = bool(contact_table and (approach_stalled or end_of_stroke_table_seated))
    result["table_contact_stalled"] = table_contact_stalled

    if not approach_reached and not table_contact_stalled and not end_of_stroke_seated:
        result.update(
            steps_used=steps_used,
            stalled=approach_stalled,
            recoverable=(steps_used == 0),
            reached=False,
            lift_status="skipped" if lift_vec is not None else None,
            status="stopped" if approach_stalled else ("partial" if steps_used > 0 else "rejected"),
            stop_reason="contact_or_stall" if approach_stalled
            else (approach_stop_reason or "approach_not_reached"),
            note="axial approach did not reach target; aborted before closing",
        )
        return finish(state_cur)

    # Stage 4: close gripper
    left = budget_left()
    ep_status, _ = _budget(primitives)
    if left <= 0 or motion._spent(ep_status):
        result.update(
            steps_used=steps_used,
            recoverable=(steps_used == 0),
            reached=False,
            lift_status="skipped" if lift_vec is not None else None,
            status="partial" if steps_used > 0 else "rejected",
            stop_reason=motion._episode_stop_reason(ep_status) if motion._spent(ep_status)
            else "step_budget",
            note="step budget exhausted before closing gripper",
        )
        return finish(state_cur)

    state_cur, status_cur, close_steps, closed_opening = _grip_to(primitives, arm, 0.0, left)
    steps_used += close_steps
    result.update(
        close_steps_used=close_steps,
        closed=close_steps > 0,
        closed_opening=round(closed_opening, 4),
    )

    if lift_vec is None:
        retained = bool(motion.retained_closure(0.0, closed_opening))
        result.update(
            steps_used=steps_used,
            retained_closure=retained,
            recoverable=(steps_used == 0),
            stalled=False,
        )
        if close_steps > 0:
            result.update(status="completed", reached=True, stop_reason=None)
            if table_contact_stalled:
                result["note"] = "tabletop contact seated fingertips before closing"
        else:
            result.update(
                status="partial" if steps_used > 0 else "rejected",
                reached=False,
                stop_reason=motion._episode_stop_reason(status_cur) or "step_budget",
            )
        return finish(state_cur)

    # Stage 5: optional post-close guarded liftoff (3 mm micro-steps for thin grips)
    left = budget_left()
    if left <= 0 or motion._spent(status_cur):
        result.update(
            steps_used=steps_used,
            retained_closure=bool(motion.retained_closure(0.0, closed_opening)),
            recoverable=(steps_used == 0),
            reached=False,
            lift_status="rejected",
            status="partial",
            stop_reason=motion._episode_stop_reason(status_cur) if motion._spent(status_cur)
            else "step_budget",
            note="step budget exhausted before post-close liftoff",
        )
        return finish(state_cur)

    lift_step_m = (
        FINE_STEP_M if (contact_table or closed_opening <= THIN_HELD_OPENING) else LINE_STEP_M
    )
    result["lift_step_m"] = lift_step_m
    lift_res = nudge(
        primitives, start_joints, arm, dxyz=lift_vec.tolist(), frame="world",
        compact=True, stop_on_stall=True, stall_steps=stall_steps,
        max_linear_m=max_lin_f, max_steps=left, reserve_steps=reserve_steps,
        step_m=lift_step_m, allow_table_contact=contact_table,
    )
    if "error" in lift_res:
        return {"error": f"delta_grasp: {lift_res['error']}"}
    lift_steps = int(lift_res.get("steps_used", 0))
    steps_used += lift_steps
    state_end = motion._obs(primitives)["state"]
    opening_after_lift = motion._opening(state_end, arm)
    retained = bool(motion.retained_closure(0.0, opening_after_lift, closed_opening))
    lift_reached = bool(lift_res.get("reached"))
    lift_stalled = bool(lift_res.get("stalled"))
    lift_status = str(lift_res.get("status") or "rejected")
    result.update(
        steps_used=steps_used,
        lift_steps_used=lift_steps,
        lift_status=lift_status,
        opening_after_lift=round(opening_after_lift, 4),
        retained_closure=retained,
        stalled=lift_stalled,
        recoverable=(steps_used == 0),
    )
    if lift_res.get("ik_error"):
        result["ik_error"] = lift_res["ik_error"]
    if lift_reached:
        result.update(status="completed", reached=True, stop_reason=None)
        if table_contact_stalled:
            result["note"] = "tabletop contact seated fingertips; closed and lifted cleanly"
    elif lift_stalled:
        result.update(
            status="stopped",
            reached=False,
            stop_reason="contact_or_stall",
            note=lift_res.get("note") or "post-close liftoff stopped on contact or stall",
        )
    else:
        result.update(
            status="partial" if steps_used > 0 else "rejected",
            reached=False,
            stop_reason=lift_res.get("stop_reason") or "lift_not_reached",
        )
        if lift_res.get("note"):
            result["note"] = lift_res["note"]
    return finish(state_end)


def preview_delta_grasp(
    primitives: Any, start_joints: dict[str, list[float]], **kwargs: Any
) -> dict[str, Any]:
    """Read-only preflight for ``delta_grasp`` without moving any arm or gripper."""

    result = delta_grasp(primitives, start_joints, **kwargs, preview=True)
    if "error" not in result:
        result["preview"] = True
    return result


preview_delta_grasp._readonly = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

_DELTA = {"type": "object", "properties": {
    "dxyz": {"type": "array", "items": {"type": "number"}},
    "dpitch": {"type": "number"}, "dyaw": {"type": "number"}, "droll": {"type": "number"},
    "gripper": {"type": "number"}}}
_VEC3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_ALIGN_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_xyz": {"type": ["array", "null"], "items": {"type": "number"}},
        "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}},
        "arm": {"type": "string", "enum": [*ARMS, "both"]},
        "axis": _VEC3,
        "standoff_m": {"type": ["number", "null"]},
        "stroke_m": {"type": ["number", "null"]},
        "stroke_dxyz": {"type": ["array", "null"], "items": {"type": "number"}},
        "left": {"type": ["object", "null"]},
        "right": {"type": ["object", "null"]},
        "independent_stall": {"type": "boolean"},
        "align_tol_m": {"type": "number"},
        "max_linear_m": {"type": "number"},
        "max_steps": {"type": "integer"},
        "reserve_steps": {"type": "integer"},
        "stop_on_stall": {"type": "boolean"},
        "stall_steps": {"type": "integer"},
        "allow_table_contact": {"type": "boolean"},
    },
    "required": ["feature_xyz", "target_xyz"],
}
_DELTA_GRASP_SCHEMA = {
    "type": "object",
    "properties": {
        "target_xyz": {"type": ["array", "null"], "items": {"type": "number"}},
        "arm": {"type": "string", "enum": [*ARMS, "both"]},
        "axis": _VEC3,
        "standoff_m": {"type": "number"},
        "overdrive_m": {"type": "number"},
        "opening": {"type": ["number", "null"]},
        "contact_table": {"type": "boolean"},
        "lift_dxyz": {"type": ["array", "null"], "items": {"type": "number"}},
        "left": {"type": ["object", "array", "null"]},
        "right": {"type": ["object", "array", "null"]},
        "align_tol_m": {"type": "number"},
        "max_linear_m": {"type": "number"},
        "max_steps": {"type": "integer"},
        "reserve_steps": {"type": "integer"},
        "stall_steps": {"type": "integer"},
    },
    "required": ["target_xyz"],
}
_VISUAL_SERVO_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "arm": {"type": "string", "enum": [*ARMS, "both"]},
        "frame": {"type": "string", "enum": list(FRAMES)},
        "max_rounds": {"type": "integer"},
        "max_steps": {"type": "integer"},
        "reserve_steps": {"type": "integer"},
        "stop_on_stall": {"type": "boolean"},
        "stop_on_contact_complete": {"type": "boolean"},
        "crop_wrist": {"type": "boolean"},
        "cameras": {"type": "array", "items": {"type": "string"}},
        "model": {"type": "string"},
        "base_url": {"type": "string"},
        "max_linear_m": {"type": "number"},
        "max_angular_deg": {"type": "number"},
        "stall_steps": {"type": "integer"},
    },
    "required": ["goal"],
}
SPECS: dict[str, dict[str, Any]] = {
    "nudge": {
        "name": "nudge",
        "description": "Guarded compact relative move of the grasp point (metres / degrees, "
        "world or tcp frame); stops cleanly on a contact stall.",
        "input_schema": {"type": "object", "properties": {
            "arm": {"type": "string", "enum": [*ARMS, "both"]},
            "dxyz": {"type": "array", "items": {"type": "number"}},
            "dpitch": {"type": "number"}, "dyaw": {"type": "number"},
            "droll": {"type": "number"}, "gripper": {"type": "number"},
            "left": _DELTA, "right": _DELTA,
            "frame": {"type": "string", "enum": list(FRAMES)},
            "compact": {"type": "boolean"}, "stop_on_stall": {"type": "boolean"},
            "independent_stall": {"type": "boolean"},
            "stall_steps": {"type": "integer"}, "max_linear_m": {"type": "number"},
            "max_angular_deg": {"type": "number"}, "max_steps": {"type": "integer"},
            "reserve_steps": {"type": "integer"}, "step_m": {"type": "number"},
            "settle_steps": {"type": "integer"}, "tol_m": {"type": "number"},
            "tol_deg": {"type": "number"},
            "allow_table_contact": {"type": "boolean"}}, "required": []},
    },
    "visual_align_and_insert": {
        "name": "visual_align_and_insert",
        "description": "Bounded closed loop: camera views + state -> PEAR visual delta "
        "policy -> guarded nudge, with physical feedback each round.",
        "input_schema": _VISUAL_SERVO_SCHEMA,
    },
    "visual_servo": {
        "name": "visual_servo",
        "description": "General closed-loop last-mile visual servoing: camera views + state "
        "-> PEAR visual delta policy -> guarded nudge, with physical feedback each round.",
        "input_schema": _VISUAL_SERVO_SCHEMA,
    },
    "align_feature": {
        "name": "align_feature",
        "description": "Zero transverse feature-to-target offset perpendicular to an "
        "engagement axis via guarded compact nudge, with optional axial standoff and "
        "guarded contact stroke.",
        "input_schema": _ALIGN_FEATURE_SCHEMA,
    },
    "preview_align_feature": {
        "name": "preview_align_feature",
        "description": "Read-only preflight for align_feature (deltas, reach/table "
        "clearance and native step estimate); never moves.",
        "input_schema": _ALIGN_FEATURE_SCHEMA,
    },
    "delta_grasp": {
        "name": "delta_grasp",
        "description": "Compact close-range grasp / handoff: optional pre-open, transverse "
        "align, guarded axial approach or tabletop seating, close, and optional micro-step lift.",
        "input_schema": _DELTA_GRASP_SCHEMA,
    },
    "preview_delta_grasp": {
        "name": "preview_delta_grasp",
        "description": "Read-only preflight for delta_grasp (reach, table clearance and native "
        "step estimate); never moves.",
        "input_schema": _DELTA_GRASP_SCHEMA,
    },
}


def handlers(primitives: Any, start: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "nudge": partial(nudge, primitives, start),
        "visual_align_and_insert": partial(visual_align_and_insert, primitives, start),
        "visual_servo": partial(visual_align_and_insert, primitives, start),
        "align_feature": partial(align_feature, primitives, start),
        "preview_align_feature": partial(preview_align_feature, primitives, start),
        "delta_grasp": partial(delta_grasp, primitives, start),
        "preview_delta_grasp": partial(preview_delta_grasp, primitives, start),
    }


__all__ = [
    "ALIGN_TOL_M",
    "FINE_STEP_M",
    "MAX_ANGULAR_DEG",
    "MAX_LINEAR_M",
    "PREOPEN_TOL",
    "SPECS",
    "TABLE_TOUCH_OVERDRIVE_M",
    "THIN_HELD_OPENING",
    "ZERO_DELTA_M",
    "achieved_delta",
    "align_feature",
    "camera_images",
    "crop_center",
    "delta_grasp",
    "handlers",
    "has_configured_policy",
    "nudge",
    "policy_status",
    "preview_align_feature",
    "preview_delta_grasp",
    "resolve_policy",
    "set_policy",
    "target_pose",
    "visual_align_and_insert",
    "visual_servo",
]
