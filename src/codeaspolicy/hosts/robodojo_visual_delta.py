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
from collections.abc import Sequence
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


def _plan_arm(arm: str, q: Sequence[float], spec: dict[str, Any], frame: str,
              step_m: float, step_deg: float) -> tuple[np.ndarray, list, str | None]:
    """(target pose, one joint target per native action, why it cannot be done)."""

    target = target_pose(arm, q, spec, frame)
    if not _moves(spec):
        return target, [], None
    if target[2, 3] < kin.TABLE_Z + 0.002:
        return target, [], "the target grasp point is below the table top"
    ways, why = kin.line_path(arm, q, target, step_m=step_m, step_deg=step_deg)
    if why is not None:
        return target, [], why
    steps = motion._followable(q, ways)
    for way in steps:
        if not _within_limits(way):
            return target, [], "the line needs a joint beyond its limits"
        bad = kin.clearance_violations(arm, way)
        if bad:
            return target, [], f"the line takes the arm into the table ({', '.join(bad)})"
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
    stall_steps: int = 2,
    max_linear_m: float = MAX_LINEAR_M,
    max_angular_deg: float = MAX_ANGULAR_DEG,
    max_steps: int | None = None,
    reserve_steps: int = 0,
    step_m: float = LINE_STEP_M,
    settle_steps: int = SETTLE_STEPS,
    tol_m: float = TOL_M,
    tol_deg: float = TOL_DEG,
) -> dict[str, Any]:
    """Move one arm (or both) by a bounded relative delta of the grasp point.

    ``dxyz`` metres, ``dpitch`` / ``dyaw`` / ``droll`` degrees, ``gripper`` in
    RoboDojo's command convention (1 close, -1 open, 0 keep).  ``arm="both"`` takes
    ``left`` / ``right`` dicts with the same keys.  See the module docstring.
    """

    try:
        specs = _arguments(arm, dxyz, dpitch, dyaw, droll, gripper, left, right, frame,
                           stall_steps, max_linear_m, max_angular_deg, max_steps,
                           reserve_steps, step_m, settle_steps, tol_m, tol_deg)
    except ValueError as exc:
        return {"error": f"nudge: {exc}"}
    if not isinstance(compact, bool) or not isinstance(stop_on_stall, bool):
        return {"error": "nudge: compact and stop_on_stall must be booleans"}
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
    step_deg = LINE_STEP_DEG * step_m / LINE_STEP_M
    plans = {a: _plan_arm(a, q0[a], specs[a], frame, step_m, step_deg) for a in specs}
    targets = {a: plan[0] for a, plan in plans.items()}
    refused = [f"{a}: {plan[2]}" for a, plan in plans.items() if plan[2]]
    if refused:
        result.update(stop_reason="ik_path_rejected", ik_error="; ".join(refused))
        return finish(state, targets)
    ways = {a: plan[1] for a, plan in plans.items()}
    planned = max(len(w) for w in ways.values())
    grips = {a: motion._command(specs[a]["gripper"]) for a in specs}
    result["planned_steps"] = planned
    status, remaining = _budget(primitives)
    allowed = remaining - reserve_steps
    if max_steps is not None:
        allowed = min(allowed, max_steps)
    if planned == 0 and all(g is None for g in grips.values()):
        result.update(status="completed", reached=True, stop_reason=None,
                      note="zero delta: nothing to do")
        return finish(state, targets)
    if allowed <= 0:
        result.update(stop_reason="episode_native_step_budget" if remaining <= 0
                      else "reserve_steps_exhausted")
        return finish(state, targets)
    return _execute(primitives, state, status, specs, start, targets, ways, grips, planned,
                    allowed, stop_on_stall, stall_steps, step_m, settle_steps, tol_m,
                    tol_deg, result, finish)


def _execute(primitives, state, status, specs, start, targets, ways, grips, planned, allowed,
             stop_on_stall, stall_steps, step_m, settle_steps, tol_m, tol_deg, result,
             finish) -> dict[str, Any]:
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
        cmd = {a: list(ways[a][min(index, len(ways[a]) - 1)]) if ways[a]
               else motion._joints(state, a) if a not in moving and grips[a] is None
               else last_cmd[a] for a in specs}
        act(cmd)
        hit = [judge(a, motion._pose_of(a, cmd[a])) for a in moving if index < len(ways[a])]
        if any(hit) and stop_on_stall:
            stop = "contact_or_stall"
            break
    # An arm whose own path ended early kept holding: judge its pose as measured now.
    measured.update({a: motion._pose_of(a, motion._joints(state, a)) for a in moving})
    settles = 0
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
        hold = {a: motion._joints(state, a) if stop == "contact_or_stall" else last_cmd[a]
                for a in specs}
        for _ in range(motion.GRIP_MAX_ACTIONS):
            if steps >= allowed or motion._spent(status):
                break
            if steps >= 2 and all(open_change[a] < motion.GRIP_STILL
                                  for a in specs if grips[a] is not None):
                break
            act(hold)
    measured.update({a: motion._pose_of(a, motion._joints(state, a)) for a in moving})
    reached = all(error(a)[0] <= tol_m and error(a)[1] <= tol_deg for a in moving)
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
    return finish(state, targets)


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
    base: dict[str, Any] = {"protocol": vd.PROTOCOL, "goal": goal, "arm": arm, "frame": frame,
                            "max_rounds": max_rounds, "reserve_steps": reserve_steps,
                            "step_allowance": max(0, allowance), "cameras": list(views)}
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


def align_feature(
    primitives: Any,
    start_joints: dict[str, list[float]],
    feature_xyz: Sequence[float],
    target_xyz: Sequence[float],
    arm: str = "right",
    *,
    axis: Sequence[float] = (0.0, 0.0, -1.0),
    standoff_m: float | None = None,
    stroke_m: float | None = None,
    stroke_dxyz: Sequence[float] | None = None,
    align_tol_m: float = ALIGN_TOL_M,
    max_linear_m: float = MAX_LINEAR_M,
    max_steps: int = 15,
    reserve_steps: int = 20,
    stop_on_stall: bool = True,
    stall_steps: int = 2,
    preview: bool = False,
) -> dict[str, Any]:
    """Zero transverse offset from a held distal feature to a target, with optional stroke.

    Projects ``target_xyz - feature_xyz`` perpendicular to the unit engagement ``axis``
    (plus an optional axial ``standoff_m`` upstream along ``-axis``), preflights reach,
    table clearance and budget, executes a compact guarded ``nudge`` for alignment,
    measures residual ``transverse_error_after_m`` from the actual TCP shift, and when
    ``aligned`` (within ``align_tol_m``) optionally executes a guarded engagement stroke
    (``stroke_m`` along ``+axis`` or explicit ``stroke_dxyz``).
    """

    try:
        if arm not in ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        f_xyz = _vec3(feature_xyz, "feature_xyz")
        t_xyz = _vec3(target_xyz, "target_xyz")
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
        if not isinstance(stop_on_stall, bool) or not isinstance(preview, bool):
            raise ValueError("stop_on_stall and preview must be booleans")
    except ValueError as exc:
        return {"error": f"align_feature: {exc}"}

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

    if skip_align:
        align_ways: list[list[float]] = []
        q_after_align_plan = q0
    else:
        align_spec = _spec({"dxyz": planned_dxyz.tolist()}, arm)
        _, align_ways, align_why = _plan_arm(arm, q0, align_spec, "world",
                                             LINE_STEP_M, LINE_STEP_DEG)
        if align_why is not None:
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm}: {align_why}",
                stroke_status="rejected" if stroke_vec is not None else None,
            )
            return finish(state0)
        q_after_align_plan = align_ways[-1] if align_ways else q0

    if stroke_vec is not None:
        stroke_spec = _spec({"dxyz": stroke_vec.tolist()}, arm)
        _, stroke_ways, stroke_why = _plan_arm(arm, q_after_align_plan, stroke_spec, "world",
                                               LINE_STEP_M, LINE_STEP_DEG)
        if stroke_why is not None:
            result.update(
                stop_reason="ik_path_rejected",
                ik_error=f"{arm} stroke: {stroke_why}",
                stroke_status="rejected",
            )
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
        if planned_steps > allowed:
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
            tol_m=min(TOL_M, tol_f),
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
        result.update(
            status="stopped",
            reached=False,
            stop_reason="contact_or_stall",
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
        "feature_xyz": _VEC3,
        "target_xyz": _VEC3,
        "arm": {"type": "string", "enum": list(ARMS)},
        "axis": _VEC3,
        "standoff_m": {"type": ["number", "null"]},
        "stroke_m": {"type": ["number", "null"]},
        "stroke_dxyz": {"type": ["array", "null"], "items": {"type": "number"}},
        "align_tol_m": {"type": "number"},
        "max_linear_m": {"type": "number"},
        "max_steps": {"type": "integer"},
        "reserve_steps": {"type": "integer"},
        "stop_on_stall": {"type": "boolean"},
        "stall_steps": {"type": "integer"},
    },
    "required": ["feature_xyz", "target_xyz"],
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
            "stall_steps": {"type": "integer"}, "max_linear_m": {"type": "number"},
            "max_angular_deg": {"type": "number"}, "max_steps": {"type": "integer"},
            "reserve_steps": {"type": "integer"}, "step_m": {"type": "number"},
            "settle_steps": {"type": "integer"}, "tol_m": {"type": "number"},
            "tol_deg": {"type": "number"}}, "required": []},
    },
    "visual_align_and_insert": {
        "name": "visual_align_and_insert",
        "description": "Bounded closed loop: camera views + state -> PEAR visual delta "
        "policy -> guarded nudge, with physical feedback each round.",
        "input_schema": {"type": "object", "properties": {
            "goal": {"type": "string"},
            "arm": {"type": "string", "enum": [*ARMS, "both"]},
            "frame": {"type": "string", "enum": list(FRAMES)},
            "max_rounds": {"type": "integer"}, "max_steps": {"type": "integer"},
            "reserve_steps": {"type": "integer"}, "stop_on_stall": {"type": "boolean"},
            "stop_on_contact_complete": {"type": "boolean"},
            "crop_wrist": {"type": "boolean"},
            "cameras": {"type": "array", "items": {"type": "string"}},
            "model": {"type": "string"}, "base_url": {"type": "string"},
            "max_linear_m": {"type": "number"}, "max_angular_deg": {"type": "number"},
            "stall_steps": {"type": "integer"}}, "required": ["goal"]},
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
}


def handlers(primitives: Any, start: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "nudge": partial(nudge, primitives, start),
        "visual_align_and_insert": partial(visual_align_and_insert, primitives, start),
        "align_feature": partial(align_feature, primitives, start),
        "preview_align_feature": partial(preview_align_feature, primitives, start),
    }


__all__ = [
    "ALIGN_TOL_M",
    "MAX_ANGULAR_DEG",
    "MAX_LINEAR_M",
    "SPECS",
    "ZERO_DELTA_M",
    "achieved_delta",
    "align_feature",
    "camera_images",
    "crop_center",
    "handlers",
    "nudge",
    "preview_align_feature",
    "resolve_policy",
    "set_policy",
    "target_pose",
    "visual_align_and_insert",
]
