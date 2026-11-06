"""A bounded, observation-driven grasp attempt; no object poses or task verdicts.

All phases are preflighted before dispatch. Only the last 2 cm of approach
and the short probe use 4 mm Cartesian targets. Native interpolation retains
the existing simulator action path; this is not force control or a C2 promise.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np

from . import robodojo_kinematics as kin
from . import robodojo_tools as motion
from .robodojo_feedback import MAX_CORRECTIONS, ProgressTracker, TrackingLimits

POSITION_TOL_M = 0.0025
TURN_TOL_DEG = 3.0
LOCAL_STEP_M = 0.004
NEAR_M = 0.02


def _vector(value, name):
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be three finite numbers") from exc
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be three finite numbers")
    return out


def try_grasp(primitives, start_joints, xyz, arm="right", *, pitch=None,
              yaw=None, roll=None, approach=(0, 0, .05), lift=(0, 0, .025),
              opening=1.0, max_steps=64, preview=False):
    """One attempt: preflight, open, fast approach, local approach, close, probe.

    A partial attempt never restarts here. Cancellation propagates to the
    existing action journal; the caller must reconcile before planning the
    remaining work with a new operation ID. No automatic opening/regrasp after
    a failed probe, and no searching around an unseen contact.
    """
    try:
        if arm not in kin.ARMS:
            raise ValueError("arm must be 'left' or 'right'")
        xyz = _vector(xyz, "xyz")
        approach = _vector(approach, "approach")
        lift = _vector(lift, "lift")
        if not .02 <= np.linalg.norm(approach) <= .12:
            raise ValueError("approach offset must be between 0.02 and 0.12 m")
        if not .01 <= np.linalg.norm(lift) <= .06:
            raise ValueError("probe lift must be between 0.01 and 0.06 m")
        if type(max_steps) is not int or not 1 <= max_steps <= 120:
            raise ValueError("max_steps must be an integer between 1 and 120")
        if type(opening) not in (int, float) or not math.isfinite(opening) or not 0 < opening <= 1:
            raise ValueError("opening must be in (0, 1]")
        for angle in (pitch, yaw, roll):
            if angle is not None and (type(angle) not in (int, float) or not math.isfinite(angle)):
                raise ValueError("angles must be finite numbers")
    except ValueError as exc:
        return {"error": str(exc)}

    state = motion._obs(primitives)["state"]
    status = primitives.env.get_status()
    remaining = max(0, int(status.get("step_limit", 10000)) - int(status.get("step", 0)))
    limit = min(max_steps, remaining)
    q = motion._joints(state, arm)
    initial = motion._pose_of(arm, q)
    target, _ = motion._target(arm, q, xyz, pitch, yaw, roll)
    result = dict(arm=arm, status="rejected", reached=False, retained=False,
                  grasp_verified=False, needs_visual_confirmation=False,
                  steps_used=0, max_steps=max_steps, effective_step_limit=limit,
                  phase="preflight", phases=[], stop_reason=None,
                  execution_mode="bounded_native_grasp", retry_performed=False)
    tracker = ProgressTracker()

    def finish(reason=None):
        if result["steps_used"]:
            result["tracking"] = tracker.summary()
        result.update(stop_reason=reason, final_xyz=motion._tcp(arm, motion._joints(state, arm)),
                      opening=float(motion._opening(state, arm)),
                      status=("completed" if reason is None else
                              "partial" if result["steps_used"] else "rejected"))
        return result

    if motion._spent(status):
        return finish(motion._episode_stop_reason(status))
    last_command = (getattr(primitives, "_grip_commands", None) or {}).get(arm)
    measured = motion._opening(state, arm)
    if (last_command is not None and last_command <= .05 and measured > motion.HOLD_MIN_OPENING
            or last_command is None and motion.HOLD_MIN_OPENING < measured < .9):
        return finish("gripper_may_hold_object")

    # Validate the complete geometry first, including the probe, with the SAME
    # grasp orientation. No equivalent half-turn or approximate target swap.
    near = xyz + approach * (NEAR_M / np.linalg.norm(approach))
    points = [("transit", xyz + approach, "auto", .02),
              ("approach", near, "line", .02),
              ("contact", xyz, "line", LOCAL_STEP_M),
              ("probe", xyz + lift, "line", LOCAL_STEP_M)]
    plans = []
    for phase, point, path, spacing in points:
        primitives._check_cancelled()
        goal = target.copy()
        goal[:3, 3] = point
        if np.linalg.norm(motion._pose_of(arm, q) - goal) < 1e-5:
            ways, why, taken = [], None, "line"
        elif path == "auto":
            ways, taken, why, _ = motion._plan(arm, q, goal, path, start_joints)
        else:
            ways, why = kin.line_path(arm, q, goal, step_m=spacing, step_deg=3)
            ways, taken = motion._followable(q, ways) if why is None else [], "line"
        end = motion._pose_of(arm, ways[-1] if ways else q)
        if (why or np.linalg.norm(end[:3, 3] - point) > .001
                or kin.rotation_error(end[:3, :3], goal[:3, :3]) > 1):
            result.update(phase=phase, note=why or "exact grasp/probe pose is not reachable")
            return finish("grasp_path_rejected")
        plans.append((phase, ways, goal))
        result["phases"].append(
            dict(phase=phase, planned_steps=len(ways), path=taken, steps_used=0))
        q = ways[-1] if ways else q
    needs_open = abs(measured - opening) > .01
    bound = sum(len(ways) for _, ways, _ in plans) + motion.GRIP_MAX_ACTIONS * (1 + needs_open)
    bound += MAX_CORRECTIONS
    result.update(planned_step_bound=bound, remaining_episode_steps=remaining,
                  initial_xyz=initial[:3, 3].tolist(), replan_before_dispatch=True)
    if bound > limit:
        return finish("grasp_exceeds_step_budget")
    if preview:
        result.update(status="planned", accepted=True, preview=True)
        return result

    def dispatch(joints, command):
        nonlocal state, status
        if result["steps_used"] >= limit:
            return "grasp_step_budget"
        if motion._spent(status):
            return motion._episode_stop_reason(status)
        primitives._check_cancelled()
        state, status = motion._act(primitives, state, {arm: joints}, {arm: command})
        result["steps_used"] += 1
        return None

    def stroke(command, phase):
        result["phase"] = phase
        entry = dict(phase=phase, steps_used=0)
        result["phases"].append(entry)
        joints = motion._joints(state, arm)
        last = motion._opening(state, arm)
        for i in range(motion.GRIP_MAX_ACTIONS):
            why = dispatch(joints, command)
            if why:
                return why
            entry["steps_used"] += 1
            now = motion._opening(state, arm)
            if i >= 1 and abs(now - last) < motion.GRIP_STILL:
                return None
            last = now
        return "gripper_not_settled"

    if needs_open:
        why = stroke(opening, "open")
        if why:
            return finish(why)
        if abs(motion._opening(state, arm) - opening) > .01:
            return finish("gripper_opening_not_reached")
    for phase, ways, goal in plans:
        if phase == "probe":
            why = stroke(0., "close")
            if why:
                return finish(why)
            if motion._opening(state, arm) <= motion.HOLD_MIN_OPENING:
                return finish("empty_grasp")
            result["opening_after_close"] = float(motion._opening(state, arm))
        result["phase"] = phase
        entry = next(p for p in result["phases"] if p["phase"] == phase)
        for waypoint_index, joints in enumerate(ways):
            # Evaluate phase arrival against the requested goal, including the
            # IK residual, just like the final phase check below.
            endpoint = waypoint_index == len(ways) - 1
            expected = goal if endpoint else motion._pose_of(arm, joints)
            corrective = False
            while True:
                before = motion._pose_of(arm, motion._joints(state, arm))
                why = dispatch(joints, 0. if phase == "probe" else opening)
                if why:
                    return finish(why)
                if corrective:
                    tracker.corrected()
                    result["local_corrections"] = tracker.corrections
                entry["steps_used"] += 1
                actual = motion._pose_of(arm, motion._joints(state, arm))
                if phase == "probe" and motion._opening(state, arm) <= motion.HOLD_MIN_OPENING:
                    return finish("grasp_lost_during_probe")
                # Settle each phase endpoint before changing speed or closing.
                # Apply the same bounded correction rule here as near contact;
                # otherwise a small servo lag passes the coarse waypoint check
                # only to fail the endpoint check without a chance to settle.
                fine = phase in ("contact", "probe") or endpoint
                tolerance = POSITION_TOL_M if fine else motion.ARRIVE_M
                decision = tracker.assess(
                    {arm: before}, {arm: actual}, {arm: expected},
                    TrackingLimits(tolerance, TURN_TOL_DEG,
                                   position_ceiling=motion.ARRIVE_M,
                                   orientation_ceiling=TURN_TOL_DEG, progress_fraction=.2),
                    spare_steps=min(limit - result["steps_used"],
                                    status.get("step_limit", 10000) - status.get("step", 0)),
                )
                if decision == "advance":
                    break
                if decision == "stop":
                    result["tracking_error_m"] = tracker.last["errors"][arm]["position_error_m"]
                    return finish("tracking_error")
                corrective = True
        actual = motion._pose_of(arm, motion._joints(state, arm))
        if (np.linalg.norm(actual[:3, 3] - goal[:3, 3]) > POSITION_TOL_M
                or kin.rotation_error(actual[:3, :3], goal[:3, :3]) > TURN_TOL_DEG):
            return finish("phase_target_not_reached")
    result.update(reached=True, retained=True, needs_visual_confirmation=True,
                  probe_displacement_m=float(np.linalg.norm(
                      motion._pose_of(arm, motion._joints(state, arm))[:3, 3] - xyz)))
    return finish()


def preview_grasp(primitives, start_joints, **kwargs):
    result = try_grasp(primitives, start_joints, **kwargs, preview=True)
    result.setdefault("accepted", False)
    result["preview"] = True
    return result


preview_grasp._readonly = True
_VECTOR = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
_SCHEMA = {"type": "object", "properties": {
    "xyz": _VECTOR, "arm": {"type": "string", "enum": list(kin.ARMS)},
    **{k: {"type": "number"} for k in ("pitch", "yaw", "roll", "opening")},
    "approach": _VECTOR, "lift": _VECTOR,
    "max_steps": {"type": "integer", "minimum": 1, "maximum": 120},
}, "required": ["xyz"], "additionalProperties": False}
SPECS = {name: dict(name=name, description=description, input_schema=_SCHEMA)
         for name, description in (
             ("try_grasp", "Bounded preflighted approach, close and short probe; "
                           "visual confirmation required."),
             ("preview_grasp", "Read-only full grasp/probe path and native step-bound preview."))}


def handlers(primitives, start):
    return {"try_grasp": partial(try_grasp, primitives, start),
            "preview_grasp": partial(preview_grasp, primitives, start)}
