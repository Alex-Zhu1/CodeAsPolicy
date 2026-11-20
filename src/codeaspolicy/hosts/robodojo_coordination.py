"""Bounded dual-arm trajectories over RoboDojo's single native action clock.

No threads, no hidden object state, no silent sequential fallback. Geometry
checks cover sampled arm/table clearance only; inter-arm collision and force
control are deliberately not advertised.
"""

from __future__ import annotations

import math
import time
import uuid
from functools import partial

import numpy as np

from ..coordination import Pose
from ..motion import resolve_motion
from . import robodojo_kinematics as kin
from . import robodojo_tools as tools
from . import robodojo_trajectory as trajectory
from .robodojo_feedback import ProgressTracker, TrackingLimits


def _positive(value, name, upper, *, integer=False):
    if isinstance(value, bool) or (integer and type(value) is not int):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'}")
    if not math.isfinite(value) or not 0 < value <= upper:
        raise ValueError(f"{name} must be positive and <= {upper}")
    return value


def _matrix(pose):
    p = Pose(**pose)
    out = np.eye(4)
    out[:3, :3] = kin.rot_from_quat(p.quat)
    out[:3, 3] = p.xyz
    return out


def _errors(now, target):
    return (
        float(np.linalg.norm(now[:3, 3] - target[:3, 3])),
        kin.rotation_error(now[:3, :3], target[:3, :3]),
    )


def capabilities(primitives, *, status=None):
    """One capability description for the live API and startup task cards."""
    from .robodojo_intent import capabilities as intent_capabilities

    return dict(
        motion_intents=intent_capabilities(),
        modes=["synchronized"],
        frames=["world"],
        points=["tcp"],
        orientation="unit quaternion [w,x,y,z]",
        position_unit="metre",
        grippers=["keep", "open", "close"],
        relative_pose="sampled_path_and_native_feedback",
        keep_initial="last command; measured opening only if no command has been recorded",
        collision_checks=["sampled_arm_table"],
        inter_arm_clearance=False,
        object_collision=False,
        **trajectory.capabilities(primitives, status=status),
        synchronized="shared Cartesian progress; bounded measured tracking, not rigid coupling",
    )


class Controller:
    def __init__(self, primitives):
        self.primitives = primitives
        self.observations: dict[str, dict] = {}
        self.plans: dict[str, dict] = {}

    def capabilities(self):
        return capabilities(self.primitives)

    def _native_steps(self):
        get_status = getattr(self.primitives.env, "get_status", None)
        status = get_status() if callable(get_status) else {}
        return status.get("step")

    def _snapshot(self):
        state = tools._obs(self.primitives)["state"]
        q = {a: tools._joints(state, a) for a in kin.ARMS}
        return dict(
            revision=getattr(self.primitives, "_motion_revision", 0),
            native_steps=self._native_steps(),
            joints=q,
            opening={a: tools._opening(state, a) for a in kin.ARMS},
            commands={
                a: (getattr(self.primitives, "_grip_commands", {}) or {}).get(
                    a, tools._opening(state, a)
                )
                for a in kin.ARMS
            },
            poses={a: tools._pose_of(a, q[a]) for a in kin.ARMS},
        )

    def _same(self, snapshot):
        now = self._snapshot()
        return (
            now["revision"] == snapshot["revision"]
            and now["native_steps"] == snapshot["native_steps"]
            and now["joints"] == snapshot["joints"]
            and now["opening"] == snapshot["opening"]
            and now["commands"] == snapshot["commands"]
        )

    def _view(self, snap):
        out = dict(native_revision=snap["revision"], native_steps=snap["native_steps"])
        for arm in kin.ARMS:
            pose = snap["poses"][arm]
            out[arm] = dict(
                pose=dict(xyz=pose[:3, 3].tolist(), quat=list(kin.quat_wxyz(pose[:3, :3]))),
                opening=snap["opening"][arm],
            )
        return out

    def observe(self):
        snap = self._snapshot()
        token = uuid.uuid4().hex
        if len(self.observations) >= 16:
            self.observations.pop(next(iter(self.observations)))
        self.observations[token] = snap
        return dict(id=token, **self._view(snap))

    def _reason(self, plan):
        if "episode_reserve" in plan:
            status = self.primitives.env.get_status()
            if tools._spent(status):
                return tools._episode_stop_reason(status)
            if status.get("step_limit", 10000) - status.get("step", 0) <= plan["episode_reserve"]:
                return "reserved_episode_budget"
        if time.monotonic() >= plan["deadline"]:
            return "motion_timeout"
        if plan["steps"] >= plan["max_steps"]:
            return "native_step_budget"
        if plan.get("clock") is not None and trajectory.backend(self.primitives) != plan["clock"]:
            return "trajectory_clock_changed"
        return None

    def plan(
        self,
        observation_id,
        targets,
        grippers,
        keep_relative,
        max_steps,
        timeout_s,
        position_tolerance_m,
        orientation_tolerance_deg,
        max_step_m,
        max_step_deg,
        checkpoint_steps,
        require_inter_arm_clearance,
        motion=None,
        via=None,
        preview=False,
        tracking_position_tolerance_m=None,
        tracking_orientation_tolerance_deg=None,
        allow_joint_fallback=False,
        simplify_tolerance=1e-4,
    ):
        started = time.monotonic()
        if type(preview) is not bool or type(allow_joint_fallback) is not bool:
            raise ValueError("preview and allow_joint_fallback must be bool")
        _positive(simplify_tolerance, "simplify_tolerance", 0.5)
        _positive(max_steps, "max_steps", 10000, integer=True)
        _positive(timeout_s, "timeout_s", 3600)
        _positive(checkpoint_steps, "checkpoint_steps", 8, integer=True)
        if type(keep_relative) is not bool or type(require_inter_arm_clearance) is not bool:
            raise ValueError("constraint switches must be bool")
        if set(targets) - set(kin.ARMS) or set(grippers) != set(kin.ARMS):
            raise ValueError("targets/grippers must name left/right")
        if any(v not in ("keep", "open", "close") for v in grippers.values()):
            raise ValueError("invalid gripper command")
        for key, value, upper in (
            ("position_tolerance_m", position_tolerance_m, 0.05),
            ("orientation_tolerance_deg", orientation_tolerance_deg, 20),
            ("max_step_m", max_step_m, 0.02),
            ("max_step_deg", max_step_deg, 8),
        ):
            _positive(value, key, upper)
        # Intent execution supplies a scene-independent tracking envelope. Keep
        # direct legacy callers compatible when they do not supply one.
        tracking_position = (position_tolerance_m if tracking_position_tolerance_m is None
                             else tracking_position_tolerance_m)
        tracking_orientation = (orientation_tolerance_deg
                                if tracking_orientation_tolerance_deg is None
                                else tracking_orientation_tolerance_deg)
        _positive(tracking_position, "tracking_position_tolerance_m", .05)
        _positive(tracking_orientation, "tracking_orientation_tolerance_deg", 20)
        if via is not None and (not isinstance(via, list) or len(via) > 128
                                or any(not isinstance(v, dict) or set(v) - set(kin.ARMS)
                                       for v in via)):
            raise ValueError("via must contain at most 128 left/right waypoint mappings")
        route = [*(via or []), targets]
        snap = self.observations.get(observation_id)
        current = self._snapshot()
        goals = dict(current["poses"])
        for leg in route:
            goals.update({a: _matrix(p) for a, p in leg.items()})
        limits = dict(
            position=position_tolerance_m,
            orientation=orientation_tolerance_deg,
            relative_position=position_tolerance_m,
            relative_angle=orientation_tolerance_deg,
            tracking_position=tracking_position,
            tracking_orientation=tracking_orientation,
        )
        plan = dict(
            id=uuid.uuid4().hex,
            expected=snap,
            goals=goals,
            steps=0,
            cursor=0,
            stopped=False,
            limits=limits,
            keep_relative=keep_relative,
            checkpoint_steps=checkpoint_steps,
            checked=[],
            max_steps=max_steps,
            deadline=started + timeout_s,
            tracker=ProgressTracker(),
            correcting=False,
        )

        def reject(reason, *, recoverable=True):
            result = self._report(plan, "rejected", reason)
            result.update(accepted=False, recoverable=recoverable)
            if "planned_path" in plan:
                result["planned_path"] = plan["planned_path"]
            if "goal_joints" in plan:
                result["goal_joints"] = plan["goal_joints"]
            if preview:
                result.update(preview=True, replan_before_dispatch=True)
            return result

        if require_inter_arm_clearance:
            return reject("unsupported: inter-arm collision geometry")
        if snap is None or not self._same(snap):
            return reject("stale_observation")
        native = motion == {"mode": "native"}
        clock = None if native else trajectory.backend(self.primitives)
        if not native and clock is None and (motion is not None or via):
            return reject("unsupported: timed joint trajectories")
        can_joint_fallback = allow_joint_fallback and not keep_relative
        used_joint_fallback = False
        rel = np.linalg.inv(snap["poses"]["left"]) @ snap["poses"]["right"]
        q = {a: np.asarray(snap["joints"][a], float) for a in kin.ARMS}
        poses = {a: snap["poses"][a].copy() for a in kin.ARMS}
        ways = []
        arms = kin.ARMS
        commands = dict(grippers)
        for leg in route:
            leg_goals = {a: _matrix(leg[a]) if a in leg else poses[a] for a in kin.ARMS}
            begin = {a: poses[a].copy() for a in kin.ARMS}
            counts = [1]
            for a in arms:
                distance, angle = _errors(begin[a], leg_goals[a])
                counts.extend((math.ceil(distance / max_step_m), math.ceil(angle / max_step_deg)))
            count = max(counts)
            raw_count_limit = 1000 if clock is not None else min(1000, max_steps)
            raw_ways_limit = 2000 if clock is not None else max_steps
            if count > raw_count_limit and not can_joint_fallback:
                plan["planned_path"] = "joint" if used_joint_fallback else "line"
                return reject("planned_path_exceeds_native_step_budget")
            q_leg = {a: q[a].copy() for a in kin.ARMS}
            poses_leg = {a: poses[a].copy() for a in kin.ARMS}
            leg_ways = []
            line_error = None
            budget_error = count > raw_count_limit
            if not budget_error:
                knots_q = [{a: q_leg[a].copy() for a in kin.ARMS}]
                knot_poses = [{a: poses_leg[a].copy() for a in kin.ARMS}]
                for i in range(1, count + 1):
                    if time.monotonic() >= plan["deadline"]:
                        return reject("motion_timeout", recoverable=False)
                    self.primitives._check_cancelled()
                    prev_q = knots_q[-1]
                    solved = {a: prev_q[a].copy() for a in kin.ARMS}
                    solved_poses = {}
                    for a in arms:
                        desired = np.eye(4)
                        desired[:3, 3] = (begin[a][:3, 3]
                                           + (leg_goals[a][:3, 3] - begin[a][:3, 3]) * i / count)
                        desired[:3, :3] = kin.slerp(
                            begin[a][:3, :3], leg_goals[a][:3, :3], i / count
                        )
                        res = kin.solve(a, desired, prev_q[a])
                        if not res.ok or res.collides:
                            line_error = (
                                f"{a}: unreachable or table collision at path fraction {i / count}"
                            )
                            break
                        pose_res = tools._pose_of(a, res.q)
                        err_m, err_deg = _errors(pose_res, desired)
                        if err_m > position_tolerance_m or err_deg > orientation_tolerance_deg:
                            line_error = f"{a}: IK cannot meet requested tolerance"
                            break
                        if max(abs(np.asarray(res.q) - prev_q[a])) > 0.6:
                            line_error = f"{a}: singularity/IK branch jump"
                            break
                        solved[a] = np.asarray(res.q, float)
                        solved_poses[a] = pose_res
                    if line_error is not None:
                        break
                    knots_q.append(solved)
                    knot_poses.append(solved_poses)
                if line_error is None:
                    all_single_subdiv = all(
                        max(
                            float(np.max(np.abs(knots_q[i][a] - knots_q[i - 1][a])))
                            for a in kin.ARMS
                        )
                        <= tools.JOINT_STEP_RAD
                        for i in range(1, count + 1)
                    )
                    if all_single_subdiv:
                        for i in range(1, count + 1):
                            point = {a: knots_q[i][a].tolist() for a in kin.ARMS}
                            now = knot_poses[i]
                            for a in kin.ARMS:
                                if kin.clearance_violations(a, point[a]):
                                    line_error = f"{a}: interpolated path intersects table"
                                    break
                                dist, angle = _errors(now[a], poses_leg[a])
                                # IK tolerance can perturb an ideal Cartesian sample slightly.
                                if dist > max_step_m + 0.001 or angle > max_step_deg + 0.5:
                                    line_error = f"{a}: interpolated path exceeds TCP step limit"
                                    break
                            if line_error is not None:
                                break
                            if keep_relative and self._relative_error(now, rel, limits):
                                line_error = (
                                    "relative_pose_constraint: sampled path changes arm relationship"
                                )
                                break
                            leg_ways.append(dict(joints=point, grippers=dict(commands)))
                            if len(ways) + len(leg_ways) > raw_ways_limit:
                                budget_error = True
                                break
                            poses_leg = now
                        if line_error is None and not budget_error:
                            q_leg = knots_q[count]
                    else:
                        def _eval_q(t):
                            if t >= count:
                                return {a: knots_q[count][a].copy() for a in kin.ARMS}
                            idx = int(math.floor(t))
                            alpha = t - idx
                            return {
                                a: knots_q[idx][a] + alpha * (knots_q[idx + 1][a] - knots_q[idx][a])
                                for a in kin.ARMS
                            }

                        def _check_cand(t, cur_q, cur_poses):
                            cand_q = _eval_q(t)
                            if (max(float(np.max(np.abs(cand_q[a] - cur_q[a]))) for a in kin.ARMS)
                                    > tools.JOINT_STEP_RAD + 1e-9):
                                return None, None, "joint_step"
                            cand_poses = (
                                knot_poses[count]
                                if t >= count
                                else {a: tools._pose_of(a, cand_q[a]) for a in kin.ARMS}
                            )
                            for a in kin.ARMS:
                                if kin.clearance_violations(a, cand_q[a]):
                                    return None, None, f"{a}: interpolated path intersects table"
                                dist, angle = _errors(cand_poses[a], cur_poses[a])
                                if dist > max_step_m + 0.001 or angle > max_step_deg + 0.5:
                                    return None, None, f"{a}: interpolated path exceeds TCP step limit"
                            if keep_relative and self._relative_error(cand_poses, rel, limits):
                                return (
                                    None,
                                    None,
                                    "relative_pose_constraint: sampled path changes arm relationship",
                                )
                            return cand_q, cand_poses, None

                        s = 0.0
                        q_cur = knots_q[0]
                        poses_cur = poses_leg
                        r_lim = tools.JOINT_STEP_RAD - 1e-9
                        while s < count - 1e-9:
                            base_seg = min(count - 1, int(math.floor(s)))
                            seg = base_seg
                            s_cap = s
                            while seg < count:
                                p0 = q_cur if seg == base_seg else knots_q[seg]
                                t0 = s if seg == base_seg else float(seg)
                                p1 = knots_q[seg + 1]
                                t1 = float(seg + 1)
                                dq_end = max(
                                    float(np.max(np.abs(p1[a] - q_cur[a]))) for a in kin.ARMS
                                )
                                if dq_end > r_lim:
                                    lam_max = 1.0
                                    for a in kin.ARMS:
                                        d0 = p0[a] - q_cur[a]
                                        dv = p1[a] - p0[a]
                                        for k in range(6):
                                            if dv[k] > 1e-12:
                                                lam_max = min(lam_max, (r_lim - d0[k]) / dv[k])
                                            elif dv[k] < -1e-12:
                                                lam_max = min(lam_max, (-r_lim - d0[k]) / dv[k])
                                    lam_max = max(0.0, min(1.0, lam_max))
                                    s_cap = t0 + lam_max * (t1 - t0)
                                    break
                                tcp_over = False
                                for a in kin.ARMS:
                                    d_k, ang_k = _errors(knot_poses[seg + 1][a], poses_cur[a])
                                    if d_k > max_step_m + 0.001 or ang_k > max_step_deg + 0.5:
                                        tcp_over = True
                                        break
                                s_cap = t1
                                if tcp_over:
                                    break
                                seg += 1
                            if s_cap >= count - 1e-9:
                                s_cap = float(count)
                            if s_cap <= s + 1e-9:
                                line_error = "interpolated path stalled on joint step limit"
                                break
                            cand_q, cand_poses, err = _check_cand(s_cap, q_cur, poses_cur)
                            chosen_s = s_cap
                            if err is not None:
                                if "exceeds TCP step limit" in err:
                                    lo_s, hi_s = s, s_cap
                                    best = None
                                    for _ in range(10):
                                        mid_s = 0.5 * (lo_s + hi_s)
                                        mq, mp, merr = _check_cand(mid_s, q_cur, poses_cur)
                                        if merr is None:
                                            best = (mid_s, mq, mp)
                                            lo_s = mid_s
                                        else:
                                            hi_s = mid_s
                                            err = merr
                                    if best is None or (best[0] < count and best[0] - s < 1e-3):
                                        line_error = err
                                        break
                                    chosen_s, cand_q, cand_poses = best
                                else:
                                    next_knot = float(math.floor(s) + 1)
                                    if s_cap > next_knot and next_knot > s + 1e-9:
                                        kq, kp, kerr = _check_cand(next_knot, q_cur, poses_cur)
                                        if kerr is None:
                                            chosen_s, cand_q, cand_poses = next_knot, kq, kp
                                        else:
                                            line_error = kerr
                                            break
                                    else:
                                        line_error = err
                                        break
                            leg_ways.append(dict(
                                joints={a: cand_q[a].tolist() for a in kin.ARMS},
                                grippers=dict(commands),
                            ))
                            if len(ways) + len(leg_ways) > raw_ways_limit:
                                budget_error = True
                                break
                            s = chosen_s
                            q_cur = cand_q
                            poses_cur = cand_poses
                        if line_error is None and not budget_error:
                            q_leg = knots_q[count]
                            poses_leg = poses_cur
            if budget_error and line_error is None:
                plan["planned_path"] = "joint" if used_joint_fallback else "line"
                return reject("planned_path_exceeds_native_step_budget")
            if line_error is None:
                ways.extend(leg_ways)
                q, poses = q_leg, poses_leg
            elif not can_joint_fallback:
                return reject(line_error)
            else:
                solved_goal = {}
                for a in arms:
                    dist_goal, ang_goal = _errors(poses[a], leg_goals[a])
                    if dist_goal <= 1e-6 and ang_goal <= 1e-4:
                        solved_goal[a] = q[a].copy()
                        continue
                    res = kin.best_solution(
                        a, leg_goals[a], q[a],
                        pos_tol=position_tolerance_m,
                        rot_tol_deg=orientation_tolerance_deg,
                        extra_seeds=[(0.0,) * 6],
                    )
                    if not res.ok or res.collides:
                        return reject(line_error)
                    err_m, err_deg = _errors(tools._pose_of(a, res.q), leg_goals[a])
                    if err_m > position_tolerance_m or err_deg > orientation_tolerance_deg:
                        return reject(line_error)
                    solved_goal[a] = np.asarray(res.q, float)
                fallback_clock = trajectory.backend(self.primitives) if native else clock
                step_rad = (
                    tools.JOINT_STEP_RAD
                    if fallback_clock is not None
                    else min(tools.JOINT_STEP_RAD, math.radians(max_step_deg))
                )
                subdivisions = max(
                    1,
                    math.ceil(
                        max(float(np.max(np.abs(solved_goal[a] - q[a]))) for a in kin.ARMS)
                        / step_rad
                    ),
                )
                if fallback_clock is None and subdivisions < 2000:
                    max_chord_ratio = 1.0
                    prev_probe = {a: poses[a] for a in kin.ARMS}
                    for j in range(1, subdivisions + 1):
                        probe_now = {
                            a: tools._pose_of(
                                a, (q[a] + (solved_goal[a] - q[a]) * j / subdivisions).tolist()
                            )
                            for a in kin.ARMS
                        }
                        for a in kin.ARMS:
                            d_step, ang_step = _errors(probe_now[a], prev_probe[a])
                            max_chord_ratio = max(
                                max_chord_ratio,
                                d_step / max(max_step_m, 1e-6),
                                ang_step / max(max_step_deg, 1e-6),
                            )
                        prev_probe = probe_now
                    if max_chord_ratio > 1.0:
                        subdivisions = math.ceil(subdivisions * max_chord_ratio)
                if subdivisions > 2000:
                    return reject("planned_path_exceeds_native_step_budget")
                joint_ways = []
                poses_joint = {a: poses[a].copy() for a in kin.ARMS}
                for j in range(1, subdivisions + 1):
                    point = {
                        a: (q[a] + (solved_goal[a] - q[a]) * j / subdivisions).tolist()
                        for a in kin.ARMS
                    }
                    now = {a: tools._pose_of(a, point[a]) for a in kin.ARMS}
                    for a in kin.ARMS:
                        if kin.clearance_violations(a, point[a]):
                            return reject(line_error)
                    joint_ways.append(dict(joints=point, grippers=dict(commands)))
                    poses_joint = now
                used_joint_fallback = True
                if native:
                    fallback_clock = trajectory.backend(self.primitives)
                    if fallback_clock is not None:
                        native = False
                        clock = fallback_clock
                        motion = resolve_motion("transit").payload()
                if clock is None and len(ways) + len(joint_ways) > max_steps:
                    plan["planned_path"] = "joint"
                    plan["goal_joints"] = {a: solved_goal[a].tolist() for a in kin.ARMS}
                    return reject("planned_path_exceeds_native_step_budget")
                ways.extend(joint_ways)
                q, poses = solved_goal, poses_joint

        plan["planned_path"] = "joint" if used_joint_fallback else "line"
        plan["goal_joints"] = {a: q[a].tolist() for a in kin.ARMS}
        if clock is not None:
            def check(joints):
                self.primitives._check_cancelled()
                if time.monotonic() >= plan["deadline"]:
                    raise ValueError("motion_timeout")
                trajectory.check_joints(joints, start=snap["joints"])
                if keep_relative:
                    measured = {a: tools._pose_of(a, joints[a]) for a in kin.ARMS}
                    if self._relative_error(measured, rel, limits):
                        raise ValueError("relative_pose_constraint: timed substep")

            grip_steps = (tools.GRIP_MAX_ACTIONS
                          if any(v != "keep" for v in grippers.values()) else 0)
            status = self.primitives.env.get_status()
            available = min(max_steps, status.get("step_limit", 10000) - status.get("step", 0))
            try:
                ways, plan["trajectory"] = trajectory.parameterize(
                    snap["joints"], [w["joints"] for w in ways], clock, motion,
                    max_actions=available - grip_steps, check=check,
                    line_tolerance_m=None if used_joint_fallback else 0.003,
                    _simplify_tolerance=simplify_tolerance,
                )
                previous = snap["poses"]
                for way in ways:
                    way["grippers"] = dict(commands)
                    if not used_joint_fallback:
                        block = way.get("trajectory")
                        samples = (
                            [{a: row[idx * 6:(idx + 1) * 6] for idx, a in enumerate(kin.ARMS)}
                             for row in block["positions"]]
                            if block else [way["joints"]]
                        )
                        for joints_sample in samples:
                            now = {a: tools._pose_of(a, joints_sample[a]) for a in kin.ARMS}
                            for a in kin.ARMS:
                                dist, angle = _errors(now[a], previous[a])
                                if dist > max_step_m + 0.001 or angle > max_step_deg + 0.5:
                                    raise ValueError(f"{a}: timed path exceeds TCP step limit")
                            previous = now
            except ValueError as exc:
                if isinstance(exc, trajectory.TrajectoryBudgetError):
                    plan["trajectory"] = exc.trajectory
                return reject(str(exc), recoverable=str(exc) != "motion_timeout")
        plan["motion_steps"] = len(ways)
        # Let each requested gripper stroke finish while both arm poses hold.
        if any(grippers[a] != "keep" for a in arms):
            for _ in range(tools.GRIP_MAX_ACTIONS):
                point = {a: q[a].tolist() for a in kin.ARMS}
                way = dict(joints=point, grippers=dict(commands))
                if clock is not None:
                    way["trajectory"] = trajectory.hold(point, clock)
                ways.append(way)
            if clock is not None:
                plan["trajectory"]["gripper_hold_steps"] = tools.GRIP_MAX_ACTIONS
                plan["trajectory"]["planned_steps"] = len(ways)
                plan["trajectory"]["planned_duration_s"] = (
                    len(ways) * plan["trajectory"]["native_dt_s"]
                )
        if len(ways) > max_steps:
            return reject("planned_path_exceeds_native_step_budget")
        if native:
            status = self.primitives.env.get_status()
            if len(ways) > status.get("step_limit", 10000) - status.get("step", 0):
                result = reject("planned_path_exceeds_native_step_budget")
                result["planned_steps"] = len(ways)
                return result
        checked = [
            "both_arm_ik",
            "sampled_arm_table",
            "joint_step_limit",
            "joint_fallback_swing" if used_joint_fallback else "tcp_step_limit",
        ]
        if clock is not None:
            checked.extend(["timed_joint_derivative_limits", "substep_arm_table"])
        if keep_relative:
            checked.append("sampled_relative_tcp_pose")
        plan.update(
            ways=ways, goals=goals, relative=rel, keep_commands=snap["commands"], checked=checked,
            clock=clock,
        )
        if preview:
            return dict(accepted=True, preview=True, status="planned", executed_steps=0,
                        planned_steps=len(ways), planned_path=plan["planned_path"],
                        trajectory=plan.get("trajectory"),
                        remaining_episode_steps=(
                            self.primitives.env.get_status().get("step_limit", 10000)
                            - (snap["native_steps"] or 0)),
                        checked=checked, replan_before_dispatch=True)
        # Only one in-flight primitive, protected by the host's normal dispatch lock.
        self.plans.clear()
        self.plans[plan["id"]] = plan
        return dict(id=plan["id"], accepted=True, planned_path=plan["planned_path"])

    @staticmethod
    def _relative_error(poses, relative, limits):
        actual = np.linalg.inv(poses["left"]) @ poses["right"]
        dist, angle = _errors(actual, relative)
        return dist > limits["relative_position"] or angle > limits["relative_angle"]

    def _report(self, plan, status, reason=None):
        snap = self._snapshot()
        arms = {}
        for a in kin.ARMS:
            dist, angle = _errors(snap["poses"][a], plan["goals"][a])
            arms[a] = dict(
                position_error_m=dist,
                orientation_error_deg=angle,
                reached=(
                    dist <= plan["limits"]["position"] and angle <= plan["limits"]["orientation"]
                ),
                **self._view(snap)[a],
            )
        # A checkpoint may pause before a bounded correction of its current
        # target. Report that honestly; arm.reached always refers to FINAL goals.
        plan["last_status"], plan["last_reason"] = status, reason
        return dict(
            plan_id=plan["id"],
            status=status,
            stop_reason=reason,
            reached=status in ("completed", "checkpoint") and not plan["correcting"],
            note=reason,
            executed_steps=plan["steps"],
            arms=arms,
            observation=dict(observation_id=uuid.uuid4().hex, **self._view(snap)),
            remaining_steps=max(0, plan["max_steps"] - plan["steps"]),
            recoverable=self._recoverable(reason),
            checked=plan["checked"],
            planned_path=plan.get("planned_path"),
            trajectory=plan.get("trajectory"),
            tracking=plan["tracker"].summary(),
        )

    def execute(self, plan_id):
        plan = self.plans.get(plan_id)
        if plan is None:
            return dict(
                status="rejected",
                stop_reason="unknown_or_superseded_plan",
                reached=False,
                executed_steps=0,
            )
        if plan["stopped"] or not self._same(plan["expected"]):
            plan["stopped"] = True
            return self._report(
                plan, "partial" if plan["steps"] else "rejected", "consumed_or_stale_plan"
            )
        reason = self._reason(plan)
        if reason:
            plan["stopped"] = True
            return self._report(plan, "partial" if plan["steps"] else "rejected", reason)
        try:
            for _ in range(plan["checkpoint_steps"]):
                reason = self._reason(plan)
                if reason:
                    break
                self.primitives._check_cancelled()
                guard = plan.get("guard")
                if guard and (reason := guard(self._snapshot())):
                    break
                way = plan["ways"][plan["cursor"]]
                state = tools._obs(self.primitives)["state"]
                before = {a: tools._pose_of(a, tools._joints(state, a)) for a in kin.ARMS}
                grips = {
                    a: {"keep": plan["keep_commands"][a], "close": 0.0, "open": 1.0}[v]
                    for a, v in way["grippers"].items()
                }
                # Reserve before native dispatch. An exception may occur AFTER movement;
                # it never refunds a physical action.
                plan["steps"] += 1
                block = way.get("trajectory")
                if plan["correcting"]:
                    plan["tracker"].corrected()
                    block = (trajectory.hold(way["joints"], plan["clock"])
                             if plan["clock"] else None)
                plan["dispatch_pending"] = True
                state, status = tools._act(
                    self.primitives, state, way["joints"], grips, block
                )
                plan["dispatch_pending"] = False
                snap = self._snapshot()
                for arm in kin.ARMS:
                    # A recoverable tracking miss must not hide a hard invariant
                    # violation on either arm or in their relative transform.
                    if kin.clearance_violations(arm, snap["joints"][arm]):
                        reason = f"{arm}: measured_table_clearance"
                        break
                if (
                    not reason
                    and plan["keep_relative"]
                    and self._relative_error(snap["poses"], plan["relative"], plan["limits"])
                ):
                    reason = "relative_pose_constraint"
                if not reason and tools._spent(status):
                    reason = tools._episode_stop_reason(status)
                if not reason and guard:
                    reason = guard(snap)
                if not reason:
                    arrival = plan["cursor"] >= plan["motion_steps"] - 1
                    bounds = plan["limits"]
                    # Timing lag inside the tracking envelope is not an arrival
                    # failure. At rest, enforce the requested goal accuracy and
                    # allow only bounded, progressing holds of that same target.
                    limits = TrackingLimits(
                        bounds["position"] if arrival else bounds["tracking_position"],
                        bounds["orientation"] if arrival else bounds["tracking_orientation"],
                        position_ceiling=max(2 * bounds["position"], bounds["tracking_position"])
                        if arrival else None,
                        orientation_ceiling=max(2 * bounds["orientation"],
                                                bounds["tracking_orientation"])
                        if arrival else None,
                    )
                    decision = plan["tracker"].assess(
                        before, snap["poses"],
                        (plan["goals"] if arrival else
                         {a: tools._pose_of(a, way["joints"][a]) for a in kin.ARMS}), limits,
                        can_correct=(plan["clock"] is None or arrival),
                        spare_steps=(min(plan["max_steps"] - plan["steps"],
                                         status.get("step_limit", 10000) - status.get("step", 0))
                                     - (len(plan["ways"]) - plan["cursor"] - 1)),
                    )
                    plan["tracker"].last.update(
                        phase="arrival" if arrival else "path_tracking",
                        tolerances=dict(position_m=limits.position,
                                        orientation_deg=limits.orientation),
                    )
                    plan["correcting"] = decision == "correct"
                    if decision == "advance":
                        plan["cursor"] += 1
                    elif decision == "stop":
                        arm = plan["tracker"].last["missed_arms"][0]
                        reason = f"{arm}: tracking_error"
                if reason or plan["cursor"] == len(plan["ways"]):
                    break
            plan["expected"] = self._snapshot()
            complete = plan["cursor"] == len(plan["ways"]) and not reason
            if complete:
                for arm in kin.ARMS:
                    dist, angle = _errors(plan["expected"]["poses"][arm], plan["goals"][arm])
                    if dist > plan["limits"]["position"] or angle > plan["limits"]["orientation"]:
                        reason, complete = f"{arm}: final_target_not_reached", False
                # OPEN must actually finish; CLOSE may stop on an object, with no
                # claim about its identity or support quality.
                for arm, command in plan["ways"][-1]["grippers"].items():
                    if command == "open" and plan["expected"]["opening"][arm] < 0.95:
                        reason, complete = f"{arm}: gripper_open_incomplete", False
                    if command == "close" and plan["expected"]["opening"][arm] >= 0.95:
                        reason, complete = f"{arm}: gripper_close_incomplete", False
            plan["stopped"] = bool(reason or complete)
            return self._report(
                plan, "completed" if complete else "partial" if reason else "checkpoint", reason
            )
        except Exception as exc:
            plan["stopped"] = True
            plan["expected"] = self._snapshot()
            cancelled = getattr(self.primitives, "cancelled", None)
            if (plan.get("resume_on_cancel") and not plan.get("dispatch_pending")
                    and cancelled is not None and cancelled.is_set()):
                return self._report(plan, "partial", "code_interrupted")
            return self._report(
                plan,
                "partial" if plan["steps"] else "rejected",
                f"controller_error: {type(exc).__name__}: {exc}",
            )
        except BaseException:
            plan["stopped"] = True
            plan["expected"] = self._snapshot()
            raise

    @staticmethod
    def _recoverable(reason):
        if reason is None:
            return True
        return reason in ("stale_observation", "consumed_or_stale_plan") or any(
            reason.endswith(": " + name)
            for name in (
                "tracking_error",
                "final_target_not_reached",
                "gripper_open_incomplete",
                "gripper_close_incomplete",
            )
        )

    def discard(self, plan_id):
        self.plans.pop(plan_id, None)
        return {}

    def stop(self, plan_id):
        plan = self.plans.get(plan_id)
        if plan is None:
            return dict(
                status="rejected", stop_reason="unknown_or_superseded_plan", executed_steps=0
            )
        plan["stopped"] = True
        return self._report(
            plan,
            plan.get("last_status", "partial" if plan["steps"] else "rejected"),
            plan.get("last_reason") or "plan_stopped",
        )


def _dispatch(controller, method, **kwargs):
    try:
        return getattr(controller, method)(**kwargs)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": f"coordination: {exc}"}


def _read(controller, method, **kwargs):
    return _dispatch(controller, method, **kwargs)


_read._readonly = True


def handlers(primitives):
    from .robodojo_intent import handlers as intent_handlers

    controller = Controller(primitives)
    names = ("capabilities", "observe", "plan", "execute", "stop", "discard")
    return {
        **intent_handlers(primitives),
        **{f"motion_{name}": partial(_dispatch if name == "execute" else _read, controller, name)
           for name in names},
    }


def register(toolkit):
    for name, handler in handlers(toolkit._primitives).items():
        toolkit.add_tool(
            name,
            dict(
                name=name,
                description="Dual-arm coordination backend protocol",
                input_schema={"type": "object", "additionalProperties": True},
            ),
            handler,
        )
