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
from . import robodojo_kinematics as kin
from . import robodojo_tools as tools


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


class Controller:
    def __init__(self, primitives):
        self.primitives = primitives
        self.observations: dict[str, dict] = {}
        self.plans: dict[str, dict] = {}

    def capabilities(self):
        return dict(
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
            force_control=False,
            speed_control=False,
            synchronized="shared Cartesian progress; bounded measured tracking, not rigid coupling",
        )

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
        if time.monotonic() >= plan["deadline"]:
            return "motion_timeout"
        if plan["steps"] >= plan["max_steps"]:
            return "native_step_budget"
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
    ):
        started = time.monotonic()
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
        snap = self.observations.get(observation_id)
        current = self._snapshot()
        goals = {a: _matrix(targets[a]) if a in targets else current["poses"][a] for a in kin.ARMS}
        limits = dict(
            position=position_tolerance_m,
            orientation=orientation_tolerance_deg,
            relative_position=position_tolerance_m,
            relative_angle=orientation_tolerance_deg,
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
        )

        def reject(reason, *, recoverable=True):
            result = self._report(plan, "rejected", reason)
            result.update(accepted=False, recoverable=recoverable)
            return result

        if require_inter_arm_clearance:
            return reject("unsupported: inter-arm collision geometry")
        if snap is None or not self._same(snap):
            return reject("stale_observation")
        goals = {a: _matrix(targets[a]) if a in targets else snap["poses"][a] for a in kin.ARMS}
        rel = np.linalg.inv(snap["poses"]["left"]) @ snap["poses"]["right"]
        q = {a: np.asarray(snap["joints"][a], float) for a in kin.ARMS}
        poses = {a: snap["poses"][a].copy() for a in kin.ARMS}
        ways = []
        arms = kin.ARMS
        commands = dict(grippers)
        begin = {a: poses[a].copy() for a in kin.ARMS}
        counts = [1]
        for a in arms:
            distance, angle = _errors(begin[a], goals[a])
            counts.extend((math.ceil(distance / max_step_m), math.ceil(angle / max_step_deg)))
        count = max(counts)
        if count > min(1000, max_steps):
            return reject("planned_path_exceeds_native_step_budget")
        for i in range(1, count + 1):
            if time.monotonic() >= plan["deadline"]:
                return reject("motion_timeout", recoverable=False)
            self.primitives._check_cancelled()
            solved = {a: q[a].copy() for a in kin.ARMS}
            for a in arms:
                desired = np.eye(4)
                desired[:3, 3] = begin[a][:3, 3] + (goals[a][:3, 3] - begin[a][:3, 3]) * i / count
                desired[:3, :3] = kin.slerp(begin[a][:3, :3], goals[a][:3, :3], i / count)
                res = kin.solve(a, desired, q[a])
                if not res.ok or res.collides:
                    return reject(
                        f"{a}: unreachable or table collision at path fraction {i / count}"
                    )
                err_m, err_deg = _errors(tools._pose_of(a, res.q), desired)
                if err_m > position_tolerance_m or err_deg > orientation_tolerance_deg:
                    return reject(f"{a}: IK cannot meet requested tolerance")
                if max(abs(np.asarray(res.q) - q[a])) > 0.6:
                    return reject(f"{a}: singularity/IK branch jump")
                solved[a] = np.asarray(res.q)
            subdivisions = max(
                1,
                math.ceil(
                    max(float(np.max(np.abs(solved[a] - q[a]))) for a in kin.ARMS)
                    / tools.JOINT_STEP_RAD
                ),
            )
            for j in range(1, subdivisions + 1):
                point = {
                    a: (q[a] + (solved[a] - q[a]) * j / subdivisions).tolist() for a in kin.ARMS
                }
                now = {a: tools._pose_of(a, point[a]) for a in kin.ARMS}
                for a in kin.ARMS:
                    if kin.clearance_violations(a, point[a]):
                        return reject(f"{a}: interpolated path intersects table")
                    dist, angle = _errors(now[a], poses[a])
                    # IK tolerance can perturb an ideal Cartesian sample slightly.
                    if dist > max_step_m + 0.001 or angle > max_step_deg + 0.5:
                        return reject(f"{a}: interpolated path exceeds TCP step limit")
                if keep_relative and self._relative_error(now, rel, limits):
                    return reject("relative_pose_constraint: sampled path changes arm relationship")
                ways.append(dict(joints=point, grippers=dict(commands)))
                poses = now
            q = solved
        # Let each requested gripper stroke finish while both arm poses hold.
        if any(grippers[a] != "keep" for a in arms):
            for _ in range(tools.GRIP_MAX_ACTIONS):
                ways.append(
                    dict(joints={a: q[a].tolist() for a in kin.ARMS}, grippers=dict(commands))
                )
        if len(ways) > max_steps:
            return reject("planned_path_exceeds_native_step_budget")
        checked = ["both_arm_ik", "sampled_arm_table", "joint_step_limit", "tcp_step_limit"]
        if keep_relative:
            checked.append("sampled_relative_tcp_pose")
        plan.update(
            ways=ways, goals=goals, relative=rel, keep_commands=snap["commands"], checked=checked
        )
        # Only one in-flight primitive, protected by the host's normal dispatch lock.
        self.plans.clear()
        self.plans[plan["id"]] = plan
        return dict(id=plan["id"], accepted=True)

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
        # Checkpoint success is segment arrival; arm.reached refers to the FINAL goal.
        plan["last_status"], plan["last_reason"] = status, reason
        return dict(
            plan_id=plan["id"],
            status=status,
            stop_reason=reason,
            reached=status in ("completed", "checkpoint"),
            note=reason,
            executed_steps=plan["steps"],
            arms=arms,
            observation=dict(observation_id=uuid.uuid4().hex, **self._view(snap)),
            remaining_steps=max(0, plan["max_steps"] - plan["steps"]),
            recoverable=self._recoverable(reason),
            checked=plan["checked"],
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
                way = plan["ways"][plan["cursor"]]
                state = tools._obs(self.primitives)["state"]
                grips = {
                    a: {"keep": plan["keep_commands"][a], "close": 0.0, "open": 1.0}[v]
                    for a, v in way["grippers"].items()
                }
                # Reserve before native dispatch. An exception may occur AFTER movement;
                # it never refunds a physical action.
                plan["steps"] += 1
                state, status = tools._act(self.primitives, state, way["joints"], grips)
                plan["cursor"] += 1
                snap = self._snapshot()
                tracking_reason = None
                for arm in kin.ARMS:
                    # A recoverable tracking miss must not hide a hard invariant
                    # violation on either arm or in their relative transform.
                    if kin.clearance_violations(arm, snap["joints"][arm]):
                        reason = f"{arm}: measured_table_clearance"
                        break
                    dist, angle = _errors(
                        snap["poses"][arm], tools._pose_of(arm, way["joints"][arm])
                    )
                    if dist > plan["limits"]["position"] or angle > plan["limits"]["orientation"]:
                        tracking_reason = tracking_reason or f"{arm}: tracking_error"
                if (
                    not reason
                    and plan["keep_relative"]
                    and self._relative_error(snap["poses"], plan["relative"], plan["limits"])
                ):
                    reason = "relative_pose_constraint"
                if not reason and tools._spent(status):
                    reason = "episode_native_step_budget"
                reason = reason or tracking_reason
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
    controller = Controller(primitives)
    names = ("capabilities", "observe", "plan", "execute", "stop", "discard")
    return {
        f"motion_{name}": partial(_dispatch if name == "execute" else _read, controller, name)
        for name in names
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
