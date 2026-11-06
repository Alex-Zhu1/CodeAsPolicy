"""Task-independent motion contracts over the shared geometric executor.

No object labels, benchmark names, scene coordinates or task recipes. The
executor selects mechanics; a changed goal/direction/path needs a new intent.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from functools import partial

import numpy as np

from ..motion import MotionWindow, resolve_motion
from . import robodojo_kinematics as kin
from . import robodojo_tools as tools
from .robodojo_coordination import Controller, _errors, _matrix, _positive
from .robodojo_feedback import ProgressTracker


def capabilities():
    return dict(
        interactions=["free", "approach", "contact"],
        holding=["keep", "empty", "retain"],
        completion=["tcp_pose", "obstructed_closure", "external_verification_required"],
        staged_grasp=["open", "approach", "align", "close", "probe"],
        local_alignment="frozen_observed_frame_bounded_translation_and_rotation",
        budget_policy="remaining_native_actions_by_default; resumable_resting_segments",
        executor_owns=["spacing", "speed", "arrival_corrections", "remaining_motion"],
        planner_owns=["goal", "direction", "holding_requirement", "completion", "budget",
                      "grasp_changes", "path_changes"],
        continuation="same_live_session_known_endpoint", process_restart_resume=False,
        operation_protocol="pyrualean.operation.v1",
        operation_methods=["operation_status", "resume_operation"],
    )


def _vector(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must contain three finite numbers")
    return np.asarray(value, float)


class IntentController:
    def __init__(self, primitives):
        self.motion = Controller(primitives)
        self.records = {}

    def _goal(self, value, initial):
        if isinstance(value, (list, tuple)):
            goal = initial.copy()
            goal[:3, 3] = _vector(value, "target")
            return goal
        if not isinstance(value, dict):
            raise ValueError("target must be xyz, Pose or xyz/angles")
        if "angles" in value:
            if set(value) != {"xyz", "angles"}:
                raise ValueError("angle target needs only xyz and angles")
            goal = np.eye(4)
            goal[:3, 3] = _vector(value["xyz"], "target xyz")
            goal[:3, :3] = kin.orientation(*_vector(value["angles"], "target angles"))
            return goal
        return _matrix(value)

    @staticmethod
    def _pose(matrix):
        return dict(xyz=matrix[:3, 3].tolist(), quat=list(kin.quat_wxyz(matrix[:3, :3])))

    def _validate(self, intent_id, contract):
        if not isinstance(intent_id, str) or not intent_id.strip() or len(intent_id) > 160:
            raise ValueError("intent_id must be nonempty text of at most 160 characters")
        if contract["interaction"] not in ("free", "approach", "contact"):
            raise ValueError("interaction must be free, approach or contact")
        targets = contract["targets"]
        if not isinstance(targets, dict) or not targets or set(targets) - set(kin.ARMS):
            raise ValueError("targets must name at least one of left/right")
        for field in ("holding", "directions"):
            if not isinstance(contract[field], dict) or set(contract[field]) - set(kin.ARMS):
                raise ValueError(f"{field} must map left/right")
        if any(v not in ("keep", "empty", "retain") for v in contract["holding"].values()):
            raise ValueError("holding must be keep, empty or retain")
        for direction in contract["directions"].values():
            if np.linalg.norm(_vector(direction, "direction")) < 1e-9:
                raise ValueError("direction must be nonzero")
        done = contract["completion"]
        if not isinstance(done, dict) or set(done) - {"position_m", "orientation_deg", "verify"}:
            raise ValueError("completion supports position_m, orientation_deg and verify")
        for key, upper in (("position_m", .05), ("orientation_deg", 20)):
            if key in done:
                _positive(done[key], key, upper)
        if "verify" in done and (not isinstance(done["verify"], str)
                                  or not done["verify"].strip() or len(done["verify"]) > 1000):
            raise ValueError("verify must be nonempty text of at most 1000 characters")
        if contract["max_steps"] is not None:
            _positive(contract["max_steps"], "max_steps", 10000, integer=True)
        reserve = contract["reserve_steps"]
        if type(reserve) is not int or not 0 <= reserve <= 10000:
            raise ValueError("reserve_steps must be an integer between 0 and 10000")
        if type(contract["keep_relative"]) is not bool:
            raise ValueError("keep_relative must be bool")

    def _policy(self, record):
        contract = record["contract"]
        mode = contract["interaction"]
        empty = all(contract["holding"].get(a) == "empty" for a in contract["targets"])
        far = "transit" if empty else "carry"
        precise = record["position_m"] < .004 or record["orientation_deg"] < 2.
        near = "precision" if precise else "approach"
        if mode == "free" and not precise:
            profile = resolve_motion("native" if empty else far)
            spacing = .02
        elif mode in ("free", "approach"):
            # Region scales with the requested accuracy, not an object's label.
            profile = MotionWindow(resolve_motion(far), resolve_motion(near),
                                   distance_m=min(.05, max(.01, 4 * record["position_m"])))
            spacing = .01
        else:
            profile = resolve_motion(near)
            spacing = .004 if precise else .01
        # Servo tracking during travel and accuracy at the final resting target
        # serve different purposes. These bounds follow interaction, not a task
        # label or the requested arrival precision. Hard geometry/holding guards
        # remain independent and can stop motion inside this envelope.
        tracking_m, tracking_deg = {
            "free": (.01, 5.), "approach": (.005, 3.), "contact": (.0025, 3.),
        }[mode]
        return dict(motion=profile.payload(), max_step_m=spacing, max_step_deg=5.,
                    tracking_position_tolerance_m=tracking_m,
                    tracking_orientation_tolerance_deg=tracking_deg)

    def _geometry(self, record, poses):
        local = record["contract"].get("local")
        if local:
            from .robodojo_local import violation

            reason = violation(local, poses[local["arm"]])
            if reason:
                return reason
        tolerance = record["position_m"]
        for arm, direction in record["contract"]["directions"].items():
            axis = np.asarray(direction, float)
            axis /= np.linalg.norm(axis)
            offset = poses[arm][:3, 3] - record["initial"]["poses"][arm][:3, 3]
            along = float(np.dot(offset, axis))
            length = float(np.dot(record["goals"][arm][:3, 3]
                                  - record["initial"]["poses"][arm][:3, 3], axis))
            if (np.linalg.norm(offset - along * axis) > tolerance
                    or along < -tolerance or along > length + tolerance):
                return f"{arm}: direction_constraint"
        return None

    def _guard(self, record, snapshot):
        for arm, requirement in record["contract"]["holding"].items():
            opening, command = snapshot["opening"][arm], snapshot["commands"][arm]
            if not math.isfinite(opening) or not math.isfinite(command):
                return f"{arm}: invalid_gripper_measurement"
            closed_on_something = command <= .05 and tools.HOLD_MIN_OPENING < opening < .95
            if requirement == "retain" and not closed_on_something:
                return f"{arm}: holding_requirement"
            if requirement == "empty" and closed_on_something:
                return f"{arm}: empty_hand_requirement"
        return self._geometry(record, snapshot["poses"])

    def _result(self, record, *, dispatch_steps=0):
        raw = copy.deepcopy(record.get("result") or {})
        status = record["status"]
        raw.update(
            intent_id=record["id"], status=status, reached=status == "completed",
            steps_used=dispatch_steps, intent_steps_used=record["spent"],
            remaining_steps=max(0, record["step_limit"] - record["spent"]),
            step_limit=record["step_limit"],
            needs_model=status in ("rejected", "needs_model"),
            accepted=status not in ("rejected", "needs_model"),
            needs_observation=(status == "completed"
                               and bool(record["contract"]["completion"].get("verify"))),
            verification=record["contract"]["completion"].get("verify"),
            policy=record["policy"], tracking=record["tracker"].summary(),
            holding_check="obstructed_closure_only", process_restart_resume=False,
        )
        # Read-only host calls also serve in-process clients. A returned policy
        # dictionary must never become a writable alias to a paused contract.
        return copy.deepcopy(raw)

    def _stop(self, record, reason, *, dispatch_steps=0):
        record["status"] = "needs_model" if record["spent"] else "rejected"
        record.setdefault("result", {})["stop_reason"] = reason
        return self._result(record, dispatch_steps=dispatch_steps)

    def _compile(self, record, *, preview=False):
        planner = Controller(self.motion.primitives) if preview else self.motion
        snap = self.motion._snapshot()
        reason = self._guard(record, snap)
        if reason:
            return self._stop(record, reason)
        errors = {a: _errors(snap["poses"][a], record["goals"][a]) for a in kin.ARMS}
        if all(d <= record["position_m"] and angle <= record["orientation_deg"]
               for d, angle in errors.values()):
            record["status"] = "completed"
            record["result"] = dict(stop_reason=None, observation=self.motion._view(snap),
                                    arms={a: dict(pose=self._pose(snap["poses"][a]),
                                                  opening=snap["opening"][a], reached=True,
                                                  position_error_m=errors[a][0],
                                                  orientation_error_deg=errors[a][1])
                                          for a in kin.ARMS})
            result = self._result(record)
            if preview:
                result.update(status="planned", reached=False, preview=True, planned_steps=0)
            return result
        status = self.motion.primitives.env.get_status()
        available = min(record["step_limit"] - record["spent"],
                        status.get("step_limit", 10000) - status.get("step", 0)
                        - record["contract"]["reserve_steps"])
        if tools._spent(status) or available <= 0:
            if not tools._spent(status):
                record.update(status="paused", result={"stop_reason": "allocation_spent"})
                answer = self._result(record)
                if preview:
                    answer.update(preview=True, accepted=False, planned_steps=0)
                return answer
            return self._stop(record, "intent_or_reserved_episode_budget")
        obs = planner.observe()
        arguments = dict(
            observation_id=obs["id"],
            targets={a: self._pose(g) for a, g in record["goals"].items()},
            grippers={a: "keep" for a in kin.ARMS},
            keep_relative=record["contract"]["keep_relative"],
            max_steps=available, timeout_s=3600., position_tolerance_m=record["position_m"],
            orientation_tolerance_deg=record["orientation_deg"], checkpoint_steps=4,
            require_inter_arm_clearance=False, **record["policy"],
        )
        result = planner.plan(**arguments)
        estimate = (result.get("trajectory") or {}).get("planned_steps")
        fraction = 1.
        # A timing estimate is not a reason to do nothing. Fit a shorter part
        # of the same Cartesian route, with a newly timed RESTING endpoint.
        # Do not truncate a moving spline or loosen any physical constraint.
        while (not result["accepted"] and "budget" in (result.get("stop_reason") or "")
               and fraction > 1 / 128):
            fraction /= 2
            goals = {}
            for arm, goal in record["goals"].items():
                start = snap["poses"][arm]
                part = np.eye(4)
                part[:3, 3] = start[:3, 3] + fraction * (goal[:3, 3] - start[:3, 3])
                part[:3, :3] = kin.slerp(start[:3, :3], goal[:3, :3], fraction)
                goals[arm] = part
            if all(d <= record["position_m"] and angle <= record["orientation_deg"]
                   for d, angle in (_errors(snap["poses"][a], g) for a, g in goals.items())):
                break  # No useful resting progress fits this allocation.
            arguments["targets"] = {a: self._pose(g) for a, g in goals.items()}
            result = planner.plan(**arguments)
        if not result["accepted"]:
            record["result"] = result
            if "budget" in (result.get("stop_reason") or "") and not tools._spent(status):
                # The model may have chosen a cap too small for even one useful
                # resting segment. Keep the identity extendable, not terminal.
                record["status"] = "paused"
                record["result"]["stop_reason"] = "allocation_too_small_for_progress"
                answer = self._result(record)
                if preview:
                    answer.update(preview=True, accepted=False, planned_steps=0)
                return answer
            return self._stop(record, result["stop_reason"])
        plan = planner.plans[result["id"]]
        record["partial_plan"] = fraction < 1.
        record["full_path_estimate"] = estimate
        # Keep the ORIGINAL relationship and correction allowance on continuation.
        plan["relative"] = record["relative"]
        plan["tracker"] = record["tracker"]
        plan["guard"] = partial(self._guard, record)
        plan["resume_on_cancel"] = True
        plan["episode_reserve"] = record["contract"]["reserve_steps"]
        for way in plan["ways"]:
            points = [way["joints"]]
            block = way.get("trajectory")
            if block:
                # Timed substeps concatenate left and right joint targets.
                points.extend({a: row[i * 6:(i + 1) * 6] for i, a in enumerate(kin.ARMS)}
                              for row in block["positions"])
            for joints in points:
                poses = {a: tools._pose_of(a, joints[a]) for a in kin.ARMS}
                reason = self._geometry(record, poses)
                if (not reason and record["contract"]["keep_relative"]
                        and self.motion._relative_error(poses, record["relative"], plan["limits"])):
                    reason = "relative_pose_constraint"
                if reason:
                    planner.discard(plan["id"])
                    return self._stop(record, reason)
        record.update(plan_id=plan["id"], accounted=0, expected=snap, status="checkpoint")
        if preview:
            planner.discard(plan["id"])
            return dict(self._result(record), status="planned", accepted=True, preview=True,
                        planned_steps=len(plan["ways"]), trajectory=plan.get("trajectory"),
                        partial_plan=fraction < 1., full_path_estimate=estimate,
                        segment_targets=arguments["targets"],
                        replan_before_dispatch=True)
        return self._result(record)

    def prepare(self, intent_id, *, preview=False, **contract):
        self._validate(intent_id, contract)
        if type(preview) is not bool:
            raise ValueError("preview must be bool")
        contract = json.loads(json.dumps(contract, allow_nan=False))
        previous = self.records.get(intent_id)
        if previous is not None and not preview:
            if contract != previous["contract"]:
                return dict(status="rejected", intent_id=intent_id, reached=False, steps_used=0,
                            needs_model=True, stop_reason="intent_identity_conflict")
            return self._result(previous)
        if len(self.records) >= 10000:
            raise ValueError("intent capacity reached; completed IDs cannot be evicted/replayed")
        snap = self.motion._snapshot()
        goals = dict(snap["poses"])
        goals.update({a: self._goal(v, snap["poses"][a]) for a, v in contract["targets"].items()})
        mode = contract["interaction"]
        done = contract["completion"]
        record = dict(
            id=intent_id, contract=contract, initial=snap, expected=snap, goals=goals,
            relative=np.linalg.inv(snap["poses"]["left"]) @ snap["poses"]["right"],
            position_m=done.get("position_m", .01 if mode == "free" else .005),
            orientation_deg=done.get("orientation_deg", 5. if mode == "free" else 3.),
            spent=0, tracker=ProgressTracker(), status="checkpoint",
            step_limit=(contract["max_steps"] if contract["max_steps"] is not None else
                        max(0, self.motion.primitives.env.get_status().get("step_limit", 10000)
                            - (snap["native_steps"] or 0))),
        )
        record["policy"] = self._policy(record)
        # Check the requested goal before permitting any part of its path.
        reason = self._geometry(record, goals)
        if not preview:
            self.records[intent_id] = record
        if reason:
            return self._stop(record, reason)
        try:
            return self._compile(record, preview=preview)
        except BaseException as exc:
            # Planning never dispatches. Keep a known zero-action intent when
            # cancellation/deadline interrupts preflight; resume can compile it.
            cancelled = getattr(self.motion.primitives, "cancelled", None)
            if not isinstance(exc, Exception) or (cancelled is not None and cancelled.is_set()):
                record.update(status="paused", result={"stop_reason": "planning_interrupted"})
            else:
                self._stop(record, f"planning_error: {type(exc).__name__}: {exc}")
            raise

    @staticmethod
    def _unknown(intent_id):
        return dict(intent_id=intent_id, status="rejected", reached=False, steps_used=0,
                    needs_model=True, stop_reason="unknown_intent_in_this_session")

    def status(self, intent_id):
        record = self.records.get(intent_id)
        if record is None:
            return self._unknown(intent_id)
        return dict(self._result(record), intent=copy.deepcopy(record["contract"]))

    def extend(self, intent_id, max_steps):
        """Idempotently raise the cumulative allowance; never reset spending."""
        _positive(max_steps, "max_steps", 10000, integer=True)
        record = self.records.get(intent_id)
        if record is None:
            return self._unknown(intent_id)
        if record["status"] in ("completed", "needs_model", "rejected"):
            return self._result(record)
        record["step_limit"] = max(record["step_limit"], max_steps)
        return self._result(record)

    def _account(self, record):
        plan = self.motion.plans.get(record.get("plan_id"))
        if plan is not None:
            record["spent"] += plan["steps"] - record["accounted"]
            record["accounted"] = plan["steps"]
        return plan

    def pause(self, intent_id):
        record = self.records.get(intent_id)
        if record is None:
            return self._unknown(intent_id)
        if record["status"] != "checkpoint":
            return self._result(record)
        plan = self._account(record)
        if plan is None or plan.get("dispatch_pending"):
            return self._stop(record, "execution_uncertain")
        if not self.motion._same(plan["expected"]):
            return self._stop(record, "state_changed_outside_intent")
        record["expected"] = self.motion._snapshot()
        record["status"] = "paused"
        self.motion.discard(plan["id"])
        return self._result(record)

    def execute(self, intent_id):
        record = self.records.get(intent_id)
        if record is None:
            return self._unknown(intent_id)
        if record["status"] not in ("checkpoint", "paused"):
            return self._result(record)
        if not self.motion._same(record["expected"]):
            return self._stop(record, "state_changed_outside_intent")
        status = self.motion.primitives.env.get_status()
        if (status.get("step_limit", 10000) - status.get("step", 0)
                <= record["contract"]["reserve_steps"]):
            return self._stop(record, "reserved_episode_budget")
        if record["status"] == "paused":
            result = self._compile(record)
            if result["status"] != "checkpoint":
                return result
        plan = self.motion.plans.get(record.get("plan_id"))
        if plan is None:
            return self._stop(record, "superseded_motion_plan")
        before = record["spent"]
        try:
            result = self.motion.execute(plan["id"])
        except BaseException:
            self.pause(intent_id)
            raise
        finally:
            self._account(record)
        record["result"] = result
        record["expected"] = self.motion._snapshot()
        if result["stop_reason"] == "code_interrupted":
            record["status"] = "paused"
            self.motion.discard(plan["id"])
        elif result["status"] == "completed" and record.get("partial_plan"):
            record["status"] = "paused"
            record["result"]["stop_reason"] = "segment_complete"
            record["result"]["partial_plan"] = True
            # Segment arrival is not arrival at the ORIGINAL intent goal.
            for arm, goal in record["goals"].items():
                d, angle = _errors(record["expected"]["poses"][arm], goal)
                record["result"]["arms"][arm].update(
                    position_error_m=d, orientation_error_deg=angle,
                    reached=d <= record["position_m"] and angle <= record["orientation_deg"])
            self.motion.discard(plan["id"])
        elif result["status"] in ("completed", "checkpoint"):
            record["status"] = result["status"]
        else:
            return self._stop(record, result["stop_reason"],
                              dispatch_steps=record["spent"] - before)
        return self._result(record, dispatch_steps=record["spent"] - before)


def handlers(primitives):
    controller = IntentController(primitives)

    def dispatch(method, **kwargs):
        try:
            return getattr(controller, method)(**kwargs)
        except (ValueError, TypeError, KeyError) as exc:
            record = controller.records.get(kwargs.get("intent_id"))
            plan = controller.motion.plans.get(record.get("plan_id")) if record else None
            if (plan or {}).get("dispatch_pending"):
                raise  # A possibly executed action is never a validation rejection.
            return dict(status="rejected", reached=False, steps_used=0, needs_model=True,
                        stop_reason=f"invalid_intent: {exc}")

    def readonly(method, **kwargs):
        return dispatch(method, **kwargs)

    readonly._readonly = True
    from .robodojo_local import prepare_local
    from .robodojo_operations import RobotOperations
    from .robodojo_staged_grasp import GraspController
    from .robodojo_staged_grasp import handlers as grasp_handlers

    grasps = GraspController(controller)
    registry = RobotOperations(getattr(primitives, "_operation_session_id", None)
                               or uuid.uuid4().hex, controller, grasps)
    primitives._operations = registry

    def local(**kwargs):
        try:
            return prepare_local(controller, **kwargs)
        except (ValueError, TypeError, KeyError) as exc:
            return dict(status="rejected", reached=False, steps_used=0, needs_model=True,
                        stop_reason=f"invalid_alignment: {exc}")

    local._readonly = True
    motion = {f"intent_{name}": partial(dispatch if name == "execute" else readonly, name)
              for name in ("prepare", "execute", "status", "pause", "extend")}
    motion["local_prepare"] = partial(local)
    return {**{name: registry.wrap("motion", "intent_id", handler)
               for name, handler in motion.items()},
            **{name: registry.wrap("grasp", "grasp_id", handler)
               for name, handler in grasp_handlers(controller, grasps).items()},
            **registry.handlers()}
