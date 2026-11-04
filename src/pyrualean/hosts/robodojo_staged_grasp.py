"""Explicit grasp phases with live-session progress and bounded local refinement."""

from __future__ import annotations

import copy
import json
import uuid
from functools import partial

import numpy as np

from . import robodojo_kinematics as kin
from . import robodojo_tools as tools
from .robodojo_coordination import _errors, _positive
from .robodojo_feedback import ProgressTracker
from .robodojo_local import prepare_local, rotation, vector

STAGES = ("open", "approach", "align", "close", "probe")


class GraspController:
    def __init__(self, intents):
        self.intents = intents
        self.motion = intents.motion
        self.records = {}

    def _remaining(self):
        status = self.motion.primitives.env.get_status()
        return max(0, status.get("step_limit", 10000) - status.get("step", 0))

    @staticmethod
    def _error(reason):
        return dict(status="rejected", reached=False, steps_used=0, needs_model=True,
                    stop_reason=reason, grasp_complete=False, retained=False)

    def _view(self, record, stage=None, *, steps=0, reason=None, status=None):
        arm = record["contract"]["arm"]
        complete = "probe" in record["completed"]
        snap = record["expected"]
        return copy.deepcopy(dict(
            grasp_id=record["id"], stage=stage, status=status or record["status"],
            reached=stage in record["completed"] if stage else complete,
            grasp_complete=complete, retained=complete, grasp_verified=False,
            needs_visual_confirmation="close" in record["completed"],
            needs_model=(status or record["status"]) in ("needs_model", "rejected"),
            stop_reason=reason or record.get("reason"), steps_used=steps,
            grasp_steps_used=record["spent"], step_limit=record["limit"],
            remaining_steps=max(0, record["limit"] - record["spent"]),
            completed_stages=[s for s in STAGES if s in record["completed"]],
            next_stage=next((s for s in STAGES if s not in record["completed"]), None),
            actual_pose=self.intents._pose(snap["poses"][arm]), opening=snap["opening"][arm],
            stage_result={k: v for k, v in (record.get("result") or {}).items() if k in (
                "status", "reached", "stop_reason", "tracking", "arms", "holding", "opening",
                "steps_used", "intent_steps_used", "partial_plan")},
            target=self.intents._pose(record["target"]),
            alignment_target=self.intents._pose(record["aligned"]),
            contract=record["contract"], process_restart_resume=False,
        ))

    def prepare(self, grasp_id, **contract):
        if not isinstance(grasp_id, str) or not grasp_id.strip() or len(grasp_id) > 160:
            raise ValueError("grasp_id must be nonempty text of at most 160 characters")
        contract = json.loads(json.dumps(contract, allow_nan=False))
        old = self.records.get(grasp_id)
        if old is not None:
            return (self._view(old) if old["contract"] == contract else
                    self._error("grasp_identity_conflict"))
        if len(self.records) >= 10000:
            raise ValueError("grasp capacity reached; completed IDs cannot be replayed")
        arm = contract["arm"]
        if arm not in kin.ARMS:
            raise ValueError("arm must be left/right")
        approach = vector(contract["approach"], "approach")
        probe = vector(contract["probe"], "probe")
        if not .01 <= np.linalg.norm(approach) <= .12:
            raise ValueError("approach must be 1..12 cm")
        if not .005 <= np.linalg.norm(probe) <= .06:
            raise ValueError("probe must be 0.5..6 cm")
        bounds = vector(contract["translation_limits"], "translation_limits")
        if np.any(bounds < 0) or np.any(bounds > .05) or not np.any(bounds):
            raise ValueError("translation limits must be 0..5 cm and not all zero")
        _positive(contract["rotation_limit_deg"], "rotation_limit_deg", 15.)
        _positive(contract["opening"], "opening", 1.)
        if contract["max_steps"] is not None:
            _positive(contract["max_steps"], "max_steps", 10000, integer=True)
        done = contract["completion"] or {"position_m": .0015, "orientation_deg": 1.}
        if not isinstance(done, dict) or set(done) - {"position_m", "orientation_deg"}:
            raise ValueError("grasp completion supports position_m and orientation_deg")
        done = {"position_m": .0015, "orientation_deg": 1., **done}
        _positive(done["position_m"], "position_m", .05)
        _positive(done["orientation_deg"], "orientation_deg", 20.)
        snap = self.motion._snapshot()
        target = self.intents._goal(contract["target"], snap["poses"][arm])
        pregrasp = target.copy()
        pregrasp[:3, 3] += approach
        if contract["pregrasp"] is not None:
            pregrasp = self.intents._goal(contract["pregrasp"], snap["poses"][arm])
        record = dict(id=grasp_id, contract=contract, target=target, pregrasp=pregrasp,
                      aligned=target.copy(), completion=done, expected=snap,
                      completed=set(), operations={}, tracker=ProgressTracker(),
                      status="ready", spent=0, limit=(contract["max_steps"]
                          if contract["max_steps"] is not None else self._remaining()))
        self.records[grasp_id] = record
        return self._view(record)

    def status(self, grasp_id):
        record = self.records.get(grasp_id)
        return self._view(record) if record else self._error("unknown_grasp_in_this_session")

    def _stroke(self, record, op, stage):
        p = self.motion.primitives
        arm = record["contract"]["arm"]
        command = record["contract"]["opening"] if stage == "open" else 0.
        snap = self.motion._snapshot()
        if stage == "open":
            if snap["commands"][arm] <= .05 and snap["opening"][arm] > tools.HOLD_MIN_OPENING:
                return dict(status="needs_model", stop_reason="gripper_may_hold_object")
            if abs(snap["opening"][arm] - command) <= .01 and op.get("ticks", 0) == 0:
                return dict(status="completed", stop_reason=None)
        elif op.get("ticks", 0) == 0:
            d, angle = _errors(snap["poses"][arm], record["aligned"])
            if (d > record["completion"]["position_m"]
                    or angle > record["completion"]["orientation_deg"]):
                return dict(status="needs_model", stop_reason="alignment_not_reached")
        while op.get("ticks", 0) < tools.GRIP_MAX_ACTIONS:
            if record["spent"] >= record["limit"] or self._remaining() <= 0:
                return dict(status="paused", stop_reason="grasp_action_limit")
            p._check_cancelled()
            state = tools._obs(p)["state"]
            before = tools._opening(state, arm)
            op["pending"] = True
            record["spent"] += 1  # Charge before dispatch, including unknown effects.
            state, _ = tools._act(p, state, {arm: tools._joints(state, arm)}, {arm: command})
            op["pending"] = False
            op["ticks"] = op.get("ticks", 0) + 1
            record["expected"] = self.motion._snapshot()
            now = tools._opening(state, arm)
            if op["ticks"] >= 2 and abs(now - before) < tools.GRIP_STILL:
                if stage == "close" and now <= tools.HOLD_MIN_OPENING:
                    return dict(status="needs_model", stop_reason="empty_grasp", opening=now)
                if stage == "open" and abs(now - command) > .01:
                    return dict(status="needs_model", stop_reason="gripper_opening_not_reached")
                return dict(status="completed", stop_reason=None, opening=now,
                            holding=stage == "close", grasp_verified=False)
        return dict(status="needs_model", stop_reason="gripper_not_settled")

    def _move(self, record, op, stage, offset, turn, refine):
        arm = record["contract"]["arm"]
        mid = op["motion_id"]
        available = max(0, record["limit"] - record["spent"])
        if available <= 0 or self._remaining() <= 0:
            return dict(status="paused", stop_reason="grasp_action_limit")
        inner = self.intents.records.get(mid)
        if inner is None:
            if refine:
                result = prepare_local(
                    self.intents, mid, arm=arm, reference=self.intents._pose(record["target"]),
                    offset=offset, frame="world", rotation_deg=turn,
                    translation_limits=record["contract"]["translation_limits"],
                    rotation_limit_deg=record["contract"]["rotation_limit_deg"],
                    holding="empty", completion=record["completion"], max_steps=available)
            else:
                goal = record["pregrasp"] if stage == "approach" else record["aligned"].copy()
                if stage == "probe":
                    # Probe FROM measured closure, not an ideal nominal position.
                    goal = record["expected"]["poses"][arm].copy()
                    goal[:3, 3] += record["contract"]["probe"]
                result = self.intents.prepare(
                    mid, targets={arm: self.intents._pose(goal)},
                    interaction="free" if stage == "approach" else "approach",
                    directions={}, holding={arm: "retain" if stage == "probe" else "empty"},
                    completion=(record["completion"] if stage == "align" else
                                {"position_m": .005 if stage == "approach" else .003,
                                 "orientation_deg": 3.}),
                    max_steps=available, reserve_steps=0, keep_relative=False)
            inner = self.intents.records.get(mid)
            if inner is None:
                return result
            inner["tracker"] = record["tracker"]
            plan = self.motion.plans.get(inner.get("plan_id"))
            if plan is not None:
                plan["tracker"] = record["tracker"]
        else:
            self.intents.extend(mid, inner["spent"] + available)
            result = self.intents.status(mid)
        before = inner["spent"]
        try:
            while result["status"] in ("checkpoint", "paused"):
                result = self.intents.execute(mid)
                if result["status"] != "checkpoint":
                    break
        finally:
            record["spent"] += inner["spent"] - before
            record["expected"] = self.motion._snapshot()
        return result

    def step(self, grasp_id, stage, *, adjustment_id=None, offset=(0., 0., 0.),
             rotation_deg=(0., 0., 0.), max_steps=None):
        record = self.records.get(grasp_id)
        if record is None:
            return self._error("unknown_grasp_in_this_session")
        if stage not in STAGES:
            raise ValueError("stage must be open, approach, align, close or probe")
        offset, turn = vector(offset, "offset"), vector(rotation_deg, "rotation_deg")
        refine = adjustment_id is not None
        if refine and (stage != "align" or not isinstance(adjustment_id, str)
                       or not adjustment_id.strip() or len(adjustment_id) > 160):
            raise ValueError("adjustment_id is a nonempty align-stage identity")
        if not refine and (np.any(offset) or np.any(turn)):
            raise ValueError("changed alignment needs an adjustment_id")
        key = (stage, adjustment_id)
        signature = (offset.tolist(), turn.tolist())
        op = record["operations"].get(key)
        if op is not None and op["signature"] != signature:
            return self._error("grasp_stage_identity_conflict")
        if op is not None and op.get("saved"):
            return dict(copy.deepcopy(op["saved"]), steps_used=0)
        if record.get("uncertain") or not self.motion._same(record["expected"]):
            return self._error("grasp_execution_uncertain_or_external_motion")
        if max_steps is not None:
            _positive(max_steps, "max_steps", 10000, integer=True)
            record["limit"] = max(record["limit"], max_steps)
        if refine and "close" in record["completed"]:
            return self._error("cannot_refine_closed_grasp")
        required = STAGES[:STAGES.index(stage)]
        if any(s not in record["completed"] for s in required):
            return self._error("grasp_stage_order")
        if (np.any(np.abs(offset) > record["contract"]["translation_limits"])
                or np.linalg.norm(turn) > record["contract"]["rotation_limit_deg"]):
            return self._error("alignment_outside_original_envelope")
        if op is None:
            op = dict(signature=signature, motion_id="grasp:" + uuid.uuid4().hex)
            record["operations"][key] = op
            if refine:
                record["completed"].discard("align")
        record["active_phase"] = key
        before = record["spent"]
        try:
            result = (self._stroke(record, op, stage) if stage in ("open", "close") else
                      self._move(record, op, stage, offset.tolist(), turn.tolist(), refine))
            record["expected"] = self.motion._snapshot()
            record["result"] = result
            record["status"] = result["status"]
            record["reason"] = result.get("stop_reason")
            if result["status"] == "completed":
                record["completed"].add(stage)
                record["status"] = "completed" if stage == "probe" else "ready"
                if refine:
                    record["aligned"] = record["target"].copy()
                    record["aligned"][:3, 3] += offset
                    record["aligned"][:3, :3] = rotation(turn) @ record["target"][:3, :3]
            out = self._view(record, stage, steps=record["spent"] - before)
            if stage in record["completed"] or result["status"] in ("needs_model", "rejected"):
                op["saved"] = copy.deepcopy(out)
            return out
        except BaseException:
            record["expected"] = self.motion._snapshot()
            inner = self.intents.records.get(op["motion_id"])
            if inner is not None:
                self.intents.pause(op["motion_id"])
            plan = self.motion.plans.get(inner.get("plan_id")) if inner else None
            record["uncertain"] = bool(op.get("pending") or (plan or {}).get("dispatch_pending"))
            record["status"] = "needs_model" if record["uncertain"] else "paused"
            record["reason"] = "grasp_execution_uncertain" if record["uncertain"] else "interrupted"
            raise


def handlers(intents, controller=None):
    controller = controller or GraspController(intents)

    def dispatch(method, **kwargs):
        try:
            return getattr(controller, method)(**kwargs)
        except (ValueError, TypeError, KeyError) as exc:
            record = controller.records.get(kwargs.get("grasp_id"))
            if record is not None and record.get("uncertain"):
                raise  # Never turn a possibly executed dispatch into a known rejection.
            return controller._error(f"invalid_grasp: {exc}")

    def readonly(method, **kwargs):
        return dispatch(method, **kwargs)

    readonly._readonly = True
    return {"grasp_prepare": partial(readonly, "prepare"),
            "grasp_status": partial(readonly, "status"),
            "grasp_step": partial(dispatch, "step")}
