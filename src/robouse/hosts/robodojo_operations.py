"""Robot adapters for the common operation protocol; controllers own progress."""

from __future__ import annotations

from functools import partial

from ..operations import OperationAdapter, OperationRegistry, feedback
from .robodojo_staged_grasp import STAGES


class RobotOperations(OperationRegistry):
    def __init__(self, session_id, intents, grasps):
        super().__init__(session_id)
        self.intents, self.grasps = intents, grasps
        self.register("motion", OperationAdapter(self.motion_status, self.motion_resume))
        self.register("grasp", OperationAdapter(self.grasp_status, self.grasp_resume))

    def motion_status(self, operation_id, result=None):
        record = self.intents.records.get(operation_id)
        if record is None and result is None:
            return None
        raw = result if result is not None else self.intents.status(operation_id)
        state = {"checkpoint": "running", "needs_model": "needs_replan",
                 "planned": "preview"}.get(raw["status"], raw["status"])
        reason = raw.get("stop_reason")
        preview = raw.get("preview", False)
        if preview:
            state = "preview"
        expected = record["expected"] if record and not preview else None
        plan = self.intents.motion.plans.get(record.get("plan_id")) if record else None
        uncertain = not preview and (
            bool((plan or {}).get("dispatch_pending")) or reason == "execution_uncertain")
        # Status is non-mutating, but continuation must not advertise an old
        # endpoint as resumable after some other operation moved the robot.
        if (expected and state in ("running", "paused")
                and not self.intents.motion._same(expected)):
            state, reason = "needs_replan", "state_changed_outside_intent"
            raw = dict(raw, status="needs_model", needs_model=True, reached=False,
                       stop_reason=reason)
        return raw, feedback(
            state=state, phase=dict(name="motion", adjustment_id=None),
            phase_complete=state == "completed",
            completed=["motion"] if state == "completed" else [], next_phase=None,
            used=record["spent"] if record and not preview else raw.get("intent_steps_used", 0),
            limit=record["step_limit"] if record and not preview else raw.get("step_limit", 0),
            unit="native_action", reason=reason, effect="unknown" if uncertain else "known",
            verification=raw.get("verification"),
            observation={"native_steps": expected["native_steps"]} if expected else None,
        )

    def motion_resume(self, operation_id, allowance):
        if allowance is not None:
            self.intents.extend(operation_id, allowance)
        return self.intents.execute(operation_id)

    def grasp_status(self, operation_id, result=None):
        record = self.grasps.records.get(operation_id)
        if record is None and result is None:
            return None
        raw = result if result is not None else self.grasps.status(operation_id)
        if record is None:
            return raw, feedback(state="rejected", phase=None, completed=[], next_phase=None,
                                 used=0, limit=0, unit="native_action",
                                 reason=raw.get("stop_reason"))
        active = record.get("active_phase")
        phase = dict(name=active[0], adjustment_id=active[1]) if active else None
        # The canonical operation status comes from its live record, including
        # when an old completed stage returns its historical cached payload.
        state = {"checkpoint": "running", "needs_model": "needs_replan"}.get(
            record["status"], record["status"])
        reason = record.get("reason")
        if raw.get("status") == "rejected":
            state, reason = "rejected", raw.get("stop_reason")
        expected = record["expected"]
        handoff_ready = False
        if (state in ("running", "paused", "ready")
                and not self.grasps.motion._same(expected)):
            if ("close" in record["completed"] and "probe" not in record["completed"]
                    and self.grasps._handoff_state(record)["ok"]):
                handoff_ready = True
                state, reason = "ready", "handoff_reacquisition_required"
                raw = dict(raw, status="ready", needs_model=False,
                           stop_reason=reason, next_stage="handoff",
                           handoff_reacquisition_required=True)
            else:
                state, reason = "needs_replan", "state_changed_outside_grasp"
                raw = dict(raw, status="needs_model", needs_model=True, reached=False,
                           stop_reason=reason)
        op = record["operations"].get(active, {})
        complete = bool(op.get("saved", {}).get("reached"))
        ordered_stages = ("open", "approach", "align", "close", "handoff", "probe")
        return raw, feedback(
            state=state, phase=phase, phase_complete=complete,
            completed=[s for s in ordered_stages if s in record["completed"]],
            next_phase=("handoff" if handoff_ready else
                        next((s for s in STAGES if s not in record["completed"]), None)),
            used=record["spent"], limit=record["limit"], unit="native_action", reason=reason,
            effect="unknown" if record.get("uncertain") else "known",
            verification="intended object remains supported" if "close" in record["completed"]
            else None, observation={"native_steps": expected["native_steps"]},
        )

    def grasp_resume(self, operation_id, allowance):
        record = self.grasps.records[operation_id]
        active = record.get("active_phase")
        if active is None:
            return self.grasps.status(operation_id)
        op = record["operations"][active]
        return self.grasps.step(operation_id, active[0], adjustment_id=active[1],
                               offset=op["signature"][0], rotation_deg=op["signature"][1],
                               max_steps=allowance)

    def wrap(self, kind, id_argument, handler):
        def call(**kwargs):
            result = handler(**kwargs)
            operation_id = kwargs.get(id_argument)
            if not isinstance(operation_id, str):
                return self._error(self.reference(kind, operation_id),
                                   "invalid_operation_reference")
            return self.attach(kind, operation_id, result)

        call._readonly = getattr(handler, "_readonly", False) or getattr(
            getattr(handler, "func", None), "_readonly", False)
        return partial(call)

    def handlers(self):
        def status(operation):
            return self.status(operation)

        def resume(operation, allowance=None):
            return self.resume(operation, allowance)

        status._readonly = True
        return {"operation_status": partial(status),
                "operation_resume": partial(resume)}
