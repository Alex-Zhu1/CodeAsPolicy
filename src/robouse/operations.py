"""Environment-independent operation feedback and live-session continuation.

Adapters own execution state, units, constraints and evidence. This registry
projects their state; it neither replays requests nor chooses a next phase.
Action receipts remain the authority for whether a dispatch's effects are known.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

SCHEMA = "pyrualean.operation.v1"
STATES = frozenset({"ready", "running", "paused", "completed", "needs_replan",
                    "rejected", "unknown", "preview"})


@dataclass(frozen=True)
class OperationAdapter:
    describe: Callable
    resume: Callable


class OperationRegistry:
    """Typed, session-scoped references over existing domain controllers.

    describe(id, result=None) returns canonical feedback or None for an unknown
    operation. resume(id, allowance=None) continues ONLY an existing phase.
    No arguments for creating a new operation/phase are accepted by this path.
    """

    def __init__(self, session_id):
        self.session_id = session_id
        self.adapters = {}
        self.receipts = {}

    def register(self, kind, adapter):
        if kind in self.adapters:
            raise ValueError(f"operation kind already registered: {kind}")
        self.adapters[kind] = adapter

    def reference(self, kind, operation_id):
        return dict(session_id=self.session_id, kind=kind, id=operation_id)

    def _error(self, ref, reason):
        return dict(status="rejected", reached=False, steps_used=0, needs_model=True,
                    stop_reason=reason, operation=dict(
                        schema=SCHEMA, ref=deepcopy(ref), state="rejected", phase=None,
                        execution_complete=False, effect="unknown", reason=reason,
                        continuation=dict(action="reconcile", allowed=False),
                        verification=dict(status="unverified")))

    def validate(self, ref):
        if (not isinstance(ref, dict) or set(ref) != {"session_id", "kind", "id"}
                or not all(isinstance(v, str) and v.strip() for v in ref.values())):
            return "invalid_operation_reference"
        if ref["session_id"] != self.session_id:
            return "operation_session_mismatch"
        if ref["kind"] not in self.adapters:
            return "unknown_operation_kind"
        return None

    def attach(self, kind, operation_id, result=None):
        ref = self.reference(kind, operation_id)
        adapter = self.adapters[kind]
        described = adapter.describe(operation_id, result)
        if described is None:
            return self._error(ref, "unknown_operation_in_this_session")
        raw, feedback = described
        feedback = deepcopy(feedback)
        if feedback["state"] not in STATES:
            raise ValueError("adapter returned an invalid operation state")
        feedback.update(schema=SCHEMA, ref=ref)
        receipt = self.receipts.get((kind, operation_id))
        if receipt and feedback["state"] != "preview":
            feedback["receipt"] = deepcopy(receipt)
            if receipt["effect_known"] is not True:
                feedback.update(state="unknown", effect="unknown", execution_complete=False,
                                phase_complete=False)
                feedback["continuation"] = dict(action="reconcile", allowed=False)
        return dict(deepcopy(raw), operation=feedback)

    def status(self, ref):
        error = self.validate(ref)
        if error:
            return self._error(ref, error)
        return self.attach(ref["kind"], ref["id"])

    def resume(self, ref, allowance=None):
        current = self.status(ref)
        feedback = current["operation"]
        if not feedback["continuation"]["allowed"]:
            return current
        result = self.adapters[ref["kind"]].resume(ref["id"], allowance)
        return self.attach(ref["kind"], ref["id"], result)

    def record_receipt(self, kind, operation_id, receipt):
        """Link the last effectful dispatch; zero-action cached reads don't replace it."""
        self.receipts[(kind, operation_id)] = {
            k: deepcopy(receipt.get(k)) for k in
            ("action_id", "status", "effect_known", "execution_complete", "observation_id")
        }


def feedback(*, state, phase, completed, next_phase, used, limit, unit,
             reason=None, effect="known", verification=None, observation=None,
             phase_complete=False):
    """Common vocabulary. Completion is execution completion, never task proof."""
    if effect == "unknown":
        state = "unknown"
        phase_complete = False
    action = ("reconcile" if state == "unknown" else
              "resume" if state in ("running", "paused") else
              "advance" if state == "ready" else
              "replan" if state in ("needs_replan", "rejected") else
              "observe" if state == "completed" and verification else "none")
    return dict(
        state=state, phase=phase, phase_complete=phase_complete,
        execution_complete=state == "completed", effect=effect, reason=reason,
        progress=dict(completed_phases=list(completed), next_phase=next_phase),
        budget=dict(unit=unit, used=used, limit=limit, remaining=max(0, limit - used)),
        continuation=dict(action=action, allowed=action == "resume", scope="current_phase"),
        verification=dict(status="required" if verification else "not_requested",
                          condition=verification),
        observation=deepcopy(observation),
    )
