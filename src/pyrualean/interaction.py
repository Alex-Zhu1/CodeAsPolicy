"""Host-owned observation identities and action reconciliation for code policies.

The backend supplies capture and execution facts. This layer keeps detached policy
views, prevents unresolved actions from being replayed, and never promotes a visual
judgement to an execution acknowledgement.
"""

from __future__ import annotations

import copy
import time
import uuid
from contextlib import contextmanager
from typing import Any

from ._backend import ToolError, unwrap


class InteractionPause(BaseException):
    """Return control without permitting a later dependent action in this cell."""


class InteractionMixin:
    def _init_interaction(self) -> None:
        self._interaction_session_id = getattr(self._backend, "session_id", None) or uuid.uuid4().hex
        self._supports_observation_snapshots = callable(
            getattr(self._backend, "refresh_observation", None)
        )
        metadata = getattr(self._backend, "observation_metadata", None)
        try:
            has_source = bool(metadata().get("source_id")) if callable(metadata) else False
        except Exception:
            has_source = False
        self._source_verification_supported = bool(getattr(
            self._backend, "supports_source_observations", has_source
        ))
        self._observations: dict[str, dict] = {}
        self._latest_observation_id: str | None = None
        self._receipts: dict[str, dict] = {}
        self._last_action_id: str | None = None
        self._pending_actions: set[str] = set()
        self._verification_depth = 0
        self._code_context: dict = {}
        self._paused_turn: int | None = None
        self._execution_journal = None
        self._operation_id: str | None = None

    def _persist_action(self, action_id):
        if self._execution_journal is not None:
            self._execution_journal.put("action", action_id, self._receipts[action_id])

    def _remember_observation(self, envelope, *, current=True) -> None:
        meta = (envelope or {}).get("observation")
        if not isinstance(meta, dict) or not meta.get("observation_id"):
            return
        if meta.get("session_id") != self._interaction_session_id:
            raise ToolError("observation", "Observation belongs to another robot session")
        saved = copy.deepcopy(meta)
        saved["action_revision"] = self._action_revision
        saved["state"] = copy.deepcopy(envelope.get("state", {}))
        saved.setdefault("step", envelope.get("step"))
        identity = saved["observation_id"]
        # A historical read must not restamp a previous capture with today's revision.
        if current or identity not in self._observations:
            if not current:
                saved["action_revision"] = None
            self._observations[identity] = saved
        if current:
            self._latest_observation_id = identity
        while len(self._observations) > 256:
            del self._observations[next(iter(self._observations))]

    def snapshot(self, *, refresh: bool = False) -> dict:
        """Return source-bound state/image/calibration metadata, without moving.

        refresh=True asks the backend for an explicit physical read-only capture;
        unsupported backends refuse it. Re-reading a source is not a new confirmation.
        Use observation_id with verify(), and step with image()/camera_meta(). A
        returned snapshot is a detached view, not permission to execute an action.
        """
        if not self._supports_observation_snapshots:
            raise NotImplementedError("This backend does not provide source-bound snapshots")
        if refresh:
            if self._execution_budget is not None:
                self._execution_budget.check()
            _, envelope = unwrap(self._backend.refresh_observation(), "refresh_observation")
            if envelope is None:
                raise ToolError("refresh_observation", "Backend did not return an observation")
            self._remember_observation(envelope)
            self._absorb(envelope)
        else:
            self._readonly(self.STATE_TOOL, {"step": -1})
        if self._latest_observation_id is None:
            raise ToolError("snapshot", "No source observation is available")
        return copy.deepcopy(self._observations[self._latest_observation_id])

    def _resolve_observation(self, observation_id=None):
        if not self._supports_observation_snapshots:
            raise NotImplementedError("Source-bound verification is not supported by this backend")
        if observation_id is None:
            observation_id = self.snapshot()["observation_id"]
        if not isinstance(observation_id, str) or observation_id not in self._observations:
            raise ValueError("Unknown observation_id; obtain it with robo.snapshot()")
        snapshot = copy.deepcopy(self._observations[observation_id])
        if not self._observation_is_current(snapshot):
            raise ValueError("Observation is stale, unaligned, or execution is unresolved")
        images = []
        for camera, meta in snapshot.get("images", {}).items():
            if camera not in self.CAMERAS or meta.get("aligned") is not True:
                continue
            name = meta.get("artifact") or self._image_name(camera, self.SHOW_RESOLUTION)
            images.append((camera, self.artifact(name, step=snapshot["step"])))
        if not images:
            raise ValueError("Snapshot has no aligned camera images")
        if not self._observation_is_current(snapshot):
            raise ValueError("Observation changed while reading its images")
        return snapshot, images

    def _observation_is_current(self, snapshot) -> bool:
        if not isinstance(snapshot, dict) or self._pending_actions:
            return False
        saved = self._observations.get(snapshot.get("observation_id"))
        if (saved is None or snapshot.get("aligned") is not True
                or not snapshot.get("source_id") or snapshot.get("source_freshness") != "captured"):
            return False
        identity = ("session_id", "observation_id", "source_id", "action_revision", "step")
        if any(snapshot.get(key) != saved.get(key) for key in identity):
            return False
        if snapshot.get("action_revision") != self._action_revision:
            return False
        # A backend may advance outside a policy call. Compare live source identity,
        # not just the policy's action counter. This query must have no motion effect.
        metadata = (getattr(self._backend, "current_observation_source", None)
                    or getattr(self._backend, "observation_metadata", None))
        if not callable(metadata):
            return False
        current = metadata()
        return bool(
            current.get("session_id") == snapshot.get("session_id")
            and current.get("observation_id") == snapshot.get("observation_id")
            and current.get("source_id") == snapshot.get("source_id")
            and current.get("source_freshness") == "captured"
            and current.get("aligned") is True
            and current.get("native_steps") == snapshot.get("native_steps")
            and current.get("native_steps") == self._backend.env_steps()
        )

    @contextmanager
    def _verification_scope(self):
        self._verification_depth += 1
        try:
            yield
        finally:
            self._verification_depth -= 1

    def _interaction_guard(self) -> None:
        if self._execution_journal is not None:
            self._execution_journal.check()
        if self._verification_depth:
            raise RuntimeError("Physical actions are not permitted during read-only verification")
        if self._pending_actions:
            raise InteractionPause(
                "Action outcome is unresolved; call robo.reconcile(action_id) before new motion"
            )
        if self._paused_turn is not None and self._paused_turn == self._code_context.get("turn"):
            raise InteractionPause("This cell was interrupted by an unresolved action; "
                                   "submit remaining work in a new cell")

    def _new_action(self, tool, record) -> dict:
        action_id = f"{self._interaction_session_id}:action:{uuid.uuid4().hex}"
        self._last_action_id = action_id
        source = self._observations.get(self._latest_observation_id, {})
        context = dict(self._code_context, session_id=self._interaction_session_id,
                       action_id=action_id, observation_id=self._latest_observation_id,
                       source_id=source.get("source_id"),
                       source_native_steps=source.get("native_steps"),
                       action_revision=self._action_revision + 1,
                       operation_id=self._operation_id)
        receipt = dict(context, schema="pyrualean.action.v1", tool=tool,
                       arguments=record.kwargs,
                       status="running", effect_known=False, execution_complete=False)
        self._receipts[action_id] = receipt
        record.session_id = self._interaction_session_id
        record.action_id = action_id
        record.observation_id = self._latest_observation_id
        record.receipt = copy.deepcopy(receipt)
        self._persist_action(action_id)  # durable intent BEFORE any backend dispatch
        return context

    def _store_receipt(self, receipt, action_id) -> dict:
        if not isinstance(receipt, dict) or (
            receipt.get("action_id") != action_id
            or receipt.get("session_id") != self._interaction_session_id
        ):
            raise ToolError("action_receipt", "Mismatched action/session receipt; reconcile required")
        previous = self._receipts[action_id]
        saved = {**previous, **copy.deepcopy(receipt)}
        self._receipts[action_id] = saved
        resolved = (saved.get("effect_known") is True
                    and saved.get("execution_complete") is True
                    and saved.get("status") in ("completed", "partial", "rejected"))
        if resolved:
            self._pending_actions.discard(action_id)
        else:
            self._pending_actions.add(action_id)
            self._paused_turn = self._code_context.get("turn")
        self._persist_action(action_id)
        return saved

    def _unknown_action(self, action_id, error) -> None:
        receipt = self._receipts[action_id]
        receipt.update(status="unknown", effect_known=False, execution_complete=False,
                       error=f"{type(error).__name__}: {error}")
        self._pending_actions.add(action_id)
        self._paused_turn = self._code_context.get("turn")
        self._persist_action(action_id)

    def action_receipt(self, action_id: str | None = None) -> dict:
        """Read a detached action receipt; no query, retry or physical action occurs.

        Omit action_id for the last attempted action. Acknowledgement, target arrival,
        effect certainty and task success have different meanings.
        """
        action_id = action_id or self._last_action_id
        if action_id not in self._receipts:
            raise ValueError("No receipt for this action in the current session")
        return copy.deepcopy(self._receipts[action_id])

    def reconcile(self, action_id: str | None = None) -> dict:
        """Query an unresolved action without replaying it, then refresh known state.

        Only a matching backend receipt confirming effects and execution completion
        removes the execution blocker. A new image or a visual yes cannot do so.
        If the backend cannot reconcile, the action stays unknown and motion stays blocked.
        """
        action_id = action_id or self._last_action_id
        if action_id not in self._receipts:
            raise ValueError("Unknown action_id for this session")
        if self._execution_budget is not None:
            self._execution_budget.check()
        query = getattr(self._backend, "reconcile", None)
        if not callable(query):
            result = self.action_receipt(action_id)
            result["reconciliation_supported"] = False
            return result
        record = self._ledger.open("reconcile", {"action_id": action_id}, stateful=False)
        record.action_id = action_id
        record.session_id = self._interaction_session_id
        started = time.perf_counter()
        try:
            receipt = query(action_id)
            self._store_receipt(receipt, action_id)
            if action_id not in self._pending_actions and self._supports_observation_snapshots:
                # An execution boundary is not a visual predicate. Evidence stays
                # invalidated until independently verified again.
                self.snapshot()
            record.ok = True
            return self.action_receipt(action_id)
        except BaseException as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.receipt = self.action_receipt(action_id)
            record.observation_id = self._latest_observation_id
            record.elapsed_s = time.perf_counter() - started

    def _interaction_handoff(self) -> dict[str, Any]:
        pending = [self._receipt_summary(key) for key in sorted(self._pending_actions)]
        stopped = self._stop_reason or ("truncated" if self._truncated else None)
        return dict(
            schema="pyrualean.handoff.v1", session_id=self._interaction_session_id,
            observation_id=self._latest_observation_id,
            last_action=self._receipt_summary(self._last_action_id) if self._last_action_id else None,
            unresolved_actions=pending,
            next_owner="execution" if pending else "episode" if stopped else "program_or_planner",
            allowed_next=("observe_or_reconcile" if pending
                          else "finish" if stopped else "check_prerequisites"),
            new_motion_allowed=False if pending or stopped else None,
            stop_reason=stopped,
            automatic_replay=False, scene_rolled_back=False,
        )

    def _receipt_summary(self, action_id) -> dict:
        """Bounded model-facing receipt; the complete original stays in the ledger."""
        receipt = self.action_receipt(action_id)
        summary = {k: receipt[k] for k in (
            "session_id", "action_id", "tool", "status", "effect_known", "execution_complete",
            "observation_id", "stop_confirmed", "receipt_supported", "reason", "turn",
            "request_id", "operation_id",
        ) if k in receipt}
        if receipt.get("error"):
            summary["error"] = str(receipt["error"])[:300]
        native = receipt.get("native_action_ids", [])
        summary["native_action_count"] = len(native)
        summary["last_native_action_ids"] = native[-3:]
        return summary
