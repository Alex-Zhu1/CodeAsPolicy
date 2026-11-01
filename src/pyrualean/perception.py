"""Snapshot-bound perception and checks before depending on its coordinates."""

from __future__ import annotations

import copy
import uuid
from dataclasses import fields

from ._backend import ToolError


def perception_summary(provenance: dict | None) -> dict | None:
    """Decision context; full sensor provenance remains on the result and in the ledger."""
    if provenance is None:
        return None
    summary = {key: copy.deepcopy(provenance[key]) for key in (
        "observation_id", "step", "source_freshness", "aligned", "binding",
        "current_at_return", "action_revision",
    ) if key in provenance}
    geometry = provenance.get("geometry")
    if isinstance(geometry, dict):
        # Measurement assumptions are decision-critical, unlike artifact paths
        # and repeated per-camera identity fields.
        summary["geometry"] = {key: copy.deepcopy(geometry[key]) for key in (
            "basis", "world_frame", "length_unit", "depth_is_measured", "assumptions",
            "table_z_m", "agreement_view_count", "height_prior_used", "prior_scope",
            "shape_model", "multiview_match", "uncertainty_note",
        ) if key in geometry}
    return summary


class PerceptionResult:
    """Compact default display without changing the public provenance dictionary."""

    def __repr__(self):
        items = [f"{item.name}={getattr(self, item.name)!r}" for item in fields(self)
                 if item.name != "provenance"]
        if self.provenance is not None:
            # Printing several detections should expose their geometry, not
            # repeat the camera/shape-model audit for every object. The full
            # provenance remains available on the result and in the ledger.
            source = perception_summary(self.provenance)
            geometry = source.pop("geometry", None)
            if geometry:
                source["geometry"] = {key: geometry[key] for key in (
                    "basis", "depth_is_measured", "height_prior_used",
                ) if key in geometry}
            items.append(f"provenance={source!r}")
        return f"{type(self).__name__}({', '.join(items)})"


class PerceptionMixin:
    def _init_perception(self):
        self._perception_records: dict[str, dict] = {}

    def _perception_snapshot(self, observation_id=None) -> dict:
        """Resolve a registered snapshot for analysis, without requiring it to be current."""
        if observation_id is None:
            snapshot = self.snapshot()
        else:
            if not isinstance(observation_id, str) or not observation_id:
                raise ValueError("observation_id must be a nonempty string from robo.snapshot()")
            if observation_id not in self._observations:
                raise ValueError("Unknown observation_id in this robot session")
            snapshot = copy.deepcopy(self._observations[observation_id])
        state = snapshot.get("state", {})
        snapshot["image_state"] = copy.deepcopy(state.get("image_state", {}))
        snapshot["cameras"] = copy.deepcopy(state.get("cameras", {}))
        return snapshot

    def _perception_call(self, tool, kwargs, *, observation_id=None) -> dict:
        supported = getattr(self._backend, "supports_snapshot_perception", False) is True
        if observation_id is not None and not supported:
            raise NotImplementedError("This backend cannot bind perception to an observation_id")
        source = self._perception_snapshot(observation_id) if supported else None
        arguments = dict(kwargs)
        if source is not None:
            arguments.update(session_id=source["session_id"],
                             observation_id=source["observation_id"],
                             observation_step=source["step"])
        if self._execution_budget is not None:
            self._execution_budget.check()
        before = len(self._ledger)
        with self._verification_scope():
            payload = copy.deepcopy(self._readonly(tool, arguments))
        record = self._ledger.records[before]
        if source is None:
            # Old synchronous adapters have no way to prove which image they used.
            payload["provenance"] = dict(
                schema="pyrualean.perception.v1", operation=tool, binding="unsupported",
                source_freshness="unverified", current_at_return=False,
            )
        else:
            provenance = payload.get("provenance")
            fields = ("session_id", "observation_id", "source_id", "step", "native_steps")
            if not isinstance(provenance, dict) or any(
                provenance.get(key) != source.get(key) for key in fields
            ) or provenance.get("frozen") is not True:
                error = ToolError(tool, "Perception result does not match the requested snapshot")
                record.ok = False
                record.error = str(error)
                raise error
            identity = f"{self._interaction_session_id}:perception:{uuid.uuid4().hex}"
            provenance.update(perception_id=identity, action_revision=source["action_revision"],
                              current_at_return=self._perception_source_is_current(source))
            self._perception_records[identity] = {
                "source": copy.deepcopy(source), "provenance": copy.deepcopy(provenance),
            }
            while len(self._perception_records) > 256:
                del self._perception_records[next(iter(self._perception_records))]
            record.session_id = source["session_id"]
            record.observation_id = source["observation_id"]
        record.summary["provenance"] = copy.deepcopy(payload["provenance"])
        return payload

    def _perception_source_is_current(self, source) -> bool:
        try:
            return self._observation_is_current(source)
        except Exception:
            # An unavailable live-source query is not permission to use old geometry.
            return False

    def perception_is_current(self, result) -> bool:
        """Check whether this perception result still describes the current source.

        Accept a locate/segment/back_project result from this robot session.
        Recheck immediately before depending on its coordinates: a source change,
        motion attempt, unresolved action or missing provenance makes it false.
        This checks applicability, not detection quality, collision safety or task success.
        Extracted xyz tuples carry no provenance; keep the original result too.
        """
        provenance = getattr(result, "provenance", None)
        if not isinstance(provenance, dict):
            return False
        saved = self._perception_records.get(provenance.get("perception_id"))
        if saved is None or provenance != saved["provenance"]:
            return False
        return self._perception_source_is_current(saved["source"])

    def require_current(self, result):
        """Return a perception result only if its source is still current; otherwise raise.

        Check found, nonempty coordinates and note/geometry before using a
        detection; require_current(target) checks only its source, not whether
        the intended object was found or the geometry is suitable.
        Historical analysis is permitted, but its coordinates are not current targets.
        This does not pin the physical world or grant permission for a later action.
        """
        if not self.perception_is_current(result):
            raise ToolError("require_current", "Perception is stale, unverified, unresolved, "
                            "or from another session; observe and locate again")
        return result
