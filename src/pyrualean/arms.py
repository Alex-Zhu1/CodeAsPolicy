# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Tool surfaces for the code arms, served to an agent loop through MCP.

Two arms, one contract.  Each is a tiny toolkit (``get_tools_spec`` /
``execute_tool``) that an MCP server can expose to Codex:

* ``cells``  - one tool ``python(code)``: the code runs in a *persistent*
  namespace where ``robo`` lives, like a notebook or OpenAI's python tool.
  Variables and functions defined in one cell are available in the next.
* ``program`` - one tool ``run_program(code | path)``: the code is a complete
  file defining ``run(robo)``; every program starts from a fresh namespace
  (files in the workspace persist and can be imported).  ``max_programs=1``
  is the single-shot protocol.

Both arms also get ``finish(status, summary)``.  Every tool result is text
(stdout, errors, the ledger of primitive calls, the robot state) plus the two
current camera images whenever the robot moved, so the agent's per-turn
perception matches what a tool-calling agent receives after each primitive.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import functools
import hashlib
import io
import json
import math
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import strategies
from ._backend import EpisodeFinished
from ._robot import RobotBase, effective_primitives
from .coordination import CoordinationPause
from .execution import BoundedOutput, ExecutionBudget, ExecutionLimit, ExecutionLimits
from .interaction import InteractionPause
from .interaction_trace import InteractionTrace
from .perception import perception_summary
from .recovery import failure_details, result_is_error
from .runner import PolicyLoadError, load_policy, run_policy
from .sandbox import audit_source, blocked_names, make_builtins
from .stuck_recovery import StuckRecoveryConfig
from .supervision import SupervisionPause, VisualSupervisor

ARMS = ("cells", "program")
#: When camera images are attached to a tool result: after every turn that
#: moved the robot (CUA-style parity), only when the code called
#: ``robo.show()`` (the author decides when to look), or never.
IMAGE_POLICIES = ("on-motion", "on-demand", "none")
#: What a tool result contains besides the program's own output.  ``pure`` is
#: python-tool semantics: stdout, stderr, the exception, the trailing value and
#: whatever ``robo.show()`` asked for. Failures also include recovery evidence.
#: ``rich`` adds the robot state, the ledger of primitive calls and step counts.
FEEDBACK_MODES = ("pure", "rich")
# Leave room for the transport wrapper below the observed ~12k delivery cut.
# This bounds the model view only; ArmResult.result and the host audit stay full.
MAX_TEXT_CHARS = 9_000


@dataclass
class ArmResult:
    """What the MCP server hands back: ``result`` (dict) and content blocks."""

    result: dict[str, Any]
    images: list[bytes] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return result_is_error(self.result)

    @property
    def content_blocks(self) -> list[dict[str, Any]]:
        text = _feedback_text(self.result)
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for data in self.images:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        return blocks


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _decision_receipt(receipt: dict) -> dict:
    """Keep query identity and execution certainty, not the native audit trail."""
    return {key: receipt[key] for key in (
        "action_id", "tool", "status", "effect_known", "execution_complete",
        "observation_id", "source_id", "stop_confirmed", "receipt_supported", "reason", "error",
    ) if key in receipt and receipt[key] is not None}


def _decision_record(record: dict) -> dict:
    """Detach the model view from the full, append-only audit record."""
    result = dict(record)
    if isinstance(result.get("provenance"), dict):
        result["provenance"] = perception_summary(result["provenance"])
    if isinstance(result.get("summary"), dict):
        result["summary"] = _decision_record(result["summary"])
    receipt = result.get("receipt")
    if isinstance(receipt, dict):
        result["receipt"] = _decision_receipt(receipt)
    return result


def _decision_observation(observation: dict) -> dict:
    """A supervisor evidence reference, not another copy of the sensor packet."""
    return {key: observation[key] for key in (
        "schema", "session_id", "observation_id", "source_id", "step", "step_id",
        "native_steps", "action_id", "action_revision", "state_version", "captured_at",
        "recorded_at", "source_freshness", "aligned",
    ) if key in observation}


def _decision_envelope(result: dict) -> dict:
    """Remove repeated protocol bookkeeping without changing runtime controls."""
    compact = dict(result)
    receipts = result.get("action_receipts")
    if isinstance(receipts, list):
        compact["action_receipts"] = [_decision_receipt(r) for r in receipts]
    observation = result.get("observation")
    if isinstance(observation, dict):
        # Receipt identities are already above. Currency is represented by the
        # actual source and revision; the generic stale-target warning adds no
        # new evidence. Image source/time/alignment fields remain intact.
        compact["observation"] = {key: value for key, value in observation.items()
                                  if key not in ("prior_targets_may_be_stale", "state_included")}
        actions = observation.get("actions", {})
        receipt_ids = {r.get("action_id") for r in (receipts or [])}
        if isinstance(actions, dict) and all(isinstance(action_id, str) and action_id in receipt_ids
                                            for action_id in actions.get("action_ids", [])):
            compact["observation"].pop("actions", None)
    handoff = result.get("handoff")
    if isinstance(handoff, dict):
        handoff = dict(handoff)
        normal = (not handoff.get("unresolved_actions") and not handoff.get("stop_reason")
                  and handoff.get("new_motion_allowed") is None
                  and handoff.get("allowed_next") == "check_prerequisites")
        if normal:
            # Absence of a blocker is not a motion permit. Omit the normal
            # default wording instead of inventing allowed_next="execute".
            for key in ("schema", "allowed_next", "new_motion_allowed", "next_owner",
                        "stop_reason", "unresolved_actions", "automatic_replay",
                        "scene_rolled_back"):
                handoff.pop(key, None)
            if isinstance(observation, dict):
                for key, source in (("session_id", "source_session_id"),
                                    ("observation_id", "source_observation_id")):
                    if handoff.get(key) == observation.get(source):
                        handoff.pop(key, None)
            last = handoff.get("last_action")
            if last is None or any(r.get("action_id") == last.get("action_id")
                                   for r in (receipts or [])):
                handoff.pop("last_action", None)
        if isinstance(handoff.get("last_action"), dict):
            handoff["last_action"] = _decision_receipt(handoff["last_action"])
        if isinstance(handoff.get("unresolved_actions"), list):
            handoff["unresolved_actions"] = [_decision_receipt(r)
                                               for r in handoff["unresolved_actions"]]
        if handoff:
            compact["handoff"] = handoff
        else:
            compact.pop("handoff", None)
    recovery = result.get("recovery")
    if isinstance(recovery, dict):
        recovery = dict(recovery)
        if isinstance(recovery.get("last_motion"), dict):
            recovery["last_motion"] = _decision_record(recovery["last_motion"])
        compact["recovery"] = recovery
    return compact


# Keep execution controls before optional diagnostic fields when a report is huge.
_CONTROL_FIELDS = (
    "output_truncated", "execution_paused", "status", "action_id", "effect_known",
    "execution_complete", "stop_confirmed", "new_motion_allowed", "allowed_next",
    "automatic_replay", "scene_rolled_back", "paused", "reason", "type", "kind",
    "next_action", "retry_allowed", "unresolved_action_count", "action_receipt_counts",
    "handoff", "supervision", "failure", "recovery", "action_receipts", "observation",
    "blocked", "last_action", "last_motion", "unresolved_actions", "summary",
    "primitive_outcome", "receipt",
    "reached", "dist_to_target_m", "orientation_error_deg", "final_xyz", "message", "error",
)


def _bounded_detail(value: Any, budget: int) -> Any:
    """Fit structured diagnostics without discarding controls behind arbitrary keys."""
    if len(_json_text(value)) <= budget:
        return value
    if isinstance(value, dict):
        rank = {key: i for i, key in enumerate(_CONTROL_FIELDS)}
        entries = sorted(value.items(), key=lambda pair: rank.get(pair[0], len(rank)))
        result = {}
        omitted = 0
        for index, (key, item) in enumerate(entries):
            overhead = len(_json_text(str(key))) + 2
            available = budget - len(_json_text(result)) - overhead - 32
            if index >= 32 or overhead > budget // 4 or available < 32:
                omitted += 1
                continue
            share = max(32, available // min(8, len(entries) - index))
            # IDs are lookup keys, not prose. Never emit a shortened ID that a
            # subsequent action_receipt / reconcile / snapshot call cannot use.
            size = len(_json_text(item))
            identity = str(key).endswith(("_id", "_ids"))
            endpoint = key in ("summary", "primitive_outcome") and size <= 2048
            if identity or endpoint:
                if size > available:
                    omitted += 1
                    continue
                share = max(share, size)
            result[str(key)] = _bounded_detail(item, share)
        if omitted:
            result["omitted_keys"] = omitted
        return result
    if isinstance(value, (list, tuple)):
        # An early unknown receipt must not disappear behind later completed calls.
        urgent = [item for item in value if isinstance(item, dict)
                  and item.get("status") in ("unknown", "partial")]
        other = [item for item in value if not (isinstance(item, dict)
                 and item.get("status") in ("unknown", "partial"))]
        selected = (urgent + other[-12:])[:12]
        result = []
        for index, item in enumerate(selected):
            available = budget - len(_json_text(result)) - 32
            if available < 32:
                break
            result.append(_bounded_detail(item, max(32, available // (len(selected) - index))))
        omitted = len(value) - len(result)
        if omitted:
            result.append({"omitted_items": omitted})
        return result
    return _fit_text(str(value), budget)


def _fit_text(value: str, budget: int) -> str:
    """Fit the JSON-encoded string, including escaped control characters."""
    if len(_json_text(value)) <= budget:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(_json_text(_trim(value, middle))) <= budget:
            low = middle
        else:
            high = middle - 1
    return _trim(value, low) if low else ("[output omitted]" if budget >= 18 else "")


def _feedback_text(result: dict) -> str:
    compact = _decision_envelope(result)
    if "primitive_calls" in compact:
        compact["primitive_calls"] = [_decision_record(r) for r in compact["primitive_calls"]]
    if isinstance(compact.get("supervision"), dict):
        report = dict(compact["supervision"])
        if isinstance(report.get("observation"), dict):
            report["observation"] = _decision_observation(report["observation"])
        if isinstance(report.get("last_action"), dict):
            report["last_action"] = _decision_record(report["last_action"])
        compact["supervision"] = report
    text = _json_text(compact)
    if len(text) <= MAX_TEXT_CHARS:
        return text

    # Redundant audit detail must never evict a short, useful policy result.
    # The original remains in ArmResult.result and the host ledger / supervisor audit.
    compact["output_truncated"] = True
    compact["details"] = "Detailed calls and evidence remain in the host audit."
    if isinstance(compact.get("observation"), dict):
        observation = dict(compact["observation"])
        actions = dict(observation.get("actions", {}))
        if "action_ids" in actions:
            actions["action_id_count"] = len(actions["action_ids"])
            actions["action_ids"] = actions["action_ids"][-3:]
        observation["actions"] = actions
        compact["observation"] = observation
    for key in ("state", "primitive_calls"):
        if key in compact:
            if key == "primitive_calls":
                compact["primitive_call_count"] = len(compact[key])
                failed = [r for r in compact[key] if not r.get("ok", True)]
                if failed:
                    compact["failed_calls"] = failed
            del compact[key]
    if isinstance(compact.get("supervision"), dict):
        compact["supervision"] = {k: v for k, v in compact["supervision"].items()
                                  if k not in ("reports", "contract", "state", "evidence")}
    if isinstance(compact.get("recovery"), dict):
        compact["recovery"] = {k: v for k, v in compact["recovery"].items() if k != "state"}
    text = _json_text(compact)
    if len(text) <= MAX_TEXT_CHARS:
        return text

    outputs = {key: str(compact.pop(key)) for key in
               ("stdout", "stderr", "value", "exception", "error") if key in compact}
    # Retain receipt counts even in an unusually long action sequence. Unknown
    # and partial results must remain visible alongside the handoff prohibition.
    receipts = compact.get("action_receipts", [])
    if receipts:
        statuses = [str(r.get("status", "unknown")) for r in receipts]
        compact["action_receipt_counts"] = {status: statuses.count(status)
                                             for status in sorted(set(statuses))}
    handoff = compact.get("handoff")
    if isinstance(handoff, dict):
        compact["handoff"] = dict(handoff, unresolved_action_count=len(
            handoff.get("unresolved_actions", [])))
    # Reserve most of the response for outputs once repeated audit fields are
    # removed. This fallback is for oversized errors / supervision reports too.
    compact = _bounded_detail(compact, MAX_TEXT_CHARS // 2)
    # Preserve small fields first; an enormous repr or stderr cannot hide a
    # normal stdout or the exception. Account for JSON escaping, not raw length.
    ordered = sorted(outputs, key=lambda key: len(_json_text(outputs[key])))
    for index, key in enumerate(ordered):
        remaining = len(ordered) - index - 1
        budget = MAX_TEXT_CHARS - len(_json_text(compact)) - len(_json_text(key)) - 4
        compact[key] = _fit_text(outputs[key], max(32, budget - remaining * 40))
    return _json_text(compact)


def _to_png(item: Any) -> bytes:
    """Encode a numpy array, PIL image or PNG bytes as PNG bytes."""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    from PIL import Image

    if hasattr(item, "save") and hasattr(item, "mode"):
        image = item
    else:
        import numpy as np

        array = np.asarray(item)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _trim(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return (
        text[: limit // 2]
        + f"\n... [{len(text) - limit} chars omitted] ...\n"
        + text[-limit // 2 :]
    )


def _state_dict(robo: RobotBase) -> dict[str, Any]:
    try:
        return robo._summary_dict()
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        return {"error": f"state unavailable: {exc}"}


def _blocked_message(names: list[str], what: str) -> str:
    """Why a cell or program was not run (it named :data:`BLOCKED_NAMES` members)."""

    return (
        f"SandboxError: {', '.join(names)} may not be used by policy code; the {what} was not "
        "run.  Stay within the documented `robo` API."
    )


class ArmToolkit:
    """MCP-servable tool surface around one robot (a :class:`RobotBase`)."""

    def __init__(
        self,
        robo: RobotBase,
        *,
        arm: str,
        workspace: Path,
        max_programs: int | None = None,
        program_timeout_s: float | None = None,
        images: bool | str = True,
        feedback: str = "rich",
        max_turns: int | None = None,
        supervisor: VisualSupervisor | None = None,
        stuck_recovery: StuckRecoveryConfig | None = None,
        execution_limits: ExecutionLimits | None = None,
        trace: InteractionTrace | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        if supervisor is not None and arm != "cells":
            raise ValueError("Visual supervision currently supports only the cells arm")
        if stuck_recovery is not None and arm != "cells":
            raise ValueError("Stuck recovery currently supports only the cells arm")
        self.stuck_recovery = stuck_recovery
        self.stuck_recovery_events: list[dict[str, Any]] = []
        self._recovery_started = time.perf_counter()
        self.supervisor = supervisor
        robo._supervisor = supervisor
        if images is True:
            images = "on-motion"
        elif images is False:
            images = "none"
        if images not in IMAGE_POLICIES:
            raise ValueError(f"images must be one of {IMAGE_POLICIES}, got {images!r}")
        if feedback not in FEEDBACK_MODES:
            raise ValueError(f"feedback must be one of {FEEDBACK_MODES}, got {feedback!r}")
        self.feedback = feedback
        self.execution_limits = execution_limits or ExecutionLimits()
        self.trace = trace
        self.agent_attempt_id: str | None = None
        self._trace_context: dict[str, Any] = {}
        self._session_id = trace.session_id if trace is not None else uuid.uuid4().hex
        self.robo = robo
        self.arm = arm
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_programs = max_programs
        self.program_timeout_s = program_timeout_s
        #: Decision budget: at most this many `python` / `run_program` calls.
        self.max_turns = max_turns
        self.turn_budget_exhausted = False
        self.images = images
        self.finish: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._displayed: list[Any] = []
        #: Audit flags of every cell / program that raised any (host-side only).
        self.sandbox_flags: list[dict[str, Any]] = []
        self._last_flags: list[str] = []
        self._builtins = make_builtins(self.workspace)
        self._globals: dict[str, Any] = {
            "__name__": "__policy__",
            "__builtins__": self._builtins,
            "robo": robo,
            "math": math,
            "time": time,
            "display": self._display,
            "log": print,
            "strategy": strategies,
        }
        if supervisor is not None:
            self._globals["watch"] = supervisor.watch
            self._globals["verify"] = functools.partial(supervisor.verify, robo)
            self._globals["evidence"] = supervisor.read_evidence
            self._globals["defer_fact"] = supervisor.defer_fact
        try:
            import numpy as np

            self._globals["np"] = np
            self._globals["numpy"] = np
        except ImportError:  # pragma: no cover - numpy is expected at runtime
            pass
        self._cell_index = 0
        self._program_index = 0
        self._failure_counts: dict[str, int] = {}

    def _display(self, image: Any) -> None:
        """Attach an image (numpy array, PIL image or PNG bytes) to this turn's result."""
        if len(self._displayed) >= self.execution_limits.max_display_images:
            error = ExecutionLimit("output_limit", "Too many display images in one code call")
            if self.robo._execution_budget is not None:
                self.robo._execution_budget.failure = error
            raise error
        # Freeze explicit displays at the time of the call. Their provenance is
        # still unknown: a policy can display an image saved in an earlier cell.
        self._displayed.append(_to_png(image))

    # -- toolkit contract ---------------------------------------------------

    def get_tools_spec(self) -> list[dict[str, Any]]:
        finish = {
            "name": "finish",
            "description": (
                "End the episode. Call it once the task predicate has fired "
                "(a tool result showed terminated=true) or when you cannot make "
                "progress, then stop."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "failure", "stuck"]},
                    "summary": {"type": "string", "description": "one or two sentences"},
                },
                "required": ["status", "summary"],
            },
        }
        if self.stuck_recovery is not None:
            finish["description"] = self.stuck_recovery.prompt()
        if self.arm == "cells":
            main = {
                "name": "python",
                "description": (
                    "Execute Python code in a persistent session where `robo` "
                    "(the robot), `np`, `math` and `time` are defined. "
                    "Variables, functions and imports persist between calls. "
                    "Returns stdout, stderr, the exception if any, the value of a "
                    "trailing expression, the robot state, and the current camera "
                    "images whenever the robot moved."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python source"}},
                    "required": ["code"],
                },
            }
        else:
            budget = (
                f" You may call it at most {self.max_programs} time(s)."
                if self.max_programs
                else ""
            )
            main = {
                "name": "run_program",
                "description": (
                    "Run a complete Python program that defines `run(robo)`: pass "
                    "the full source as `code`, or `path` of a .py file you wrote "
                    "in the workspace. Each program starts from a fresh namespace "
                    "(workspace files persist and can be imported). Returns "
                    "stdout, stderr, the exception if any, the ledger of primitive "
                    "calls, the robot state and the current camera images." + budget
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "full program source"},
                        "path": {"type": "string", "description": "workspace-relative .py file"},
                    },
                },
            }
        main["description"] += " " + self.execution_limits.prompt()
        return [main, finish]

    def _trace_event(self, event: str, **fields: Any) -> None:
        if self.trace is not None:
            self.trace.emit(event, **{**self._trace_context, **fields})

    def execute_tool(
        self, name: str, input_dict: dict[str, Any], *,
        trace_context: dict[str, Any] | None = None,
    ) -> ArmResult:
        context = {
            key: value for key, value in (trace_context or {}).items()
            if key in {"local_call_id", "decision_id", "attempt_id", "provider_call_id",
                       "agent_attempt_id", "transport"}
        }
        context["tool"] = name
        context.setdefault("agent_attempt_id", self.agent_attempt_id)
        if self.trace is not None:
            context.setdefault("local_call_id", self.trace.new_id("tool"))
            # The MCP adapter records receipt before its async lock. Direct
            # callers reach this point before the toolkit's threading lock.
            if context.get("transport") != "mcp":
                self.trace.emit("tool_received", **context)
        with self._lock:
            self._trace_context = context
            self.robo._code_context = dict(context, host_session_id=self._session_id,
                                           turn=len(self.turns) + 1)
            try:
                self._trace_event("tool_started")
                started = time.perf_counter()
                calls_before = len(self.robo._ledger)
                steps_before = self.robo._backend.env_steps()
                try:
                    if name == "finish":
                        result = self._finish(input_dict)
                    elif self.finish is not None:
                        result = ArmResult({
                            "error": "Episode already finished; code was not executed.",
                            "episode_finished": True,
                            "finish": dict(self.finish),
                        })
                    elif name == "python" and self.arm == "cells":
                        result = self._python(str(input_dict.get("code", "")))
                    elif name == "run_program" and self.arm == "program":
                        result = self._run_program(input_dict)
                    else:
                        result = ArmResult({"error": f"unknown tool: {name}"})
                except Exception as exc:  # noqa: BLE001 - never crash the server
                    result = ArmResult({"error": f"{type(exc).__name__}: {exc}"})
                self._observe_result(result, calls_before, steps_before)
                moved_flag = any(r.stateful for r in self.robo._ledger.records[calls_before:])
                steps_after = self.robo._backend.env_steps()
                steps_delta = (
                    steps_after - steps_before
                    if steps_before is not None and steps_after is not None
                    else None
                )
                flags, self._last_flags = self._last_flags, []
                self.turns.append(
                    {
                        "index": len(self.turns),
                        "tool": name,
                        "elapsed_s": round(time.perf_counter() - started, 3),
                        "ok": not result.is_error,
                        "moved": bool(result.result.get("moved", moved_flag)),
                        "env_steps": result.result.get("env_steps", steps_delta),
                        "terminated": self.robo._reported_success(),
                        "sandbox_flags": flags,
                        **{k: result.result[k] for k in (
                            "failure", "recovery", "observation", "handoff", "action_receipts")
                           if k in result.result},
                    }
                )
                if flags:
                    self.sandbox_flags.append({"turn": len(self.turns) - 1, "flags": flags})
                if self.trace is not None:
                    self.turns[-1]["local_call_id"] = context["local_call_id"]
                observation = result.result.get("observation") or {}
                self._trace_event(
                    "tool_result_ready", outcome="error" if result.is_error else "returned",
                    observation_id=observation.get("observation_id"),
                    execution_paused=bool(result.result.get("execution_paused")),
                    episode_finished=result.result.get("episode_finished"),
                )
                return result
            finally:
                self._trace_context = {}
                self.robo._code_context = {}

    # -- tools ----------------------------------------------------------------

    def _finish(self, args: dict[str, Any]) -> ArmResult:
        if self.robo._pending_actions and args.get("status") == "success":
            return ArmResult({"error": "Cannot declare success while an action is unresolved; "
                                      "inspect its receipt and reconcile first."})
        if self.finish is not None:
            return ArmResult({"ok": True, **self.finish,
                              **({"episode_finished": True} if self.stuck_recovery else {})})
        if self.stuck_recovery is not None and args.get("status") == "stuck":
            if not str(args.get("summary", "")).strip():
                return ArmResult({"error": "Describe the confirmed blockage in summary."})
            stopped = self.robo._stop_reason or (
                "truncated" if self.robo._truncated else None
            )
            if self.robo._reported_success():
                stopped = "terminated"
            calls_left = (
                max(0, self.max_turns - self._cell_index)
                if self.max_turns is not None else None
            )
            if self.turn_budget_exhausted or calls_left == 0:
                stopped = stopped or "call_budget"
            if (self.program_timeout_s is not None
                    and time.perf_counter() - self._recovery_started >= self.program_timeout_s):
                stopped = stopped or "wall_budget"
            rounds = sum(e["event"] == "recovery_dispatched" for e in self.stuck_recovery_events)
            if rounds >= self.stuck_recovery.max_rounds:
                stopped = stopped or "recovery_round_limit"
            event = {
                "event": "stuck_finalized" if stopped else "recovery_dispatched",
                "confirmed_stuck": dict(args),
                "round": rounds if stopped else rounds + 1,
                "step": self.robo._backend.env_steps(),
                "calls_left": calls_left,
                "stop_reason": stopped,
            }
            self.stuck_recovery_events.append(event)
            with (self.workspace.parent / "stuck_recovery.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            if not stopped:
                images, cameras, evidence = [], [], []
                for camera in self.robo.CAMERAS:
                    try:
                        step = self.robo._backend.env_steps()
                        revision = self.robo._action_revision
                        images.append(self.robo._backend.artifact(
                            self.robo._image_name(camera, self.robo.SHOW_RESOLUTION)))
                        cameras.append(camera)
                        evidence.append(self._camera_evidence(camera, step, revision))
                    except Exception:
                        pass  # Missing views must not make a stuck declaration terminal.
                return ArmResult({
                    "ok": True,
                    "episode_finished": False,
                    "execution_paused": True,
                    "status": "stuck_confirmed",
                    "recovery": {
                        **event,
                        "next_action": "apply_stuck_recovery_skill",
                        "retry_allowed": True,
                        "scene_rolled_back": False,
                        "automatic_replay": False,
                        "instruction": self.stuck_recovery.instructions,
                        "state": _state_dict(self.robo),
                    },
                    "images": cameras,
                    "_image_evidence": evidence,
                }, images)
        self.finish = {
            "status": str(args.get("status", "")),
            "summary": str(args.get("summary", "")),
        }
        if self._live_success():
            self.finish["environment_terminated"] = self.robo._reported_success()
        self.robo._stop_episode(self.robo._stop_reason or "finished")
        return ArmResult({"ok": True, **self.finish,
                          **({"episode_finished": True} if self.stuck_recovery else {})})

    def _live_success(self) -> bool:
        """Whether the robot reports task success during the episode (it has ``done``).

        Without it the feedback carries no success field at all: a constant
        False would read as "not solved yet".
        """
        return "done" not in getattr(self.robo, "ABSENT_MEMBERS", ())

    def _snapshot(self) -> tuple[int, int | None]:
        return len(self.robo._ledger), self.robo._backend.env_steps()

    def _observe_result(self, output: ArmResult, calls_before: int, steps_before: int | None):
        """A common evidence envelope for cells, programs, errors and finish."""
        records = self.robo._ledger.records[calls_before:]
        motions = [r for r in records if r.stateful]
        step = self.robo._backend.env_steps()
        images = output.result.pop("_image_evidence", [])
        # Non-code results without capture metadata never claim image freshness.
        if not images:
            images = [dict(name=name, source="unknown", step_id=None, state_version=None)
                      for name in output.result.get("images", [])]
        output.result["handoff"] = self.robo._interaction_handoff()
        output.result["action_receipts"] = [self.robo._receipt_summary(r.action_id)
                                            for r in motions if r.action_id]
        output.result["observation"] = {
            "observation_id": f"{self._session_id}-{len(self.turns) + 1}",
            "observed_at": time.time(),
            "source_session_id": self.robo._interaction_session_id,
            "source_observation_id": self.robo._latest_observation_id,
            "step_id": step,
            "state_version": self.robo._action_revision,
            "state_included": "state" in output.result or "state" in output.result.get(
                "recovery", {}
            ),
            "actions": {
                "attempted": len(motions),
                "acknowledged": sum(r.ok for r in motions),
                "first_ledger_index": motions[0].index if motions else None,
                "last_ledger_index": motions[-1].index if motions else None,
                "action_ids": [r.action_id for r in motions if r.action_id],
            },
            "prior_targets_may_be_stale": bool(motions) or step != steps_before,
            "images": images,
        }

    def _camera_evidence(self, name, step_before, revision_before, source=None):
        step = self.robo._backend.env_steps()
        revision = self.robo._action_revision
        stable = step == step_before and revision == revision_before
        result = dict(name=name, source="camera", retrieved_at=time.time(),
                      step_id=step if stable else None,
                      state_version=revision if stable else None,
                      changed_during_read=not stable)
        if source is not None:
            camera = next((item for item in source.get("images", {}).values()
                           if item.get("artifact") == name), {})
            snapshot = self.robo._observations.get(source.get("observation_id"))
            try:
                current = self.robo._observation_is_current(snapshot)
            except Exception:  # A failed source check cannot establish freshness.
                current = False
            result.update(source_observation_id=source.get("observation_id"),
                          source_id=camera.get("source_id"),
                          recorded_step=source.get("step"),
                          captured_at=source.get("captured_at"),
                          aligned=camera.get("aligned", False),
                          source_freshness=source.get("source_freshness", "unverified"),
                          step_id=source.get("native_steps") if stable else None,
                          state_version=(revision if stable and current
                                         and camera.get("aligned") is True else None))
        return result

    @staticmethod
    def _output_fields(stdout: BoundedOutput, stderr: BoundedOutput):
        fields = {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
        if stdout.omitted or stderr.omitted:
            fields.update(output_truncated=True, output_omitted_chars={
                "stdout": stdout.omitted, "stderr": stderr.omitted,
            })
        return fields

    def _failure_feedback(
        self,
        before: tuple[int, int | None],
        extra: dict[str, Any],
        failure: dict[str, Any],
        code: str,
    ) -> ArmResult:
        """Return enough evidence to repair from the current scene, never replay it."""
        motions = [r for r in self.robo._ledger.records[before[0]:] if r.stateful]
        used = self._cell_index if self.arm == "cells" else self._program_index
        limits = [self.max_turns]
        if self.arm == "program":
            limits.append(self.max_programs)
        remaining = [max(0, limit - used) for limit in limits if limit is not None]
        calls_left = min(remaining) if remaining else None
        stopped = self.robo._stop_reason or ("truncated" if self.robo._truncated else None)
        if self._live_success() and self.robo._reported_success():
            stopped = "terminated"
        retry_allowed = not stopped and calls_left != 0
        fingerprint = hashlib.sha256(
            (code + repr((failure["kind"], failure["type"],
                          (failure.get("location") or {}).get("line")))).encode()
        ).hexdigest()
        count = self._failure_counts.get(fingerprint, 0) + 1
        self._failure_counts[fingerprint] = count
        next_action = "inspect_and_replan" if motions else "repair_code"
        if count >= 3:
            next_action = "change_approach_or_finish"
        if not retry_allowed:
            next_action = "finish"
        elif self.robo._pending_actions:
            next_action = "observe_and_reconcile"
        extra["failure"] = failure
        steps_after = self.robo._backend.env_steps()
        extra["recovery"] = {
            "next_action": next_action,
            "retry_allowed": bool(retry_allowed),
            "new_motion_allowed": bool(retry_allowed and not self.robo._pending_actions),
            "calls_left": calls_left,
            "stop_reason": stopped,
            "same_failure_count": count,
            "automatic_replay": False,
            "scene_rolled_back": False,
            "namespace_may_have_changed": failure["stage"] != "validation",
            "motion_attempts": len(motions),
            "env_steps": (
                steps_after - before[1]
                if steps_after is not None and before[1] is not None else None
            ),
            "completed_motion_calls": sum(r.ok for r in motions),
            "last_motion": motions[-1].to_dict() if motions else None,
            "instruction": (
                "Read the error and current state. Earlier actions and assignments are not "
                "undone; an interrupted assignment may still hold an older value. Inspect "
                "the returned camera evidence when available, then submit only the repaired "
                "remaining work. Check primitive outcomes and verify the resulting scene. "
                "Do not blindly replay this code; if the budget is exhausted, call finish."
            ),
        }
        result = self._feedback(before, extra)
        # Capture state after _feedback has collected policy-issued calls, so
        # this host observation does not masquerade as the last policy action.
        result.result["recovery"]["state"] = (
            result.result["state"] if "state" in result.result else _state_dict(self.robo)
        )
        return result

    def _feedback(self, before: tuple[int, int | None], extra: dict[str, Any]) -> ArmResult:
        calls_before, steps_before = before
        records = self.robo._ledger.records[calls_before:]
        moved = any(r.stateful for r in records)
        steps_after = self.robo._backend.env_steps()
        result: dict[str, Any] = dict(extra)
        if self.robo._coordination_feedback is not None:
            result["coordination"] = self.robo._coordination_feedback
        if self.supervisor is not None:
            result["supervision"] = self.supervisor.feedback()
            if self.supervisor.paused:
                executed = [record for record in records if record.stateful]
                result["supervision"]["last_action"] = (
                    executed[-1].to_dict() if executed else None
                )
                result["supervision"]["state"] = _state_dict(self.robo)
        if self.feedback == "pure":
            return self._attach_images(result, moved, {})
        result["primitive_calls"] = [
            {
                "tool": r.tool,
                "action_id": r.action_id,
                "observation_id": r.observation_id,
                "kwargs": r.kwargs,
                "ok": r.ok,
                "error": r.error,
                **{k: perception_summary(v) if k == "provenance" else v
                   for k, v in r.summary.items() if k != "eef_pos"},
            }
            for r in records
            if r.tool not in ("artifact",)
        ][-40:]
        result["moved"] = moved
        result["env_steps"] = (
            steps_after - steps_before
            if steps_before is not None and steps_after is not None
            else None
        )
        result["state"] = _state_dict(self.robo)
        if self._live_success():
            result["episode_done"] = self.robo._reported_success()
        return self._attach_images(result, moved, result)

    def _attach_images(
        self, result: dict[str, Any], moved: bool, _unused: dict[str, Any]
    ) -> ArmResult:
        images: list[bytes] = []
        requested = self.robo._take_show_requests()
        if self.images == "on-motion" and moved:
            wanted = list(type(self.robo).ON_MOTION_IMAGES)
        elif self.images == "on-demand":
            wanted = self.robo._show_artifacts(requested)
            if (not wanted and not self._displayed and moved
                    and (result.get("failure") or result.get("execution_paused"))):
                # An exception/pause can skip the policy's final show(). Return
                # the existing recorded views, without recapturing or moving.
                wanted = list(type(self.robo).ON_MOTION_IMAGES)
        else:
            wanted = []
        attached: list[str] = []
        evidence: list[dict[str, Any]] = []
        for name in wanted:
            try:
                step, revision = self.robo._backend.env_steps(), self.robo._action_revision
                metadata = getattr(self.robo._backend, "observation_metadata", None)
                source = metadata() if callable(metadata) else None
                data = self.robo.artifact(name, step=source["step"] if source else -1)
                image_evidence = self._camera_evidence(name, step, revision, source)
                images.append(data)
                attached.append(name)
                evidence.append(image_evidence)
            except Exception:  # noqa: BLE001 - image is optional
                continue
        displayed, self._displayed = self._displayed, []
        for index, item in enumerate(displayed):
            try:
                images.append(_to_png(item))
                attached.append(f"display_{index}")
                evidence.append(dict(name=f"display_{index}", source="display",
                                     step_id=None, state_version=None))
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                result.setdefault("display_errors", []).append(f"{type(exc).__name__}: {exc}")
        if self.supervisor is not None and self.supervisor.paused:
            # Return the exact evidence inspected by Valen, even if show() was
            # never called or the normal image policy is disabled.
            for index, data in enumerate(self.supervisor.evidence):
                images.append(data)
                name = f"supervision_{self.supervisor.evidence_cameras[index]}"
                attached.append(name)
                observation = result["supervision"].get("observation") or {}
                evidence.append(dict(name=name, source="supervision",
                                     observation_id=observation.get("observation_id"),
                                     step_id=observation.get("step_id"),
                                     captured_at=observation.get("captured_at"),
                                     state_version=None))
        if attached:
            result["images"] = attached
            result["_image_evidence"] = evidence
        return ArmResult(result, images)

    def _budget_left(self, used: int) -> bool:
        """False once the decision budget is spent; ends the episode on the robot."""
        if self.max_turns is None or used < self.max_turns:
            return True
        if not self.turn_budget_exhausted:
            self.turn_budget_exhausted = True
            self.robo._stop_episode("call_budget")
        return False

    def _python(self, code: str) -> ArmResult:
        if not self._budget_left(self._cell_index):
            return ArmResult(
                {
                    "error": "call budget exhausted: the episode is over, call finish",
                    "cell": self._cell_index,
                }
            )
        self._cell_index += 1
        self.robo._coordination_feedback = None
        index = self._cell_index
        if self.supervisor is not None:
            self.supervisor.begin_cell(index)
            self._globals["watch"] = self.supervisor.watch
            self._globals["verify"] = functools.partial(self.supervisor.verify, self.robo)
            self._globals["evidence"] = self.supervisor.read_evidence
            self._globals["defer_fact"] = self.supervisor.defer_fact
        source_path = self.workspace / f"cell_{index:03d}.py"
        source_path.write_text(code, encoding="utf-8")
        filename = str(source_path.resolve())
        self._last_flags = audit_source(code, workspace=self.workspace)
        before = self._snapshot()
        stdout = BoundedOutput(self.execution_limits.max_output_chars)
        stderr = BoundedOutput(self.execution_limits.max_output_chars)
        extra: dict[str, Any] = {"cell": index}
        self._trace_event("validation_started", cell=index, code_chars=len(code))
        blocked = blocked_names(code)
        if blocked:
            self._trace_event("validation_failed", cell=index, error_type="SandboxError")
            extra["exception"] = _blocked_message(blocked, "cell")
            return self._failure_feedback(before, extra, {
                "kind": "sandbox", "stage": "validation", "type": "SandboxError",
                "message": extra["exception"], "location": None,
            }, code)
        started = time.perf_counter()
        try:
            tree = ast.parse(code, filename=filename)
            # Compile the complete cell before executing any prefix. ast.parse
            # alone accepts e.g. a top-level return/yield that compile rejects.
            compile(tree, filename, "exec")
            tail: ast.Expr | None = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                tail = tree.body.pop()  # type: ignore[assignment]
            body_code = compile(tree, filename, "exec")
            tail_code = compile(ast.Expression(tail.value), filename, "eval") if tail else None
        except SyntaxError as exc:
            self._trace_event("validation_failed", cell=index, error_type="SyntaxError")
            extra["exception"] = f"SyntaxError: {exc}"
            return self._failure_feedback(
                before, extra, failure_details(exc, stage="validation", filename=filename), code
            )
        self._trace_event(
            "code_validated", cell=index, validation_scope="sandbox_and_complete_cell_compile",
        )
        failure = None
        execution_started = False
        interrupted = False
        try:
            with (ExecutionBudget(self.execution_limits, self.workspace, self.robo) as budget,
                  contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                self._trace_event("execution_started", cell=index)
                execution_started = True
                exec(body_code, self._globals)  # noqa: S102 - code arm
                if tail_code is not None:
                    value = eval(  # noqa: S307 - trailing expression echo
                        tail_code, self._globals,
                    )
                    if value is not None:
                        extra["value"] = _trim(repr(value), 4000)
                budget.check()
        except (SupervisionPause, CoordinationPause, InteractionPause):
            extra["execution_paused"] = True
        except EpisodeFinished as exc:
            extra["episode_finished"] = exc.reason
        except (ExecutionLimit, Exception, SystemExit) as exc:
            extra["exception"] = _trim(traceback.format_exc(), 4000)
            failure = failure_details(exc, stage="execution", filename=filename)
        except BaseException:
            interrupted = True
            raise
        finally:
            self._trace_event(
                "execution_finished" if execution_started else "execution_setup_failed", cell=index,
                outcome=("interrupted" if interrupted else "error" if failure
                         else "paused" if extra.get("execution_paused")
                         else "episode_finished" if extra.get("episode_finished") else "returned"),
            )
        extra.update(self._output_fields(stdout, stderr))
        extra["elapsed_s"] = round(time.perf_counter() - started, 2)
        if failure is not None:
            return self._failure_feedback(before, extra, failure, code)
        return self._feedback(before, extra)

    def _run_program(self, args: dict[str, Any]) -> ArmResult:
        self.robo._coordination_feedback = None
        if not self._budget_left(self._program_index):
            return ArmResult({"error": "call budget exhausted: the episode is over, call finish"})
        if self.max_programs is not None and self._program_index >= self.max_programs:
            return ArmResult(
                {
                    "error": f"program budget exhausted ({self.max_programs}); call finish",
                    "state": _state_dict(self.robo),
                }
            )
        code = args.get("code")
        path = args.get("path")
        if not code and not path:
            return ArmResult({"error": "run_program needs `code` or `path`"})
        self._program_index += 1
        index = self._program_index
        if code:
            target = self.workspace / f"program_{index:02d}.py"
            target.write_text(str(code), encoding="utf-8")
        else:
            target = (self.workspace / str(path)).resolve()
            if (
                self.workspace.resolve() not in target.parents
                and target != self.workspace.resolve()
            ):
                return ArmResult({"error": "path must be inside the workspace"})
            if not target.is_file():
                return ArmResult({"error": f"no such file in workspace: {path}"})
        try:
            source = target.read_text(encoding="utf-8")
        except OSError:
            source = ""
        self._last_flags = audit_source(source, workspace=self.workspace)
        before = self._snapshot()
        extra: dict[str, Any] = {"program": index, "file": target.name}
        blocked = blocked_names(source)
        if blocked:  # before load_policy: importing the file runs its module code
            extra["exception"] = _blocked_message(blocked, "program")
            return self._failure_feedback(before, extra, {
                "kind": "sandbox", "stage": "validation", "type": "SandboxError",
                "message": extra["exception"], "location": None,
            }, source)
        stdout = BoundedOutput(self.execution_limits.max_output_chars)
        stderr = BoundedOutput(self.execution_limits.max_output_chars)
        try:
            compile(source, str(target.resolve()), "exec")
        except SyntaxError as exc:
            extra["exception"] = f"SyntaxError: {exc}"
            return self._failure_feedback(
                before, extra, failure_details(exc, stage="validation"), source
            )
        started = time.perf_counter()
        stage = "load"
        try:
            with (ExecutionBudget(self.execution_limits, self.workspace, self.robo) as budget,
                  contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                policy = load_policy(target, builtins=self._builtins,
                                     bindings={"strategy": strategies})
                stage = "execution"
                outcome = run_policy(policy, self.robo, timeout_s=self.program_timeout_s)
                if outcome.status != "timeout":
                    budget.check()
        except PolicyLoadError as exc:
            extra["exception"] = str(exc)
            extra.update(self._output_fields(stdout, stderr))
            return self._failure_feedback(
                before, extra, failure_details(
                    exc.__cause__ or exc, stage="load", filename=str(target.resolve())
                ),
                source,
            )
        except ExecutionLimit as exc:
            extra.update(self._output_fields(stdout, stderr))
            extra["elapsed_s"] = round(time.perf_counter() - started, 2)
            extra["exception"] = str(exc)
            return self._failure_feedback(
                before, extra, failure_details(exc, stage=stage, filename=str(target.resolve())),
                source,
            )
        except EpisodeFinished as exc:
            extra.update(self._output_fields(stdout, stderr))
            extra.update(status="finished", episode_finished=exc.reason)
            return self._feedback(before, extra)
        extra["status"] = outcome.status
        if outcome.status == "paused":
            extra["execution_paused"] = True
        if outcome.reason:
            extra["reason"] = outcome.reason
        if outcome.error_type:
            extra["exception"] = _trim(
                outcome.traceback or f"{outcome.error_type}: {outcome.error_message}", 4000
            )
        extra.update(self._output_fields(stdout, stderr))
        extra["elapsed_s"] = round(outcome.elapsed_s, 2)
        extra["programs_left"] = (
            None if self.max_programs is None else self.max_programs - self._program_index
        )
        if outcome.failure is not None or outcome.status == "timeout":
            failure = {
                "kind": "timeout", "stage": "execution", "type": "TimeoutError",
                "message": "Program execution deadline expired", "location": None,
            } if outcome.status == "timeout" else outcome.failure
            return self._failure_feedback(before, extra, failure, source)
        return self._feedback(before, extra)


CELLS_CONTRACT = """\
# How you control the robot

You control {robot_blurb} through ONE tool,
`python(code)`.  Each call executes your code in a persistent Python session
(like a notebook): `robo` (a `{cls}` bound to the live simulator{svc_nl}), `np`, `math`, `time`,
`log(value)` and `display(image)` are predefined, and every variable,
function or import you create stays
available in later calls: helpers you define once can be called directly in
any later call, without re-defining or re-importing them.  {feedback_note}
{image_note}
The session is sandboxed: you may import `numpy`, `math`, `time`, `json`,
`re`, `PIL` and other pure-computation modules, but not `os`, `sys`,
`subprocess`, `pathlib` or networking, and `open` only works inside your
workspace; code that reaches for Python internals (`__globals__`,
`__subclasses__`, ...) is refused.

# How the episode is scored

{scoring_head}
{ending_note}
{scoring_tail}

Your working directory is a scratch workspace: you may keep notes or helper
files there.  Do not read files outside it and do not use the shell to reach
the simulator; every action goes through `python`.
"""

PROGRAM_CONTRACT = """\
# How you control the robot

You control {robot_blurb} through ONE tool,
`run_program`.  Each call runs a complete Python program that defines
`run(robo)`: pass the whole source as `code`, or write a `.py` file into your
working directory and pass its `path`.  `robo` is a `{cls}` bound to the
live simulator{svc}.  Every
program starts from a fresh namespace (files in the workspace persist and can
be imported); no language model runs while a program executes, so every
decision inside it must be made in code from the return values documented
below.  The tool returns the program's stdout, stderr, the exception if any,
{feedback_note}  {image_note}
{program_budget}

# How the episode is scored

{scoring_head}
{ending_note}
  A program that raises stops there; the episode continues with the robot
  where it stopped.
{scoring_tail}

Programs run in a sandbox: they may import `math`, `time`, `numpy`, `json`,
`re`, `PIL` and other pure-computation modules, but not `os`, `sys`,
`subprocess`, `pathlib` or networking, and `open` only works inside the
workspace; code that reaches for Python internals (`__globals__`,
`__subclasses__`, ...) is refused; keep them self-contained.
Do not read files outside the workspace and do not use the shell to reach the
simulator; every action goes through `run_program`.
"""


IMAGE_NOTES = {
    "on-motion": (
        "Whenever the robot moved during the call, the two current camera "
        "images (agentview and wrist) are attached as well.  Write short cells "
        "when you need to look before deciding, and loops when you do not."
    ),
    "on-demand": (
        "Normal calls attach images only on request: call {show_examples} "
        "to see a current camera view, or "
        "`display(image)` to show any image array you built (a crop, an "
        "overlay, a picture you kept in memory), when you need to look before "
        "writing the next turn.  Prefer cells that carry out several steps with "
        "in-code checks (segment, back_project, gripper opening, Move.reached) "
        "and look only when a decision depends on the picture."
    ),
    "none": ("Camera images are never attached; rely on the returned state and results."),
}


FEEDBACK_NOTES = {
    "cells": {
        "pure": (
            "The tool returns exactly what your code produced: stdout, stderr, "
            "the exception if any, and the value of a trailing expression.  "
            "On a successful call, print what you want to know (for example "
            "`print(robo.state())` or the result of a motion call). Execution "
            "failures also include the host's failure and recovery report."
            " Every reply adds observation IDs, action revisions and image provenance."
        ),
        "rich": (
            "The tool returns stdout, stderr, the exception if any, the value of "
            "a trailing expression, the ledger of primitive calls made during "
            "the call and the robot state."
        ),
    },
    "program": {
        "pure": (
            "The tool returns exactly what the program produced: stdout, stderr "
            "and the exception if any. On a successful call, print what you "
            "want to know. Execution failures also include the host's failure "
            "and recovery report."
            " Every reply adds observation IDs, action revisions and image provenance."
        ),
        "rich": (
            "The tool returns the program's stdout, stderr, the exception if any, "
            "the ledger of primitive calls it made and the robot state."
        ),
    },
}


#: The scoring bullets around ``{ending_note}``, per arm.  ``live``: the
#: benchmark reports success while the episode runs.  ``final``: it judges the
#: scene the episode leaves behind and reports nothing meanwhile (a robot
#: without ``done``, i.e. RoboDojo, whose success RPent keeps from its own
#: planner too).
SCORING_NOTES = {
    "live": {
        "cells": (
            "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
            "  moment.  The simulator checks it, not you; motion results carry a\n"
            "  `terminated` field and `robo.done` reports it.",
            "- Once the task terminates, motion calls raise `EpisodeFinished`: call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
        "program": (
            "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
            "  moment while a program runs.  The simulator checks it, not you.",
            "- Once the task terminates, motion calls raise `EpisodeFinished`: call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
    },
    "final": {
        "cells": (
            "- Success is judged by the benchmark from the final scene, after the episode\n"
            "  has ended.  Nothing reports it while you work: no result carries a\n"
            "  task-success flag, so judge progress from the images and the robot state.",
            "- When the scene shows that the instruction is satisfied, call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
        "program": (
            "- Success is judged by the benchmark from the final scene, after the episode\n"
            "  has ended.  Nothing reports it while a program runs: no result carries a\n"
            "  task-success flag.",
            "- When the scene shows that the instruction is satisfied, call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
    },
}


ENDING_NOTES = {
    "wall": (
        "- The episode ends after {budget_s} s of wall clock in total (your thinking\n"
        "  included) or {max_env_steps} simulator steps; that counts as a failure."
    ),
    "calls": (
        "- The number of `{tool}` calls in an episode is limited (every call counts,\n"
        "  including calls that only look, print or display); the episode also ends\n"
        "  after {max_env_steps} simulator steps.  Reaching either limit counts as a\n"
        "  failure, so make every call do purposeful work and verify results in code."
    ),
}
BUDGET_MODES = tuple(ENDING_NOTES)

RECOVERY_NOTE = """\
# Execution, verification and recovery

Check outcomes such as `reached`, `found` and geometric error before dependent
actions. Use ordinary `if` branches and bounded corrections for results your
program can handle, continuing useful steps in the same cell. Skip dependent
actions when their condition is false; use assertions for genuinely unexpected
conditions. Source freshness does not establish object detection or task success.

When a new model judgment is needed, print the relevant result, request
`robo.show()` and end the cell. show queues images for feedback at cell end;
placing it before a move does not let Astra inspect the scene before that move.
After a motion failure/pause, feedback includes requested images or available
recorded camera evidence (unless images are disabled). Inspect provenance before
treating images as current.

`failure` identifies the error and source location; `recovery` reports execution
state and the remaining budget, including with pure feedback. Rejected syntax
does not execute; runtime failures can follow actions already executed. Nothing
rolls back. Earlier variables persist, and an interrupted assignment can retain
an old value. Inspect feedback and repair only the remaining work; do not replay
the whole cell. Submit dependent code-tool calls one at a time. Inspect host
pauses before continuing; reconcile unknown action outcomes before motion. Repeated
failure with no progress calls for a changed approach or finishing stuck.
If `retry_allowed` is false, call `finish`. Verify the scene before claiming
task success; a clean tool execution is not the benchmark's verdict.
"""
#: The same notes for a robot whose benchmark judges the final scene: a limit
#: ends the episode, and what the scene shows then is what counts.
ENDING_NOTES_FINAL = {
    "wall": (
        "- The episode ends after {budget_s} s of wall clock in total (your thinking\n"
        "  included) or {max_env_steps} simulator steps; the scene is judged as it is\n"
        "  then."
    ),
    "calls": (
        "- The number of `{tool}` calls in an episode is limited (every call counts,\n"
        "  including calls that only look, print or display); the episode also ends\n"
        "  after {max_env_steps} simulator steps.  At either limit the scene is judged\n"
        "  as it is, so make every call do purposeful work and verify results in code."
    ),
}


GUIDES_NOTE = (
    "Reference guides: the directory `guides/` in your working directory holds "
    "the three operating guides that the tool-calling version of this system "
    "reads at run time, written for its tool interface: "
    "`strict_hybrid_guide.md` (rules, localisation, command vocabulary, common "
    "failure modes), `pro_hybrid_guide.md` (LIBERO-PRO scene frames and "
    "swap-variant gotchas) and `env_calibration.md` (reachable workspace and "
    "reference heights).  The tool names used there are the `robo` methods of "
    "the same name; remarks about the runner, MCP or servers do not apply "
    "here.  Reading them is optional."
)


#: Wording of the LIBERO contract (the other robots pass their adapter's).
LIBERO_ROBOT = {
    "robot_blurb": "a simulated Franka arm in the LIBERO benchmark",
    "cls": "LiberoRobot",
    "svc": ", a frozen Pi0.5 VLA and a SAM3 segmentation service",
    "svc_nl": ", a\nfrozen Pi0.5 VLA and a SAM3 segmentation service",
    "show_examples": "`robo.show('agentview')` or `robo.show('wrist')`",
    "scoring": "live",
}


def arm_contract(
    arm: str,
    *,
    budget_s: int,
    max_env_steps: int,
    max_programs: int | None,
    images: str = "on-motion",
    feedback: str = "rich",
    guides: bool = False,
    budget_mode: str = "wall",
    robot: dict[str, str] | None = None,
) -> str:
    """Render the arm's contract; ``guides=True`` appends :data:`GUIDES_NOTE`.

    ``budget_mode="wall"`` states the wall-clock budget; ``"calls"`` states
    that the number of tool calls is limited (without the number, which the
    tool-calling arm's own prompt does not state either).  ``robot`` names
    the robot for the contract (keys of :data:`LIBERO_ROBOT`; default LIBERO);
    its ``scoring`` key picks the scoring bullets (:data:`SCORING_NOTES`).
    """

    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, got {budget_mode!r}")
    text = _arm_contract(
        arm,
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        max_programs=max_programs,
        images=images,
        feedback=feedback,
        budget_mode=budget_mode,
        robot=robot,
    )
    text = text.rstrip() + "\n\n" + RECOVERY_NOTE
    text += "\n\n" + strategies.PROMPT
    text += (
        "\nFeedback identifies observations and action receipts. Observation state_version "
        "counts motion attempts, including failures; image provenance identifies the actual "
        "capture. Explicit display images have unknown provenance; "
        "do not assume they are current.\n"
    )
    if guides:
        text = text.rstrip() + "\n\n" + GUIDES_NOTE + "\n"
    return text


def robot_words(adapter: Any, primitives: str = "full") -> dict[str, str]:
    """Contract wording for a :class:`pyrualean.robots.RobotAdapter`.

    ``primitives="no-vla"`` names the adapter's ``services_novla`` (the
    services left once the VLA is removed) instead of ``services``.
    """

    primitives = effective_primitives(adapter.robot_cls, primitives)
    key = "services" if primitives == "full" else "services_novla"
    phrase = getattr(adapter, key, "") or ""
    head, _, tail = phrase.partition(" ")
    return {
        "robot_blurb": adapter.blurb,
        "cls": adapter.robot_cls.__name__,
        "svc": f", {phrase}" if phrase else "",
        # the cells contract breaks the line after the first word of the phrase
        "svc_nl": f", {head}\n{tail}" if tail else (f", {phrase}" if phrase else ""),
        "show_examples": adapter.show_examples,
        "scoring": "live" if getattr(adapter, "live_success", True) else "final",
    }


def _arm_contract(
    arm: str,
    *,
    budget_s: int,
    max_env_steps: int,
    max_programs: int | None,
    images: str = "on-motion",
    feedback: str = "rich",
    budget_mode: str = "wall",
    robot: dict[str, str] | None = None,
) -> str:
    words = dict(LIBERO_ROBOT, **(robot or {}))
    image_note = IMAGE_NOTES[images].format(show_examples=words["show_examples"])
    feedback_note = FEEDBACK_NOTES[arm][feedback]
    ending_notes = ENDING_NOTES if words["scoring"] == "live" else ENDING_NOTES_FINAL
    ending_note = ending_notes[budget_mode].format(
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        tool="python" if arm == "cells" else "run_program",
    )
    scoring_head, scoring_tail = SCORING_NOTES[words["scoring"]][arm]
    if arm == "cells":
        return CELLS_CONTRACT.format(
            budget_s=budget_s,
            max_env_steps=max_env_steps,
            image_note=image_note,
            feedback_note=feedback_note,
            ending_note=ending_note,
            scoring_head=scoring_head,
            scoring_tail=scoring_tail,
            **words,
        )
    if max_programs == 1:
        budget = (
            "You may call it EXACTLY ONCE: there is no second attempt and no "
            "feedback before it runs, so the program must localise, verify and "
            "recover on its own."
        )
    elif max_programs:
        budget = f"You may call it at most {max_programs} times; use the feedback between calls."
    else:
        budget = "You may call it as many times as the budget allows."
    return PROGRAM_CONTRACT.format(
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        program_budget=budget,
        image_note=image_note,
        feedback_note=feedback_note,
        ending_note=ending_note,
        scoring_head=scoring_head,
        scoring_tail=scoring_tail,
        **words,
    )


__all__ = [
    "ARMS",
    "LIBERO_ROBOT",
    "BUDGET_MODES",
    "FEEDBACK_MODES",
    "GUIDES_NOTE",
    "IMAGE_POLICIES",
    "ArmResult",
    "ArmToolkit",
    "arm_contract",
    "robot_words",
]
