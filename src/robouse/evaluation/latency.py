"""Offline interaction timing, without importing a model client or robot host.

Host trace measurements and Codex rollout proxies deliberately have different
metric names. Missing/ambiguous intervals are reported, never filled with zero.
Only identifiers, timestamps, counts and lengths leave the parser; generated
code, tool payloads, model messages and reasoning are not copied to the report.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "pyrualean.interaction.v1"
SUMMARY_SCHEMA = "pyrualean.latency-summary.v1"
_CODE_TOOLS = {"python", "run_program"}
_HOST_STAGES = {
    "tool_queue_s": ("tool_received", "tool_started"),
    "tool_setup_s": ("tool_started", "validation_started"),
    "validation_s": ("validation_started", "code_validated"),
    "validated_to_execution_s": ("code_validated", "execution_started"),
    "execution_s": ("execution_started", "execution_finished"),
    "result_preparation_s": ("execution_finished", "tool_result_ready"),
    "feedback_serialization_s": ("tool_result_ready", "feedback_ready"),
    "tool_total_s": ("tool_received", "feedback_ready"),
}
_HOST_EVENTS = {event for pair in _HOST_STAGES.values() for event in pair}
_FAILURE_EVENTS = {"validation_failed", "execution_setup_failed", "feedback_failed"}
_CALL_EVENTS = _HOST_EVENTS | _FAILURE_EVENTS | {"tool_arguments_complete"}
_DETAILED_EVENTS = {
    "validation_started", "code_validated", "execution_started", "execution_finished",
}


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _distribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(sample["seconds"] for sample in samples)

    def percentile(q: float) -> float | None:
        if not values:
            return None
        at = (len(values) - 1) * q
        low, high = math.floor(at), math.ceil(at)
        return round(values[low] + (values[high] - values[low]) * (at - low), 6)

    return {
        "n": len(values),
        "p50_s": percentile(0.5),
        "p95_s": percentile(0.95),
        "min_s": values[0] if values else None,
        "max_s": values[-1] if values else None,
        "mean_s": round(sum(values) / len(values), 6) if values else None,
        "percentile_method": "linear_interpolation",
        # Twenty samples still do not establish a stable tail estimate. This
        # threshold only flags obviously sparse tails, not statistical power.
        "p95_stability": "insufficient_samples" if len(values) < 20 else "not_assessed",
        "samples": samples,
    }


@dataclass(frozen=True)
class _Point:
    line: int
    t: float
    clock: str
    segment: int


@dataclass
class _Call:
    session: str
    identifier: str
    tool: str | None
    points: dict[str, _Point] = field(default_factory=dict)
    event_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    invalid: bool = False
    overlap: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _missing_reasons(call: _Call, missing: set[str]) -> dict[str, dict[str, str]]:
    reasons = {name: {"reason": "unknown"} for name in sorted(missing)}
    detailed = _DETAILED_EVENTS | {"validation_failed", "execution_setup_failed"}
    if call.tool == "run_program" and not detailed.intersection(call.points):
        # The current program host emits the outer tool lifecycle only. Do not
        # mistake its documented instrumentation boundary for lost cell events.
        for name in missing & _DETAILED_EVENTS:
            reasons[name] = {"reason": "not_instrumented", "scope": "run_program"}
    for failure, unreachable in {
        "validation_failed": {"code_validated", "execution_started", "execution_finished"},
        "execution_setup_failed": {"execution_started", "execution_finished"},
        "feedback_failed": {"feedback_ready"},
    }.items():
        if failure in call.points:
            for name in missing & unreachable:
                reasons[name] = {"reason": "not_reached", "cause": failure}
    return reasons


class _Report:
    def __init__(self, path: Path, source_format: str):
        self.path = path
        self.source_format = source_format
        self.id_field = "local_call_id" if source_format == "interaction" else "call_id"
        self.counts: Counter[str] = Counter(
            lines=0, events=0, bad_json=0, error_events=0, compactions=0,
        )
        self.gaps: list[dict[str, Any]] = []
        self.samples: dict[str, list[dict[str, Any]]] = {}
        self.calls: dict[tuple[str, str], _Call] = {}
        self.active: dict[str, set[str]] = {}
        self.last_feedback: dict[str, _Call] = {}
        self.previous: dict[tuple[str, str], _Call] = {}
        self.origins: dict[str, _Point] = {}
        self.first: dict[tuple[str, str], _Point] = {}
        self.last_time: dict[tuple[str, str], float] = {}
        self.segment = 0

    def gap(self, kind: str, line: int | None = None, **fields: Any) -> None:
        self.gaps.append({"kind": kind, **({"line": line} if line else {}), **fields})

    def barrier(self) -> None:
        """A lost record might have contained a tool call. Do not bridge it."""
        self.segment += 1
        self.last_feedback.clear()

    def rows(self):
        with self.path.open(encoding="utf-8", errors="replace") as stream:
            for line, text in enumerate(stream, 1):
                self.counts["lines"] += 1
                if not text.strip():
                    continue
                try:
                    event = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    self.counts["bad_json"] += 1
                    self.gap("bad_json", line)
                    self.barrier()
                    continue
                if not isinstance(event, dict):
                    self.gap("non_object_event", line)
                    self.barrier()
                    continue
                self.counts["events"] += 1
                yield line, event

    def point(self, line: int, session: str, clock: str, t: Any) -> _Point | None:
        if not _number(t):
            self.gap("missing_or_invalid_time", line, session_id=session)
            self.barrier()
            return None
        key = (session, clock)
        if key in self.last_time and t < self.last_time[key]:
            self.gap("clock_regression", line, session_id=session, clock_domain=clock)
            self.barrier()
        self.last_time[key] = t
        return _Point(line, float(t), clock, self.segment)

    def mark(self, session: str, identifier: str, tool: str | None,
             event: str, point: _Point, meta: dict[str, Any] | None = None) -> _Call:
        key = (session, identifier)
        call = self.calls.setdefault(key, _Call(session, identifier, tool))
        if tool is not None:
            if call.tool not in (None, tool):
                call.invalid = True
                self.gap("conflicting_tool", point.line, session_id=session,
                         **{self.id_field: identifier})
            call.tool = tool
        if event in call.points:
            call.invalid = True
            self.gap("duplicate_call_stage", point.line, stage=event, session_id=session,
                     **{self.id_field: identifier})
            self.barrier()
            return call
        if meta:
            call.meta.update(meta)
        call.points[event] = point
        if event == "tool_arguments_complete":
            # A generated call is not yet executing or queued at the host.
            return call
        active = self.active.setdefault(session, set())
        if event == "tool_received":
            if active:
                call.overlap = True
                for prior in active:
                    self.calls[(session, prior)].overlap = True
                self.gap("overlapping_calls", point.line, session_id=session,
                         **{self.id_field: identifier})
                self.last_feedback.pop(session, None)
            elif session in self.last_feedback:
                self.previous[key] = self.last_feedback.pop(session)
            active.add(identifier)
            if call.tool in _CODE_TOOLS and session in self.origins:
                self.first[key] = self.origins.pop(session)
        elif event == "feedback_ready":
            if identifier not in active:
                self.gap("unmatched_feedback", point.line, session_id=session,
                         **{self.id_field: identifier})
                call.invalid = True
            active.discard(identifier)
            if not active and not call.invalid and not call.overlap:
                self.last_feedback[session] = call
        else:
            # An omitted receive event is still outstanding work; do not pair
            # a later call through it as though this invocation never existed.
            active.add(identifier)
        return call

    def interval(self, metric: str, start: _Point | None, end: _Point | None,
                 **fields: Any) -> float | None:
        if start is None or end is None:
            return None
        reason = None
        if start.clock != end.clock:
            reason = "cross_clock_interval"
        elif start.segment != end.segment:
            reason = "interrupted_interval"
        elif end.t < start.t or end.line < start.line:
            reason = "reversed_interval"
        if reason:
            self.gap(reason, end.line, metric=metric, **fields)
            return None
        seconds = round(end.t - start.t, 6)
        self.samples.setdefault(metric, []).append({
            **fields, "seconds": seconds, "clock_domain": start.clock,
            "from_line": start.line, "to_line": end.line,
        })
        return seconds

    def finish(self, stages: dict[str, tuple[str, str]]) -> dict[str, Any]:
        output = []
        proxy = self.source_format == "codex-rollout"
        received_metric = "feedback_to_next_python_s" if proxy else "feedback_to_tool_received_s"
        first_metric = "task_started_to_first_python_s" if proxy else "context_to_first_tool_s"
        for name in [*stages, received_metric, first_metric]:
            self.samples.setdefault(name, [])
        if not proxy:
            for name in ("feedback_to_code_validated_s", "context_to_first_validated_s"):
                self.samples.setdefault(name, [])
        for key, call in self.calls.items():
            identity = {"session_id": call.session, self.id_field: call.identifier,
                        "tool": call.tool}
            durations = {}
            if not call.invalid:
                for metric, (start, end) in stages.items():
                    value = self.interval(metric, call.points.get(start), call.points.get(end),
                                          **identity)
                    if value is not None:
                        durations[metric] = value
            missing = {"tool_received", "feedback_ready"} - call.points.keys()
            if not proxy and call.tool in _CODE_TOOLS:
                missing |= _HOST_EVENTS - call.points.keys()
            reasons = _missing_reasons(call, missing) if not proxy else {}
            if "feedback_ready" not in call.points:
                self.gap("missing_feedback", session_id=call.session,
                         **{self.id_field: call.identifier},
                         **(reasons["feedback_ready"] if not proxy else {}))
            output.append({**identity, **call.meta, "invalid": call.invalid,
                           "overlapping": call.overlap, "missing_events": sorted(missing),
                           **({"missing_event_reasons": reasons} if not proxy else {}),
                           "events": {name: {"t": p.t, "clock_domain": p.clock, "line": p.line,
                                             **call.event_meta.get(name, {})}
                                      for name, p in call.points.items()},
                           "durations_s": durations})
            code_tool = call.tool == "python" if proxy else call.tool in _CODE_TOOLS
            if not code_tool or call.invalid or call.overlap:
                continue
            if key in self.first:
                self.interval(first_metric, self.first[key], call.points.get("tool_received"),
                              **identity)
                if not proxy:
                    self.interval("context_to_first_validated_s", self.first[key],
                                  call.points.get("code_validated"), **identity)
            previous = self.previous.get(key)
            if previous is None or previous.invalid or previous.overlap:
                continue
            if previous.meta.get("episode_finished") is True:
                self.gap("terminal_feedback_followed_by_code", session_id=call.session,
                         **{self.id_field: call.identifier})
                continue
            pair = {"session_id": call.session, "source_tool": previous.tool,
                    f"from_{self.id_field}": previous.identifier,
                    f"to_{self.id_field}": call.identifier}
            self.interval(received_metric, previous.points.get("feedback_ready"),
                          call.points.get("tool_received"), **pair)
            if not proxy:
                self.interval("feedback_to_code_validated_s", previous.points.get("feedback_ready"),
                              call.points.get("code_validated"), **pair)
        self.counts["tool_calls"] = len(self.calls)
        self.counts["tool_received_calls"] = sum(
            "tool_received" in c.points for c in self.calls.values()
        )
        self.counts["python_calls"] = sum(c.tool == "python" for c in self.calls.values())
        self.counts["finish_calls"] = sum(c.tool == "finish" for c in self.calls.values())
        return {
            "schema": SUMMARY_SCHEMA, "source_format": self.source_format,
            "source": str(self.path), "counts": dict(self.counts), "calls": output,
            "metrics": {name: _distribution(samples) for name, samples in self.samples.items()},
            "gaps": self.gaps,
            "notes": [
                "Percentiles are descriptive; sample count alone does not establish stable p95.",
                "Waiting intervals include client/network work, not pure model reasoning or TTFT.",
                "Serial waits can span retries and replies without code, not just one request.",
                "Clock domains are never subtracted; missing events remain missing.",
                "Only adjacent serial feedback-to-code pairs are included; overlaps are excluded.",
                ("Codex function_call is a historical proxy, not host validation or code_ready."
                 if proxy else "code_validated records compilation, not action admission."),
            ],
        }


def summarize_interaction(path: Path) -> dict[str, Any]:
    """Summarize ``pyrualean.interaction.v1`` events; no runtime is started."""
    path = Path(path)
    if path.is_dir():
        path /= "interaction_events.jsonl"
    report = _Report(path, "interaction")
    seen: set[tuple[str, str]] = set()
    attempts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line, event in report.rows():
        session = _identifier(event.get("session_id"))
        clock = _identifier(event.get("clock_domain"))
        event_id = _identifier(event.get("event_id"))
        name = _identifier(event.get("event"))
        if event.get("schema") != TRACE_SCHEMA or not all((session, clock, event_id, name)):
            report.gap("invalid_trace_envelope", line)
            report.barrier()
            continue
        if (session, event_id) in seen:
            report.gap("duplicate_event_id", line, session_id=session, event_id=event_id)
            report.barrier()
            continue
        seen.add((session, event_id))
        if (name in _FAILURE_EVENTS or "error" in name
                or event.get("ok") is False or event.get("success") is False
                or event.get("status") == "error" or event.get("outcome") == "error"):
            report.counts["error_events"] += 1
        if "compact" in name:
            report.counts["compactions"] += 1
        if "retry" in name:
            report.counts["retry_events"] += 1
        point = report.point(line, session, clock, event.get("t"))
        if point is None:
            continue
        if name == "context_ready":
            report.origins[session] = point
        if name in _CALL_EVENTS:
            identifier = _identifier(event.get("local_call_id"))
            if identifier is None:
                report.gap("missing_local_call_id", line, stage=name, session_id=session)
                report.barrier()
                continue
            meta = {key: event[key] for key in (
                "decision_id", "attempt_id", "agent_attempt_id", "provider_call_id",
            )
                    if _identifier(event.get(key)) is not None}
            for key in ("episode_finished", "execution_paused", "ok", "arguments_valid"):
                if type(event.get(key)) is bool:
                    meta[key] = event[key]
            if event.get("outcome") == "finished":
                meta["episode_finished"] = True
            call = report.mark(session, identifier, _identifier(event.get("tool")),
                               name, point, meta)
            if name in _FAILURE_EVENTS:
                details = {}
                error_type = _identifier(event.get("error_type"))
                if error_type and error_type.isidentifier() and len(error_type) <= 128:
                    details["error_type"] = error_type
                if _identifier(event.get("outcome")) in {
                    "interrupted", "error", "paused", "episode_finished", "returned",
                }:
                    details["outcome"] = event["outcome"]
                call.event_meta[name] = details
        if name in {"agent_attempt_started", "agent_attempt_finished",
                    "request_started", "request_finished", "response_parsed"}:
            kind = "agent_attempt" if name.startswith("agent_attempt_") else "request"
            id_field = "agent_attempt_id" if kind == "agent_attempt" else "attempt_id"
            identifier = _identifier(event.get(id_field))
            if identifier is None:
                report.gap(f"missing_{id_field}", line, stage=name, session_id=session)
                continue
            key = (session, kind, identifier)
            attempt = attempts.setdefault(key, {"points": {}, "decision_id": None})
            if name in attempt["points"]:
                report.gap("duplicate_attempt_stage", line, session_id=session,
                           **{id_field: identifier})
                attempt["invalid"] = True
            attempt["points"][name] = point
            decision = _identifier(event.get("decision_id"))
            if decision and attempt["decision_id"] not in (None, decision):
                report.gap("conflicting_attempt_decision", line, session_id=session,
                           **{id_field: identifier})
                attempt["invalid"] = True
            attempt["decision_id"] = decision or attempt["decision_id"]
            if type(event.get("tool_calls")) is int and event["tool_calls"] >= 0:
                attempt["reported_tool_calls"] = event["tool_calls"]
    public_attempts = []
    grouped: Counter[tuple[str, str, str]] = Counter()
    for (session, kind, identifier), attempt in attempts.items():
        id_field = "agent_attempt_id" if kind == "agent_attempt" else "attempt_id"
        identity = {"session_id": session, "kind": kind, id_field: identifier,
                    "decision_id": attempt["decision_id"]}
        if attempt["decision_id"]:
            grouped[(session, kind, attempt["decision_id"])] += 1
        start, end = f"{kind}_started", f"{kind}_finished"
        points = attempt["points"]
        elapsed = None if attempt.get("invalid") else report.interval(
            f"{kind}_s", points.get(start), points.get(end), **identity,
        )
        missing = sorted({start, end} - points.keys())
        if missing:
            report.gap("incomplete_attempt", session_id=session, **{id_field: identifier},
                       missing_events=missing)
        public_attempts.append({**identity, "elapsed_s": elapsed, "missing_events": missing,
                                "invalid": bool(attempt.get("invalid")),
                                "reported_tool_calls": attempt.get("reported_tool_calls")})
    for call in report.calls.values():
        attempt = attempts.get((call.session, "request", call.meta.get("attempt_id")))
        if attempt and not call.invalid and not attempt.get("invalid"):
            for stage in ("tool_arguments_complete", "tool_received", "code_validated"):
                report.interval(
                    f"request_to_{stage}_s", attempt["points"].get("request_started"),
                    call.points.get(stage), session_id=call.session,
                    local_call_id=call.identifier, attempt_id=call.meta["attempt_id"],
                )
    report.counts["retry_attempts"] = sum(max(0, n - 1) for n in grouped.values())
    report.counts["requests_without_tool_calls"] = sum(
        a["kind"] == "request" and a["reported_tool_calls"] == 0 for a in public_attempts
    )
    report.counts["requests_with_unknown_tool_count"] = sum(
        a["kind"] == "request" and a["reported_tool_calls"] is None for a in public_attempts
    )
    result = report.finish({
        **_HOST_STAGES,
        "arguments_to_tool_received_s": ("tool_arguments_complete", "tool_received"),
    })
    result["attempts"] = public_attempts
    return result


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _finish_flags(output: Any) -> dict[str, bool]:
    """Extract only explicit lifecycle flags, never prose or model reasoning."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return {}
    if isinstance(output, list):
        flags = {}
        for block in output:
            flags.update(_finish_flags(block))
        return flags
    if not isinstance(output, dict):
        return {}
    flags = {key: output[key] for key in ("episode_finished", "execution_paused")
             if type(output.get(key)) is bool}
    for key in ("text", "content", "result"):
        if key in output:
            flags.update(_finish_flags(output[key]))
    return flags


def summarize_codex_rollout(path: Path) -> dict[str, Any]:
    """Historical timestamp proxies, paired by session and rollout call_id.

    This reads only the supplied rollout. CLI-only errors absent from that file
    are not reconstructed, and provider timing is not inferred from token usage.
    """
    report = _Report(Path(path), "codex-rollout")
    session = None
    usage = []
    since_usage_code = 0
    for line, event in report.rows():
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            report.gap("invalid_rollout_payload", line)
            report.barrier()
            continue
        kind = _identifier(event.get("type"))
        subtype = _identifier(payload.get("type"))
        if kind == "session_meta":
            session = _identifier(payload.get("id")) or _identifier(payload.get("session_id"))
            report.last_feedback.clear()
            since_usage_code = 0
            if session is None:
                report.gap("missing_session_id", line)
                report.barrier()
            continue
        if kind == "compacted" or (isinstance(subtype, str) and "compact" in subtype):
            report.counts["compactions"] += 1
        item = payload.get("item")
        if (kind in {"error", "turn.failed"} or subtype in {"error", "turn.failed"}
                or (isinstance(item, dict) and item.get("type") == "error")):
            report.counts["error_events"] += 1
        if kind == "token_usage_record":
            raw = payload.get("usage") or {}
            if isinstance(raw, dict):
                record = {key: raw[key] for key in (
                    "input_tokens", "cached_input_tokens", "output_tokens",
                    "reasoning_output_tokens", "total_tokens",
                ) if _number(raw.get(key)) and raw[key] >= 0}
                usage.append({"session_id": session, "line": line, **record})
                if since_usage_code == 0:
                    report.counts["usage_records_without_code_call"] += 1
                since_usage_code = 0
        relevant = subtype == "task_started" or (
            kind == "response_item" and subtype in {"function_call", "function_call_output"}
        )
        if not relevant:
            continue
        if session is None:
            report.gap("missing_session_id", line)
            report.barrier()
            continue
        point = report.point(line, session, "codex_rollout_utc", _timestamp(event.get("timestamp")))
        if point is None:
            continue
        if subtype == "task_started":
            report.origins[session] = point
            report.last_feedback.pop(session, None)
            report.counts["task_starts"] += 1
            continue
        identifier = _identifier(payload.get("call_id"))
        if identifier is None:
            report.gap("missing_rollout_call_id", line, session_id=session)
            report.barrier()
            continue
        if subtype == "function_call":
            tool = _identifier(payload.get("name"))
            meta = {}
            if tool in _CODE_TOOLS:
                since_usage_code += 1
                arguments = payload.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        arguments = None
                code = arguments.get("code") if isinstance(arguments, dict) else None
                if isinstance(code, str):
                    meta = {"code_chars": len(code), "code_lines": len(code.splitlines())}
                else:
                    report.gap("code_text_unavailable", line, session_id=session,
                               call_id=identifier)
            report.mark(session, identifier, tool, "tool_received", point, meta)
        else:
            call = report.calls.get((session, identifier))
            flags = _finish_flags(payload.get("output")) if call and call.tool == "finish" else {}
            report.mark(session, identifier, None, "feedback_ready", point, flags)
    result = report.finish({"tool_response_proxy_s": ("tool_received", "feedback_ready")})
    result["usage_records"] = usage
    result["notes"].extend([
        "Rollout call_id is not an MCP JSON-RPC request ID or a host local_call_id.",
        "Error/compaction counts cover this file only; CLI-only errors may be absent.",
        "Usage records without code calls are a log coverage count, not proven provider requests.",
    ])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--format", choices=("interaction", "codex-rollout"), default="interaction")
    parser.add_argument("--output", type=Path,
                        help="write a new JSON file; never overwrite evidence")
    args = parser.parse_args(argv)
    summarize = summarize_interaction if args.format == "interaction" else summarize_codex_rollout
    try:
        result = summarize(args.path)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
