"""Offline timing regression tests; these never start a model or robot."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pyrualean.evaluation.latency import (
    TRACE_SCHEMA,
    main,
    summarize_codex_rollout,
    summarize_interaction,
)


def _write(path: Path, rows: list[dict | str]) -> Path:
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def _trace(name: str, t: float, call: str | None = None, **fields) -> dict:
    return {
        "schema": TRACE_SCHEMA,
        "event": name,
        "event_id": f"{name}:{t}:{call}:{fields.get('session_id', 's')}",
        "session_id": "s",
        "clock_domain": "test:monotonic",
        "t": t,
        **({"local_call_id": call, "tool": "python"} if call else {}),
        **fields,
    }


def _cell(call: str, start: float, **fields) -> list[dict]:
    return [
        _trace(name, start + offset, call, **fields)
        for name, offset in (
            ("tool_received", 0), ("tool_started", 1), ("validation_started", 2),
            ("code_validated", 3), ("execution_started", 4), ("execution_finished", 6),
            ("tool_result_ready", 8), ("feedback_ready", 9),
        )
    ]


def _rollout(t: float, kind: str, **payload) -> dict:
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=t)
    return {"timestamp": stamp.isoformat(), "type": kind, "payload": payload}


def _call(t: float, call: str, tool: str = "python", **fields) -> dict:
    return _rollout(t, "response_item", type="function_call", call_id=call, name=tool,
                    arguments=json.dumps({"code": "secret generated code"}), **fields)


def _output(t: float, call: str, **output) -> dict:
    return _rollout(t, "response_item", type="function_call_output", call_id=call,
                    output=json.dumps(output or {"text": "secret tool output"}))


def _kinds(result: dict) -> set[str]:
    return {gap["kind"] for gap in result["gaps"]}


def test_interaction_stages_first_wait_and_recoverable_finish(tmp_path):
    rows = [_trace("context_ready", 0), *_cell("a", 2)]
    rows += [
        _trace("tool_received", 12, "f", tool="finish"),
        _trace("tool_started", 12.5, "f", tool="finish"),
        _trace("tool_result_ready", 13, "f", tool="finish", episode_finished=False,
               execution_paused=True),
        _trace("feedback_ready", 14, "f", tool="finish"),
        *_cell("b", 20),
    ]
    _write(tmp_path / "interaction_events.jsonl", rows)
    result = summarize_interaction(tmp_path)
    metrics = result["metrics"]
    assert result["source_format"] == "interaction"
    assert result["counts"]["python_calls"] == 2
    assert metrics["context_to_first_tool_s"]["p50_s"] == 2
    assert metrics["context_to_first_validated_s"]["p50_s"] == 5
    assert metrics["execution_s"]["n"] == 2
    assert metrics["execution_s"]["p50_s"] == 2
    assert metrics["tool_total_s"]["p50_s"] == 9
    assert metrics["feedback_to_tool_received_s"]["samples"][0]["source_tool"] == "finish"
    assert metrics["feedback_to_tool_received_s"]["p50_s"] == 6
    assert metrics["feedback_to_code_validated_s"]["p50_s"] == 9
    assert metrics["tool_queue_s"]["p95_stability"] == "insufficient_samples"
    assert result["gaps"] == []


def test_interaction_attempts_retry_no_code_and_local_argument_timing(tmp_path):
    rows = [
        _trace("context_ready", 0),
        _trace("agent_attempt_started", .1, agent_attempt_id="agent1"),
        _trace("request_started", 1, attempt_id="r1", decision_id="d1"),
        _trace("request_finished", 3, attempt_id="r1", decision_id="d1", success=False),
        _trace("retry_wait_started", 3.1),
        _trace("request_started", 4, attempt_id="r2", decision_id="d1"),
        _trace("request_finished", 7, attempt_id="r2", decision_id="d1", success=True),
        _trace("response_parsed", 7.1, attempt_id="r2", decision_id="d1", tool_calls=0),
        _trace("request_started", 8, attempt_id="r3", decision_id="d2"),
        _trace("request_finished", 11, attempt_id="r3", decision_id="d2", success=True),
        _trace("response_parsed", 11.1, attempt_id="r3", decision_id="d2", tool_calls=1),
        _trace("tool_arguments_complete", 11.2, "a", attempt_id="r3", decision_id="d2",
               provider_call_id="provider1", code="must never leave parser"),
        *_cell("a", 12),
        _trace("agent_attempt_finished", 30, agent_attempt_id="agent1"),
        _trace("request_started", 31, attempt_id="r4", decision_id="d3"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    assert result["counts"]["error_events"] == 1
    assert result["counts"]["retry_attempts"] == 1
    assert result["counts"]["retry_events"] == 1
    assert result["counts"]["requests_without_tool_calls"] == 1
    assert result["counts"]["requests_with_unknown_tool_count"] == 2
    assert result["metrics"]["request_s"]["n"] == 3
    assert result["metrics"]["request_s"]["p50_s"] == 3
    assert result["metrics"]["agent_attempt_s"]["p50_s"] == 29.9
    assert result["metrics"]["request_to_tool_arguments_complete_s"]["p50_s"] == 3.2
    assert result["metrics"]["request_to_code_validated_s"]["p50_s"] == 7
    assert result["metrics"]["arguments_to_tool_received_s"]["p50_s"] == .8
    assert result["calls"][0]["provider_call_id"] == "provider1"
    assert "must never leave parser" not in json.dumps(result)
    assert "incomplete_attempt" in _kinds(result)


def test_interaction_missing_feedback_cannot_bridge_to_next_cell(tmp_path):
    rows = [*_cell("a", 1), *_cell("b", 12)[:-1], *_cell("c", 30)]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    samples = result["metrics"]["feedback_to_tool_received_s"]["samples"]
    # b is later seen to overlap c; neither ambiguous pair is included.
    assert samples == []
    assert "missing_feedback" in _kinds(result)
    assert "overlapping_calls" in _kinds(result)
    assert result["metrics"]["context_to_first_tool_s"]["n"] == 0


@pytest.mark.parametrize("fault,expected", [
    ("bad_json", "bad_json"), ("bad_envelope", "invalid_trace_envelope"),
    ("bad_time", "missing_or_invalid_time"), ("regression", "clock_regression"),
    ("duplicate", "duplicate_event_id"), ("duplicate_stage", "duplicate_call_stage"),
])
def test_interaction_corruption_preserves_gaps_without_wait_samples(tmp_path, fault, expected):
    rows = _cell("a", 1)
    if fault == "bad_json":
        rows.append("not json: secret raw data")
    elif fault == "bad_envelope":
        rows.append({"schema": "unknown", "payload": "secret raw data"})
    elif fault == "bad_time":
        rows.append(_trace("context_ready", float("nan")))
    elif fault == "regression":
        rows.append(_trace("other", -1))
    elif fault == "duplicate":
        rows.append(rows[-1].copy())
    else:
        rows.append(_trace("feedback_ready", 11, "a"))
    rows += _cell("b", 20)
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    assert result["metrics"]["feedback_to_tool_received_s"]["n"] == 0
    assert expected in _kinds(result)
    assert "secret raw data" not in json.dumps(result)


def test_interaction_cross_clock_and_session_are_never_subtracted(tmp_path):
    rows = [
        _trace("tool_received", 1, "same", clock_domain="one"),
        _trace("feedback_ready", 5, "same", clock_domain="two"),
        *_cell("same", 8, session_id="other"),
        *_cell("next", 20, clock_domain="one"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    assert result["counts"]["tool_calls"] == 3
    assert result["metrics"]["tool_total_s"]["n"] == 2
    assert result["metrics"]["feedback_to_tool_received_s"]["n"] == 0
    assert "cross_clock_interval" in _kinds(result)


def test_interaction_reversed_execution_is_missing_not_negative(tmp_path):
    rows = [
        _trace("tool_received", 1, "a"),
        _trace("execution_finished", 2, "a"),
        _trace("execution_started", 3, "a"),
        _trace("feedback_ready", 4, "a"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    assert result["metrics"]["execution_s"]["n"] == 0
    assert "reversed_interval" in _kinds(result)


def test_interaction_parallel_calls_do_not_form_serial_proxy(tmp_path):
    rows = [
        _trace("tool_received", 1, "a"), _trace("tool_received", 2, "b"),
        _trace("feedback_ready", 3, "a"), _trace("feedback_ready", 4, "b"),
        *_cell("c", 5), *_cell("d", 20),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    samples = result["metrics"]["feedback_to_tool_received_s"]["samples"]
    assert len(samples) == 1
    assert (samples[0]["from_local_call_id"], samples[0]["to_local_call_id"]) == ("c", "d")
    assert [c["overlapping"] for c in result["calls"]] == [True, True, False, False]


def test_interaction_rejected_arguments_preserve_absent_execution(tmp_path):
    rows = [
        _trace("tool_arguments_complete", 1, "a", arguments_valid=False),
        _trace("feedback_ready", 2, "a", outcome="error"),
        *_cell("b", 5),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    assert result["counts"]["error_events"] == 1
    assert result["counts"]["tool_calls"] == 2
    assert result["counts"]["tool_received_calls"] == 1
    assert result["calls"][0]["arguments_valid"] is False
    assert "code_validated" in result["calls"][0]["missing_events"]
    assert result["metrics"]["feedback_to_tool_received_s"]["n"] == 0
    assert result["metrics"]["execution_s"]["n"] == 1
    assert "unmatched_feedback" in _kinds(result)


def test_interaction_compile_rejection_explains_expected_missing_stages(tmp_path):
    rows = [
        _trace("tool_received", 1, "a"), _trace("tool_started", 2, "a"),
        _trace("validation_started", 3, "a"),
        _trace("validation_failed", 4, "a", error_type="SyntaxError",
               error="secret source or exception text"),
        _trace("tool_result_ready", 5, "a"), _trace("feedback_ready", 6, "a"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    call = result["calls"][0]
    assert set(call["missing_events"]) == {
        "code_validated", "execution_started", "execution_finished",
    }
    assert all(reason == {"reason": "not_reached", "cause": "validation_failed"}
               for reason in call["missing_event_reasons"].values())
    assert call["events"]["validation_failed"]["error_type"] == "SyntaxError"
    assert result["counts"]["error_events"] == 1
    assert result["metrics"]["execution_s"]["n"] == 0
    assert result["gaps"] == []
    assert "secret source or exception text" not in json.dumps(result)


def test_interaction_program_missing_detail_is_not_instrumented(tmp_path):
    rows = [
        _trace("tool_received", 1, "p", tool="run_program"),
        _trace("tool_started", 2, "p", tool="run_program"),
        _trace("tool_result_ready", 5, "p", tool="run_program"),
        _trace("feedback_ready", 6, "p", tool="run_program"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    call = result["calls"][0]
    assert set(call["missing_events"]) == {
        "validation_started", "code_validated", "execution_started", "execution_finished",
    }
    assert all(reason == {"reason": "not_instrumented", "scope": "run_program"}
               for reason in call["missing_event_reasons"].values())
    assert result["metrics"]["tool_total_s"]["p50_s"] == 5
    assert result["metrics"]["execution_s"]["n"] == 0
    assert result["gaps"] == []


def test_interaction_execution_setup_failure_does_not_erase_validation(tmp_path):
    rows = [
        *_cell("a", 0)[:4],
        _trace("execution_setup_failed", 4, "a", outcome="error"),
        _trace("tool_result_ready", 5, "a"), _trace("feedback_ready", 6, "a"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    call = result["calls"][0]
    assert set(call["missing_events"]) == {"execution_started", "execution_finished"}
    assert all(reason == {"reason": "not_reached", "cause": "execution_setup_failed"}
               for reason in call["missing_event_reasons"].values())
    assert call["events"]["execution_setup_failed"]["outcome"] == "error"
    assert result["metrics"]["validation_s"]["n"] == 1
    assert result["metrics"]["execution_s"]["n"] == 0


@pytest.mark.parametrize("late_execution", [False, True])
def test_interaction_feedback_failure_only_explains_feedback(tmp_path, late_execution):
    rows = [
        _trace("tool_received", 1, "a"), _trace("tool_started", 2, "a"),
        _trace("feedback_failed", 3, "a", error_type="CancelledError"),
    ]
    if late_execution:
        rows += [
            _trace("execution_started", 4, "a"),
            _trace("execution_finished", 5, "a"),
            _trace("tool_result_ready", 6, "a"),
        ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    call = result["calls"][0]
    reasons = call["missing_event_reasons"]
    assert reasons["feedback_ready"] == {"reason": "not_reached", "cause": "feedback_failed"}
    assert all(reason == {"reason": "unknown"}
               for stage, reason in reasons.items() if stage != "feedback_ready")
    assert call["events"]["feedback_failed"]["error_type"] == "CancelledError"
    assert result["metrics"]["execution_s"]["n"] == int(late_execution)


def test_interaction_feedback_absence_without_failure_remains_unknown(tmp_path):
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", _cell("a", 0)[:-1]))
    assert result["calls"][0]["missing_event_reasons"] == {
        "feedback_ready": {"reason": "unknown"},
    }
    assert result["gaps"][0]["reason"] == "unknown"


def test_interaction_conflicting_request_decision_has_no_latency(tmp_path):
    rows = [
        _trace("request_started", 1, attempt_id="r", decision_id="d1"),
        _trace("request_finished", 2, attempt_id="r", decision_id="d2"),
    ]
    result = summarize_interaction(_write(tmp_path / "trace.jsonl", rows))
    assert result["attempts"][0]["elapsed_s"] is None
    assert "conflicting_attempt_decision" in _kinds(result)


def test_rollout_adjacent_output_includes_finish_recovery_and_other_tools(tmp_path):
    rows = [
        _rollout(0, "session_meta", id="s"),
        _rollout(1, "event_msg", type="task_started"),
        _call(5, "a"), _output(7, "a"),
        _call(10, "f", "finish"),
        _output(11, "f", content=[{"text": json.dumps({"episode_finished": False,
                                                       "execution_paused": True})}]),
        _call(19, "b"), _output(22, "b"),
        _call(25, "look", "inspect"), _output(28, "look"),
        _call(40, "c"), _output(42, "c"),
        _call(43, "end", "finish"), _output(44, "end", episode_finished=True),
    ]
    result = summarize_codex_rollout(_write(tmp_path / "rollout.jsonl", rows))
    metric = result["metrics"]["feedback_to_next_python_s"]
    assert result["source_format"] == "codex-rollout"
    assert result["counts"]["python_calls"] == 3
    assert result["counts"]["finish_calls"] == 2
    assert metric["n"] == 2
    assert metric["p50_s"] == 10
    assert metric["p95_s"] == 11.8
    assert [s["source_tool"] for s in metric["samples"]] == ["finish", "inspect"]
    assert result["metrics"]["task_started_to_first_python_s"]["p50_s"] == 4
    assert "feedback_to_code_validated_s" not in result["metrics"]
    assert "secret generated code" not in json.dumps(result)
    assert "secret tool output" not in json.dumps(result)
    assert result["gaps"] == []


def test_rollout_usage_errors_compactions_and_no_code_coverage(tmp_path):
    rows = [
        _rollout(0, "session_meta", id="s"),
        _rollout(1, "token_usage_record", usage={"input_tokens": 100,
                 "cached_input_tokens": 20, "reasoning_output_tokens": 7,
                 "bad": "secret", "output_tokens": -1}),
        _call(2, "a"), _output(3, "a"),
        _rollout(4, "token_usage_record", usage={"input_tokens": 150}),
        _rollout(5, "error", type="error", message="secret provider error"),
        _rollout(6, "compacted", replacement_history="secret history"),
    ]
    result = summarize_codex_rollout(_write(tmp_path / "rollout.jsonl", rows))
    assert result["counts"]["error_events"] == 1
    assert result["counts"]["compactions"] == 1
    assert result["counts"]["usage_records_without_code_call"] == 1
    assert result["usage_records"][0] == {
        "session_id": "s", "line": 2, "input_tokens": 100, "cached_input_tokens": 20,
        "reasoning_output_tokens": 7,
    }
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("fault,expected", [
    ("missing_output", "missing_feedback"), ("overlap", "overlapping_calls"),
    ("bad_json", "bad_json"), ("regression", "clock_regression"),
    ("terminal", "terminal_feedback_followed_by_code"),
    ("missing_id", "missing_rollout_call_id"),
])
def test_rollout_ambiguous_waits_are_excluded(tmp_path, fault, expected):
    rows = [_rollout(0, "session_meta", id="s"), _call(2, "a")]
    if fault == "missing_output":
        rows += [_call(10, "b"), _output(11, "b")]
    elif fault == "overlap":
        rows += [_call(3, "b"), _output(4, "a"), _output(5, "b"),
                 _call(10, "c"), _output(11, "c")]
    elif fault == "terminal":
        rows += [_output(3, "a"), _call(4, "f", "finish"),
                 _output(5, "f", episode_finished=True), _call(10, "b"), _output(11, "b")]
    else:
        rows.append(_output(3, "a"))
        if fault == "bad_json":
            rows.append("this log record is truncated")
        elif fault == "missing_id":
            rows.append(_rollout(4, "response_item", type="function_call_output"))
        else:
            rows.append(_call(1, "b"))
        if fault != "regression":
            rows.append(_call(10, "b"))
        rows.append(_output(11, "b"))
    result = summarize_codex_rollout(_write(tmp_path / "rollout.jsonl", rows))
    assert result["metrics"]["feedback_to_next_python_s"]["n"] == 0
    assert expected in _kinds(result)


def test_rollout_requires_session_and_timezone_and_scopes_call_ids(tmp_path):
    no_zone = _call(1, "bad_time")
    no_zone["timestamp"] = "2026-01-01T00:00:01"
    rows = [
        _call(0, "no_session"), _rollout(0, "session_meta", id="s1"), no_zone,
        _call(2, "same"), _output(3, "same"),
        _rollout(4, "session_meta", id="s2"), _call(5, "same"), _output(6, "same"),
    ]
    result = summarize_codex_rollout(_write(tmp_path / "rollout.jsonl", rows))
    assert result["counts"]["python_calls"] == 2
    assert result["metrics"]["feedback_to_next_python_s"]["n"] == 0
    assert result["metrics"]["task_started_to_first_python_s"]["n"] == 0
    assert {"missing_session_id", "missing_or_invalid_time"} <= _kinds(result)


def test_cli_stdout_json_and_output_never_overwrites_input_or_report(tmp_path, capsys):
    path = _write(tmp_path / "trace.jsonl", _cell("a", 0))
    original = path.read_bytes()
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["python_calls"] == 1
    output = tmp_path / "summary.json"
    assert main([str(path), "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert json.loads(saved)["source_format"] == "interaction"
    with pytest.raises(SystemExit) as exc:
        main([str(path), "--output", str(path)])
    assert exc.value.code == 2
    assert path.read_bytes() == original
    with pytest.raises(SystemExit):
        main([str(path), "--output", str(output)])
    assert output.read_bytes() == saved


def test_empty_file_has_null_statistics_not_zero_or_synthetic_samples(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    for summarize in (summarize_interaction, summarize_codex_rollout):
        result = summarize(path)
        assert result["counts"]["tool_calls"] == 0
        assert all(m["n"] == 0 and m["p50_s"] is None for m in result["metrics"].values())
