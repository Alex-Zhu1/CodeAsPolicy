"""Timing metadata must remain usable under concurrency and harmless on failure."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pyrualean.interaction_trace import SCHEMA, InteractionTrace


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_relative_trace_survives_host_working_directory_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ticks = iter((100.0, 100.25, 101.5))
    trace = InteractionTrace(Path("episode") / "interaction.jsonl",
                             session_id="session-one", clock=lambda: next(ticks))
    elsewhere = tmp_path / "workspace"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    trace.emit("request_started", decision_id="d1")
    trace.emit("request_finished", decision_id="d1", success=True)
    rows = _events(tmp_path / "episode" / "interaction.jsonl")
    assert [row["t"] for row in rows] == [0.25, 1.5]
    assert {row["session_id"] for row in rows} == {"session-one"}
    assert {row["schema"] for row in rows} == {SCHEMA}
    assert {row["clock_domain"] for row in rows} == {trace.clock_domain}
    assert trace.summary() == {
        "path": str(tmp_path / "episode" / "interaction.jsonl"), "schema": SCHEMA,
        "session_id": "session-one", "clock_domain": trace.clock_domain, "write_errors": 0,
    }
    assert not (elsewhere / "episode").exists()


def test_concurrent_events_and_local_ids_are_unique_and_complete(tmp_path):
    trace = InteractionTrace(tmp_path / "interaction.jsonl")

    def record(worker):
        ids = []
        for _ in range(25):
            call_id = trace.new_id("tool")
            ids.append(call_id)
            trace.emit("tool_received", local_call_id=call_id, worker=worker)
        return ids

    with ThreadPoolExecutor(max_workers=8) as pool:
        local_ids = [item for group in pool.map(record, range(8)) for item in group]
    rows = _events(trace.path)
    assert len(rows) == len(set(local_ids)) == 200
    assert {row["local_call_id"] for row in rows} == set(local_ids)
    assert len({row["event_id"] for row in rows} | set(local_ids)) == 400
    assert [row["t"] for row in rows] == sorted(row["t"] for row in rows)
    assert trace.summary()["write_errors"] == 0


def test_write_failure_is_counted_without_error_text_and_next_write_can_recover(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    trace = InteractionTrace(blocker / "events.jsonl")
    trace.emit("request_started", decision_id="one")
    assert trace.summary()["write_errors"] == 1
    blocker.unlink()
    trace.emit("request_finished", decision_id="one", success=True)
    assert len(_events(trace.path)) == 1
    assert trace.summary()["write_errors"] == 1


def test_serialization_error_does_not_log_arbitrary_object_or_exception(tmp_path):
    secret = "sensitive value that must not appear"

    class Unsupported:
        def __repr__(self):
            raise RuntimeError(secret)

    trace = InteractionTrace(tmp_path / "events.jsonl")
    trace.emit("bad_metadata", unsupported=Unsupported())
    trace.emit("request_started", decision_id="valid")
    assert trace.summary()["write_errors"] == 1
    assert secret not in json.dumps(trace.summary()) + trace.path.read_text()
    assert [row["event"] for row in _events(trace.path)] == ["request_started"]


def test_trace_envelope_cannot_be_replaced_by_metadata(tmp_path):
    trace = InteractionTrace(tmp_path / "events.jsonl", session_id="real-session")
    trace.emit("request_started", schema="other", session_id="other",
               clock_domain="other", event_id="other", t=-999)
    row = _events(trace.path)[0]
    assert row["schema"] == SCHEMA and row["session_id"] == "real-session"
    assert row["clock_domain"] == trace.clock_domain and row["event_id"] != "other"
    assert row["t"] >= 0
