"""Batch accounting never turns missing results into task failures."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("psutil")

SPEC = importlib.util.spec_from_file_location(
    "supervised_batch", Path(__file__).parents[1] / "examples/run_supervised_batch.py"
)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_missing_or_boot_error_is_not_task_failure(tmp_path):
    assert batch.collect_result(tmp_path, 1)["outcome"] == "error"
    assert batch.collect_result(tmp_path, 1)["native_success"] is None
    (tmp_path / "episode").mkdir()
    batch.write_json(
        tmp_path / "episode/result.json", {"status": "boot_error", "environment_success": False}
    )
    assert batch.collect_result(tmp_path, 2)["outcome"] == "error"


def test_native_verdict_wins_over_planner_stuck(tmp_path):
    native = tmp_path / "episode/bench/eval_result/task"
    native.mkdir(parents=True)
    batch.write_json(native / "_result.json", {"success_rate": 1.0, "score": 100})
    batch.write_json(
        tmp_path / "episode/result.json",
        {
            "status": "completed",
            "environment_success": True,
            "finish": {"status": "stuck"},
            "env_steps": 42,
            "stuck_recovery": {"events": [{"event": "recovery_dispatched"}]},
        },
    )
    result = batch.collect_result(tmp_path, 0)
    assert result["outcome"] == "success" and result["native_score"] == 100
    assert result["planner_finish"]["status"] == "stuck"
    assert result["stuck_recovery_rounds"] == 1


def test_summary_excludes_pending_and_errors_from_evaluated_rate():
    rows = [
        {"state": "pending"},
        {"state": "running"},
        {"state": "finished", "outcome": "error", "native_score": 0},
        {"state": "finished", "outcome": "success", "native_score": 100},
        {"state": "finished", "outcome": "failure", "native_score": 20},
    ]
    result = batch.summary(rows)
    assert result["evaluated"] == 2 and result["errors"] == 1
    assert result["success_rate_evaluated"] == 0.5
    assert result["mean_native_score_evaluated"] == 60
    assert batch.summary([{"state": "pending"}])["success_rate_evaluated"] is None


def test_resume_preserves_finished_and_reconciles_dead_running(tmp_path, monkeypatch):
    batch.write_json(tmp_path / "config.json", {"tasks": ["one", "two", "three"]})
    batch.write_json(
        tmp_path / "state.json",
        {
            "jobs": [
                {"task": "one", "state": "finished", "outcome": "failure"},
                {"task": "two", "state": "running", "directory": str(tmp_path / "two"), "pid": 123},
                {"task": "three", "state": "pending"},
            ]
        },
    )
    monkeypatch.setattr(batch, "still_running", lambda job: False)
    run = batch.Batch(tmp_path)
    assert [j["state"] for j in run.jobs] == ["finished", "finished", "pending"]
    assert run.jobs[0]["outcome"] == "failure" and run.jobs[1]["outcome"] == "error"
    monkeypatch.setattr(batch, "still_running", lambda job: True)
    with pytest.raises(RuntimeError, match="Existing job still running"):
        batch.Batch(tmp_path)


def test_queue_continues_after_error_and_finishes_every_job(tmp_path, monkeypatch):
    import io
    import time
    from types import SimpleNamespace

    batch.write_json(
        tmp_path / "config.json",
        {
            "tasks": ["bad", "good", "unsolved"],
            "simulator_gpus": [1, 2],
            "job_timeout_s": 100,
        },
    )
    started = []

    def launch(self, job, gpu):
        directory = Path(job["directory"])
        (directory / "episode").mkdir(parents=True)
        failed = job["task"] == "bad"
        batch.write_json(
            directory / "episode/result.json",
            {
                "status": "boot_error" if failed else "completed",
                "environment_success": job["task"] == "good",
            },
        )
        process = SimpleNamespace(
            proc=SimpleNamespace(poll=lambda: 2 if failed else 0),
            started=time.monotonic(),
            capture=lambda: None,
            stop=lambda: None,
        )
        job.update(state="running", gpu=gpu)
        self.active[gpu] = (job, process, io.StringIO())
        started.append((job["task"], gpu))

    monkeypatch.setattr(batch.Batch, "start_valen", lambda self: None)
    monkeypatch.setattr(batch.Batch, "launch", launch)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(batch.signal, "signal", lambda *args: None)
    run = batch.Batch(tmp_path)
    run.run()
    assert started == [("bad", 1), ("good", 2), ("unsolved", 1)]
    state = batch.read_json(tmp_path / "state.json")
    assert state["state"] == "completed" and state["summary"]["finished"] == 3
    assert state["summary"]["errors"] == 1 and state["summary"]["successes"] == 1
    assert state["summary"]["task_failures"] == 1


def test_process_capture_rejects_reused_parent_pid(monkeypatch):
    from types import SimpleNamespace

    owned = batch.OwnedProcess.__new__(batch.OwnedProcess)
    owned.proc = SimpleNamespace(pid=123)
    owned.root_created = 10
    owned.identities = {}
    monkeypatch.setattr(
        batch.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 11)
    )
    owned.capture()
    assert owned.identities == {}
