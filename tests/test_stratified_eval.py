"""Tests for Pillar 4: Stratified Benchmark Regression Suite (4D acceptance metrics)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeaspolicy.evaluation import stratified as st
from codeaspolicy.evaluation.latency import TRACE_SCHEMA


def _write_trace(path: Path, first_wait_s: float = 2.5, inter_wait_s: float = 1.5) -> None:
    events = [
        {
            "schema": TRACE_SCHEMA,
            "session_id": "s1",
            "clock_domain": "monotonic",
            "event_id": "e1",
            "event": "context_ready",
            "t": 0.0,
        },
        {
            "schema": TRACE_SCHEMA,
            "session_id": "s1",
            "clock_domain": "monotonic",
            "event_id": "e2",
            "event": "tool_received",
            "local_call_id": "c1",
            "tool": "python",
            "t": first_wait_s,
        },
        {
            "schema": TRACE_SCHEMA,
            "session_id": "s1",
            "clock_domain": "monotonic",
            "event_id": "e3",
            "event": "feedback_ready",
            "local_call_id": "c1",
            "tool": "python",
            "t": first_wait_s + 0.5,
        },
        {
            "schema": TRACE_SCHEMA,
            "session_id": "s1",
            "clock_domain": "monotonic",
            "event_id": "e4",
            "event": "tool_received",
            "local_call_id": "c2",
            "tool": "python",
            "t": first_wait_s + 0.5 + inter_wait_s,
        },
        {
            "schema": TRACE_SCHEMA,
            "session_id": "s1",
            "clock_domain": "monotonic",
            "event_id": "e5",
            "event": "feedback_ready",
            "local_call_id": "c2",
            "tool": "python",
            "t": first_wait_s + 0.5 + inter_wait_s + 0.4,
        },
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _write_run(
    root: Path,
    *,
    task: str,
    success: bool,
    score: float | None = None,
    env_steps: int = 120,
    elapsed_s: float = 35.0,
    prompt_chars: int | None = 42000,
    input_tokens: int | None = 12000,
    with_trace: bool = False,
    first_wait_s: float = 2.0,
    inter_wait_s: float = 1.0,
    seed: int = 0,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "task": task,
        "seed": seed,
        "status": "completed",
        "environment_success": success,
        "env_steps": env_steps,
        "policy_wall_s": elapsed_s,
    }
    if score is not None:
        result["score"] = score
    if input_tokens is not None:
        result["generation"] = {"input_tokens": input_tokens}
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if prompt_chars is not None:
        (root / "prompt.txt").write_text("x" * prompt_chars, encoding="utf-8")
    if with_trace:
        _write_trace(root / "interaction_events.jsonl", first_wait_s, inter_wait_s)
    return root


def test_stratified_suite_definition_covers_all_three_regimes():
    assert st.STRATIFIED_SUITE == {
        "anti_regression": ("stack_bowls", "organize_table"),
        "last_mile_contact": ("insert_key", "plug_in_charger", "hang_mugs"),
        "dense_and_continuous": ("fasten_screws", "pour_liquid_into_cup", "build_tower"),
    }
    assert st.task_stratum("stack_bowls") == "anti_regression"
    assert st.task_stratum("robodojo:insert_key") == "last_mile_contact"
    assert st.task_stratum("pour_liquid_into_cup") == "dense_and_continuous"
    assert st.task_stratum("unknown_task") is None


def test_load_run_metrics_reads_4d_artifacts_and_optional_trace(tmp_path):
    run = _write_run(
        tmp_path / "ep_key",
        task="robodojo:insert_key",
        success=True,
        score=0.92,
        env_steps=84,
        elapsed_s=28.5,
        prompt_chars=48000,
        input_tokens=15000,
        with_trace=True,
        first_wait_s=3.0,
        inter_wait_s=1.0,
    )
    m = st.load_run_metrics(run)
    assert m.task == "insert_key" and m.stratum == "last_mile_contact"
    assert m.success is True and m.score == pytest.approx(0.92)
    assert m.env_steps == 84 and m.elapsed_s == pytest.approx(28.5)
    assert m.prompt_chars == 48000 and m.input_tokens == 15000
    assert m.first_tool_wait_s == pytest.approx(3.0)
    assert m.pre_tool_wait_s == pytest.approx(2.0)  # mean(3.0, 1.0)

    # Missing prompt.txt / interaction_events.jsonl + binary score fallback + usage.prompt_tokens
    minimal = tmp_path / "minimal"
    minimal.mkdir()
    (minimal / "result.json").write_text(
        json.dumps(
            {
                "task": "stack_bowls",
                "environment_success": False,
                "env_steps": 200,
                "elapsed_s": 50.0,
                "usage": {"prompt_tokens": 9000},
            }
        ),
        encoding="utf-8",
    )
    m2 = st.load_run_metrics(minimal)
    assert m2.task == "stack_bowls" and m2.stratum == "anti_regression"
    assert m2.success is False and m2.score == 0.0
    assert m2.prompt_chars is None and m2.pre_tool_wait_s is None
    assert m2.input_tokens == 9000 and m2.elapsed_s == 50.0


def _build_full_suite(root: Path, *, prompt_chars: int = 48000, overrides: dict | None = None) -> Path:
    overrides = overrides or {}
    defaults = {
        "stack_bowls": dict(success=True, score=1.0, env_steps=100, elapsed_s=30.0, input_tokens=10000),
        "organize_table": dict(success=True, score=0.9, env_steps=140, elapsed_s=40.0, input_tokens=12000),
        "insert_key": dict(success=True, score=1.0, env_steps=90, elapsed_s=35.0, input_tokens=14000),
        "plug_in_charger": dict(success=False, score=0.4, env_steps=160, elapsed_s=55.0, input_tokens=16000),
        "hang_mugs": dict(success=True, score=1.0, env_steps=110, elapsed_s=45.0, input_tokens=15000),
        "fasten_screws": dict(success=True, score=0.8, env_steps=180, elapsed_s=60.0, input_tokens=18000),
        "pour_liquid_into_cup": dict(success=True, score=0.85, env_steps=150, elapsed_s=50.0, input_tokens=17000),
        "build_tower": dict(success=False, score=0.5, env_steps=210, elapsed_s=70.0, input_tokens=19000),
    }
    for task, cfg in defaults.items():
        merged = dict(cfg, prompt_chars=prompt_chars, with_trace=True)
        merged.update(overrides.get(task, {}))
        _write_run(root / task, task=task, **merged)
    return root


def test_evaluate_stratified_suite_4d_metrics_and_clean_pass(tmp_path):
    base = _build_full_suite(tmp_path / "base", prompt_chars=52000)
    cand = _build_full_suite(
        tmp_path / "cand",
        prompt_chars=46000,
        overrides={
            "plug_in_charger": {"success": True, "score": 1.0, "env_steps": 110},
        },
    )
    report = st.evaluate_stratified_suite(cand, baseline_dirs=base, max_prompt_chars=55000)
    assert report["passed"] is True
    assert report["prompt_budget_exceeded"] is False
    assert report["anti_regression_drop"] is False
    assert report["flags"] == []

    ar = report["strata"]["anti_regression"]
    assert ar["episodes"] == 2 and ar["tasks_missing"] == []
    assert ar["success_rate"] == 1.0 and ar["mean_score"] == pytest.approx(0.95)
    assert ar["mean_env_steps"] == pytest.approx(120.0)
    assert ar["mean_elapsed_s"] == pytest.approx(35.0)
    assert ar["max_prompt_chars"] == 46000
    assert ar["mean_input_tokens"] == pytest.approx(11000.0)
    assert ar["mean_pre_tool_wait_s"] == pytest.approx(1.5)

    lm = report["strata"]["last_mile_contact"]
    assert lm["success_rate"] == 1.0
    assert report["comparison"]["strata"]["last_mile_contact"]["success_rate_delta"] > 0.3
    assert report["comparison"]["overall"]["max_prompt_chars_delta"] == -6000.0


def test_evaluate_stratified_suite_flags_prompt_budget_and_anti_regression_drops(tmp_path):
    base = _build_full_suite(tmp_path / "base", prompt_chars=48000)

    # 1. Prompt budget exceeded
    bloated = _build_full_suite(
        tmp_path / "bloated",
        prompt_chars=48000,
        overrides={"insert_key": {"prompt_chars": 56000}},
    )
    r_bloat = st.evaluate_stratified_suite(bloated, baseline_dirs=base, max_prompt_chars=55000)
    assert r_bloat["passed"] is False
    assert r_bloat["prompt_budget_exceeded"] is True
    assert r_bloat["anti_regression_drop"] is False
    assert r_bloat["flags"] == ["prompt_budget_exceeded"]
    assert len(r_bloat["prompt_budget_offenders"]) == 1
    assert r_bloat["prompt_budget_offenders"][0]["task"] == "insert_key"

    # 2. Anti-regression SR drop
    sr_drop = _build_full_suite(
        tmp_path / "sr_drop",
        overrides={"stack_bowls": {"success": False, "score": 0.0}},
    )
    r_sr = st.evaluate_stratified_suite(sr_drop, baseline_dirs=base)
    assert r_sr["passed"] is False
    assert r_sr["anti_regression_drop"] is True
    assert r_sr["prompt_budget_exceeded"] is False
    assert r_sr["flags"] == ["anti_regression_drop"]

    # 3. Anti-regression score drop (even with same SR)
    score_drop = _build_full_suite(
        tmp_path / "score_drop",
        overrides={"organize_table": {"success": True, "score": 0.65}},
    )
    r_score = st.evaluate_stratified_suite(score_drop, baseline_dirs=base)
    assert r_score["passed"] is False
    assert r_score["anti_regression_drop"] is True

    # 4. Drop in dense_and_continuous does NOT trigger anti_regression_drop
    other_drop = _build_full_suite(
        tmp_path / "other_drop",
        overrides={"fasten_screws": {"success": False, "score": 0.2}},
    )
    r_other = st.evaluate_stratified_suite(other_drop, baseline_dirs=base)
    assert r_other["anti_regression_drop"] is False and r_other["passed"] is True


def test_stratified_cli_exit_codes(tmp_path, capsys):
    base = _build_full_suite(tmp_path / "base", prompt_chars=45000)
    good = _build_full_suite(tmp_path / "good", prompt_chars=44000)
    bad = _build_full_suite(tmp_path / "bad", prompt_chars=60000)

    assert st.main([str(good), "--baseline", str(base)]) == 0
    out_ok = json.loads(capsys.readouterr().out)
    assert out_ok["passed"] is True

    assert st.main([str(bad), "--baseline", str(base), "--max-prompt-chars", "55000"]) == 1
    out_bad = json.loads(capsys.readouterr().out)
    assert out_bad["passed"] is False and out_bad["prompt_budget_exceeded"] is True
