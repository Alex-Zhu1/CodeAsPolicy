# Copyright 2026 CodeAsPolicy Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Stratified Benchmark Regression Suite for RoboDojo 4D acceptance evaluation.

Replaces single-case tuning with a codified three-stratum regression suite covering
RoboDojo's distinct capability regimes:

* ``anti_regression`` (high-SR coarse baseline): ``stack_bowls``, ``organize_table``
* ``last_mile_contact`` (sub-5 mm aperture / hook / plug): ``insert_key``,
  ``plug_in_charger``, ``hang_mugs``
* ``dense_and_continuous`` (multi-stage assembly / fluid / torque): ``fasten_screws``,
  ``pour_liquid_into_cup``, ``build_tower``

Computes per-stratum and overall 4D acceptance metrics:
1. ``success_rate`` and ``mean_score`` (from ``result.json``),
2. ``mean_env_steps``,
3. ``mean_elapsed_s`` (plus pre-tool wait when ``interaction_events.jsonl`` is present),
4. ``max_prompt_chars`` (from ``prompt.txt``) and ``mean_input_tokens`` (from ``result.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .latency import summarize_interaction

STRATIFIED_SUITE: dict[str, tuple[str, ...]] = {
    "anti_regression": ("stack_bowls", "organize_table"),
    "last_mile_contact": ("insert_key", "plug_in_charger", "hang_mugs"),
    "dense_and_continuous": ("fasten_screws", "pour_liquid_into_cup", "build_tower"),
}

DEFAULT_MAX_PROMPT_CHARS = 55000


def normalize_task_name(task: str) -> str:
    """Strip optional ``suite:`` prefixes and surrounding whitespace from a task name."""

    clean = str(task or "").strip()
    if ":" in clean:
        clean = clean.rsplit(":", 1)[-1].strip()
    return clean


def task_stratum(
    task: str, suite: Mapping[str, Sequence[str]] = STRATIFIED_SUITE
) -> str | None:
    """Return the stratum name containing ``task``, or ``None`` if outside the suite."""

    name = normalize_task_name(task)
    for stratum, tasks in suite.items():
        if name in tasks:
            return stratum
    return None


@dataclass(frozen=True)
class RunMetrics:
    """4D acceptance metrics extracted from a single episode run directory."""

    run_dir: str
    task: str
    stratum: str | None
    seed: int | None = None
    status: str | None = None
    success: bool | None = None
    score: float | None = None
    env_steps: int | None = None
    elapsed_s: float | None = None
    pre_tool_wait_s: float | None = None
    first_tool_wait_s: float | None = None
    prompt_chars: int | None = None
    input_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_input_tokens(data: Mapping[str, Any]) -> int | None:
    direct = _int(data.get("input_tokens"))
    if direct is not None:
        return direct
    for container in (
        data.get("usage"),
        data.get("generation"),
        (data.get("generation") or {}).get("usage")
        if isinstance(data.get("generation"), Mapping)
        else None,
    ):
        if not isinstance(container, Mapping):
            continue
        for key in ("input_tokens", "prompt_tokens"):
            val = _int(container.get(key))
            if val is not None:
                return val
    return None


def _extract_score(data: Mapping[str, Any], success: bool | None) -> float | None:
    for candidate in (
        data.get("score"),
        data.get("environment_score"),
        (data.get("evaluation") or {}).get("score")
        if isinstance(data.get("evaluation"), Mapping)
        else None,
        (data.get("env_status") or {}).get("score")
        if isinstance(data.get("env_status"), Mapping)
        else None,
    ):
        val = _float(candidate)
        if val is not None:
            return val
    if success is not None:
        return 1.0 if success else 0.0
    return None


def _extract_pre_tool_waits(root: Path) -> tuple[float | None, float | None]:
    """Return ``(mean_pre_tool_wait_s, first_tool_wait_s)`` from ``interaction_events.jsonl``."""

    trace_path = root / "interaction_events.jsonl"
    if not trace_path.is_file():
        return None, None
    try:
        summary = summarize_interaction(trace_path)
    except Exception:  # noqa: BLE001 - malformed trace must not break suite aggregation
        return None, None
    metrics = summary.get("metrics") if isinstance(summary, Mapping) else None
    if not isinstance(metrics, Mapping):
        return None, None
    first_block = metrics.get("context_to_first_tool_s") or {}
    inter_block = metrics.get("feedback_to_tool_received_s") or {}
    first_wait = _float(first_block.get("mean_s")) if isinstance(first_block, Mapping) else None
    samples: list[float] = []
    for block in (first_block, inter_block):
        if isinstance(block, Mapping) and isinstance(block.get("samples"), list):
            for entry in block["samples"]:
                if isinstance(entry, Mapping):
                    sec = _float(entry.get("seconds"))
                    if sec is not None:
                        samples.append(sec)
    if samples:
        return round(sum(samples) / len(samples), 6), first_wait
    inter_wait = _float(inter_block.get("mean_s")) if isinstance(inter_block, Mapping) else None
    vals = [v for v in (first_wait, inter_wait) if v is not None]
    return (round(sum(vals) / len(vals), 6) if vals else None), first_wait


def load_run_metrics(
    run_dir: str | Path,
    *,
    suite: Mapping[str, Sequence[str]] = STRATIFIED_SUITE,
) -> RunMetrics:
    """Load 4D acceptance metrics from a run directory (or direct path to ``result.json``)."""

    path = Path(run_dir)
    if path.is_file() and path.name == "result.json":
        root = path.parent
        result_path = path
    else:
        root = path
        result_path = root / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"result.json not found in {root}")

    data = _read_json(result_path)
    card = _read_json(root / "card.json")
    raw_task = data.get("task") or card.get("task") or root.name
    task = normalize_task_name(str(raw_task))

    if "environment_success" in data and data["environment_success"] is not None:
        success: bool | None = bool(data["environment_success"])
    elif "success" in data and data["success"] is not None:
        success = bool(data["success"])
    else:
        success = None

    score = _extract_score(data, success)
    env_steps = next(
        (
            v
            for k in ("env_steps", "env_steps_total", "steps")
            if (v := _int(data.get(k))) is not None
        ),
        None,
    )
    elapsed_s = next(
        (
            v
            for k in ("elapsed_s", "policy_wall_s", "total_wall_s", "wall_time_s")
            if (v := _float(data.get(k))) is not None
        ),
        None,
    )

    prompt_chars: int | None = None
    prompt_path = root / "prompt.txt"
    if prompt_path.is_file():
        prompt_chars = len(prompt_path.read_text(encoding="utf-8"))
    elif _int(data.get("prompt_chars")) is not None:
        prompt_chars = _int(data.get("prompt_chars"))
    elif isinstance(data.get("prompt"), str):
        prompt_chars = len(data["prompt"])

    input_tokens = _extract_input_tokens(data)
    pre_tool_wait_s, first_tool_wait_s = _extract_pre_tool_waits(root)

    return RunMetrics(
        run_dir=str(root.resolve()),
        task=task,
        stratum=task_stratum(task, suite),
        seed=_int(data.get("seed")),
        status=str(data["status"]) if data.get("status") is not None else None,
        success=success,
        score=score,
        env_steps=env_steps,
        elapsed_s=elapsed_s,
        pre_tool_wait_s=pre_tool_wait_s,
        first_tool_wait_s=first_tool_wait_s,
        prompt_chars=prompt_chars,
        input_tokens=input_tokens,
    )


def _discover_run_dirs(run_dirs: str | Path | Iterable[str | Path]) -> list[Path]:
    items = [run_dirs] if isinstance(run_dirs, (str, Path)) else list(run_dirs)
    found: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)

    for raw in items:
        p = Path(raw)
        if p.is_file() and p.name == "result.json":
            add(p.parent)
        elif (p / "result.json").is_file():
            add(p)
        elif p.is_dir():
            for marker in sorted(p.rglob("result.json")):
                add(marker.parent)
        else:
            raise FileNotFoundError(f"run path does not exist: {p}")
    return found


def _mean(values: Sequence[float | int | None], digits: int = 4) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(sum(clean) / len(clean), digits) if clean else None


def _aggregate(
    runs: Sequence[RunMetrics], *, expected_tasks: Sequence[str] | None = None
) -> dict[str, Any]:
    known_success = [r.success for r in runs if r.success is not None]
    prompt_vals = [r.prompt_chars for r in runs if r.prompt_chars is not None]
    tasks_present = sorted({r.task for r in runs})
    tasks_missing = (
        sorted(set(expected_tasks) - set(tasks_present)) if expected_tasks is not None else []
    )
    return {
        "episodes": len(runs),
        "tasks_present": tasks_present,
        "tasks_missing": tasks_missing,
        "success_rate": (
            round(sum(bool(v) for v in known_success) / len(known_success), 4)
            if known_success
            else None
        ),
        "mean_score": _mean([r.score for r in runs]),
        "mean_env_steps": _mean([r.env_steps for r in runs], digits=2),
        "mean_elapsed_s": _mean([r.elapsed_s for r in runs], digits=3),
        "mean_pre_tool_wait_s": _mean([r.pre_tool_wait_s for r in runs], digits=4),
        "mean_first_tool_wait_s": _mean([r.first_tool_wait_s for r in runs], digits=4),
        "max_prompt_chars": max(prompt_vals) if prompt_vals else None,
        "mean_prompt_chars": _mean(prompt_vals, digits=1),
        "mean_input_tokens": _mean([r.input_tokens for r in runs], digits=1),
    }


def _delta(cur: float | int | None, base: float | int | None, digits: int = 4) -> float | None:
    if cur is None or base is None:
        return None
    return round(float(cur) - float(base), digits)


def _compare_blocks(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float | None]:
    return {
        "success_rate_delta": _delta(current.get("success_rate"), baseline.get("success_rate")),
        "mean_score_delta": _delta(current.get("mean_score"), baseline.get("mean_score")),
        "mean_env_steps_delta": _delta(
            current.get("mean_env_steps"), baseline.get("mean_env_steps"), digits=2
        ),
        "mean_elapsed_s_delta": _delta(
            current.get("mean_elapsed_s"), baseline.get("mean_elapsed_s"), digits=3
        ),
        "mean_pre_tool_wait_s_delta": _delta(
            current.get("mean_pre_tool_wait_s"), baseline.get("mean_pre_tool_wait_s")
        ),
        "max_prompt_chars_delta": _delta(
            current.get("max_prompt_chars"), baseline.get("max_prompt_chars"), digits=1
        ),
        "mean_input_tokens_delta": _delta(
            current.get("mean_input_tokens"), baseline.get("mean_input_tokens"), digits=1
        ),
    }


def evaluate_stratified_suite(
    run_dirs: str | Path | Iterable[str | Path],
    *,
    baseline_dirs: str | Path | Iterable[str | Path] | None = None,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    suite: Mapping[str, Sequence[str]] = STRATIFIED_SUITE,
    anti_regression_tol: float = 1e-9,
) -> dict[str, Any]:
    """Aggregate 4D acceptance metrics per stratum and overall, with regression checks.

    Flags:
    * ``prompt_budget_exceeded`` when any run's ``prompt_chars > max_prompt_chars``.
    * ``anti_regression_drop`` when ``baseline_dirs`` is given and the
      ``anti_regression`` stratum's ``success_rate`` or ``mean_score`` drops below
      baseline by more than ``anti_regression_tol``.
    """

    paths = _discover_run_dirs(run_dirs)
    runs = [load_run_metrics(p, suite=suite) for p in paths]
    all_expected = [task for tasks in suite.values() for task in tasks]

    def build_strata(items: Sequence[RunMetrics]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for stratum, tasks in suite.items():
            subset = [r for r in items if r.stratum == stratum]
            out[stratum] = _aggregate(subset, expected_tasks=tasks)
        other = [r for r in items if r.stratum is None]
        if other:
            out["other"] = _aggregate(other)
        return out

    overall = _aggregate(runs, expected_tasks=all_expected)
    strata = build_strata(runs)

    offenders = [
        {"run_dir": r.run_dir, "task": r.task, "prompt_chars": r.prompt_chars}
        for r in runs
        if r.prompt_chars is not None and r.prompt_chars > max_prompt_chars
    ]
    prompt_budget_exceeded = bool(offenders)

    baseline_summary: dict[str, Any] | None = None
    comparison: dict[str, Any] | None = None
    anti_regression_drop = False

    if baseline_dirs is not None:
        base_paths = _discover_run_dirs(baseline_dirs)
        base_runs = [load_run_metrics(p, suite=suite) for p in base_paths]
        base_overall = _aggregate(base_runs, expected_tasks=all_expected)
        base_strata = build_strata(base_runs)
        baseline_summary = {"overall": base_overall, "strata": base_strata}
        comparison = {
            "overall": _compare_blocks(overall, base_overall),
            "strata": {
                name: _compare_blocks(strata[name], base_strata[name])
                for name in suite
                if name in strata and name in base_strata
            },
        }
        if "anti_regression" in comparison["strata"]:
            ar = comparison["strata"]["anti_regression"]
            sr_delta = ar.get("success_rate_delta")
            score_delta = ar.get("mean_score_delta")
            anti_regression_drop = bool(
                (sr_delta is not None and sr_delta < -anti_regression_tol)
                or (score_delta is not None and score_delta < -anti_regression_tol)
            )

    flags: list[str] = []
    if prompt_budget_exceeded:
        flags.append("prompt_budget_exceeded")
    if anti_regression_drop:
        flags.append("anti_regression_drop")

    return {
        "passed": not flags,
        "prompt_budget_exceeded": prompt_budget_exceeded,
        "anti_regression_drop": anti_regression_drop,
        "flags": flags,
        "max_prompt_chars_limit": int(max_prompt_chars),
        "prompt_budget_offenders": offenders,
        "overall": overall,
        "strata": strata,
        "baseline": baseline_summary,
        "comparison": comparison,
        "runs": [r.to_dict() for r in runs],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m codeaspolicy.evaluation.stratified",
        description="Evaluate a RoboDojo stratified benchmark regression suite.",
    )
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run directories or roots")
    parser.add_argument(
        "--baseline",
        nargs="+",
        type=Path,
        default=None,
        help="baseline run directories or roots for anti-regression comparison",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CHARS,
        help=f"maximum allowed prompt.txt character count (default {DEFAULT_MAX_PROMPT_CHARS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_stratified_suite(
        args.run_dirs,
        baseline_dirs=args.baseline,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


__all__ = [
    "DEFAULT_MAX_PROMPT_CHARS",
    "STRATIFIED_SUITE",
    "RunMetrics",
    "evaluate_stratified_suite",
    "load_run_metrics",
    "main",
    "normalize_task_name",
    "task_stratum",
]


if __name__ == "__main__":
    sys.exit(main())
