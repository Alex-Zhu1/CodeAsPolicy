"""Read bounded RoboDojo run outputs without starting a model or changing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_object(path: Path, issues: list[str]) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as exc:
        issues.append(f"{path}: {exc}")
        return {}


def inspect_episode(episode: Path) -> dict:
    issues: list[str] = []
    result = read_object(episode / "result.json", issues)
    fields = (
        "task", "seed", "backend", "status", "environment_success", "env_steps",
        "env_steps_total", "tool_turns", "startup_s", "policy_wall_s", "total_wall_s",
    )
    report = {"episode": str(episode), "result": {k: result.get(k) for k in fields}}
    finish = result.get("finish")
    report["planner_finish_status"] = finish.get("status") if isinstance(finish, dict) else None
    report["native_results"] = native = []
    for path in sorted((episode / "bench/eval_result").rglob("_result.json")):
        data = read_object(path, issues)
        for key in ("success_rate", "score"):
            value = data.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                issues.append(f"{path}: missing or invalid native {key}.")
        details = data.get("details")
        if not isinstance(details, dict) or not details or any(
            not isinstance(row, dict) or type(row.get("success")) is not bool
            for row in details.values()
        ):
            issues.append(f"{path}: missing or invalid native episode verdicts.")
        native.append({
            "path": str(path),
            **{key: data.get(key) for key in ("success_rate", "score", "eval_time", "details")},
        })
    if not native:
        issues.append("No RoboDojo native _result.json found; native verdict unavailable.")
    elif len(native) > 1:
        issues.append("Multiple native result files; select the matching evaluation explicitly.")
    else:
        details = native[0].get("details")
        if isinstance(details, dict) and len(details) == 1:
            detail = next(iter(details.values()))
            native_success = detail.get("success") if isinstance(detail, dict) else None
            reported_success = result.get("environment_success")
            if (type(native_success) is bool and type(reported_success) is bool
                    and native_success != reported_success):
                issues.append("Native success disagrees with result.environment_success.")
    if report["planner_finish_status"] == "success" and result.get("environment_success") is False:
        issues.append("Planner claimed success but environment_success is false.")
    if type(result.get("environment_success")) is not bool:
        issues.append("Environment success is missing or not boolean; do not infer it from finish.")
    report["issues"] = issues
    return report


def verify_snapshot(root: Path) -> dict:
    issues: list[str] = []
    manifest = read_object(root / "manifest.json", issues)
    hashes = manifest.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        issues.append("No nonempty source_hashes manifest; snapshot is unverified.")
        return {"verified": False, "checked_files": 0, "issues": issues}
    frozen = (root / "frozen").resolve()
    checked = 0
    for name, expected in hashes.items():
        path = (frozen / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(frozen):
            issues.append(f"Invalid snapshot-relative path: {name}")
            continue
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            issues.append(f"{name}: {exc}")
            continue
        checked += 1
        if actual != expected:
            issues.append(f"Hash mismatch: {name}")
    return {
        "verified": not issues,
        "checked_files": checked,
        "scope": "Only files listed in manifest.source_hashes under this run's frozen/.",
        "issues": issues,
    }


def inspect_run(root: Path, *, verify_source: bool = False) -> dict:
    if (root / "result.json").is_file() or root.name == "episode":
        episodes = [root]
    elif (root / "episode").is_dir():
        episodes = [root / "episode"]
    else:
        episodes = sorted(p for p in root.glob("*/episode") if p.is_dir())
    report = {"run": str(root), "episodes": [inspect_episode(p) for p in episodes]}
    report["issues"] = [] if episodes else ["No episode directory found at supported depth."]
    if verify_source:
        report["snapshot"] = verify_snapshot(root)
    report["inspection_complete"] = (
        not report["issues"]
        and all(not item["issues"] for item in report["episodes"])
        and report.get("snapshot", {}).get("verified", True)
    )
    report["note"] = (
        "inspection_complete concerns reading/consistency only, not task success. "
        "Scores retain their original units; this does not audit actions, video or live processes."
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run root, task directory, or episode directory")
    parser.add_argument("--verify-source", action="store_true", help="Check run-root frozen hashes")
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error(f"not a directory: {args.run}")
    report = inspect_run(args.run.resolve(), verify_source=args.verify_source)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["inspection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
