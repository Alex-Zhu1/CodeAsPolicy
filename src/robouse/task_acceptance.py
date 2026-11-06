"""Public task requirements, separate from planner claims and evaluator results."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA = "pyrualean.task_acceptance.v1"
CATALOG_SCHEMA = "pyrualean.task_acceptance_catalog.v1"
KINDS = frozenset({"outcome", "order", "invariant", "final_state"})


def default_robodojo_acceptance_path() -> Path | None:
    """Return the packaged RoboDojo public acceptance catalog path when present."""
    candidate = Path(str(resources.files(__package__ or "robouse").joinpath("knowledge/robodojo_acceptance.json")))
    return candidate if candidate.is_file() else None


def normalize_acceptance(value: dict) -> dict:
    """Validate supplied public requirements without deriving any from a task name."""
    defaults = dict(
        schema=SCHEMA, requirements=[], coverage="unknown", coverage_source=None,
        verdict_authority="environment",
    )
    if not isinstance(value, dict) or set(value) - defaults.keys():
        raise ValueError("acceptance must contain only task-acceptance contract fields")
    result = defaults | copy.deepcopy(value)
    if result["schema"] != SCHEMA:
        raise ValueError("unsupported task-acceptance schema")
    if result["coverage"] not in ("unknown", "partial", "complete"):
        raise ValueError("acceptance coverage must be unknown/partial/complete")
    source = result["coverage_source"]
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("acceptance coverage_source must be nonempty text or None")
    if result["coverage"] != "unknown" and source is None:
        raise ValueError("a coverage claim needs a public coverage_source")
    authority = result["verdict_authority"]
    if not isinstance(authority, str) or not authority.strip():
        raise ValueError("acceptance verdict_authority must be nonempty text")
    requirements = result["requirements"]
    if not isinstance(requirements, list):
        raise ValueError("acceptance requirements must be a list")
    ids = set()
    for requirement in requirements:
        if not isinstance(requirement, dict) or set(requirement) != {
            "id", "kind", "condition", "source",
        }:
            raise ValueError("each acceptance requirement needs id, kind, condition and source")
        if any(not isinstance(v, str) or not v.strip() for v in requirement.values()):
            raise ValueError("acceptance requirement fields must be nonempty text")
        if requirement["kind"] not in KINDS:
            raise ValueError("acceptance kind must be outcome/order/invariant/final_state")
        if requirement["id"] in ids:
            raise ValueError("acceptance requirement IDs must be unique")
        ids.add(requirement["id"])
    if result["coverage"] == "complete" and not requirements:
        raise ValueError("empty requirements cannot claim complete acceptance coverage")
    return result


def _resolve_catalog_entry(
    payload: dict[str, Any], *, suite: str | None = None, task: int | str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Select one task's acceptance entry from a public catalog without guessing rules."""
    if payload.get("schema") != CATALOG_SCHEMA:
        return payload, None
    allowed = {"schema", "tasks", "source"}
    if set(payload) - allowed or not isinstance(payload.get("tasks"), dict):
        raise ValueError("acceptance catalog requires schema and a tasks mapping")
    tasks = payload["tasks"]
    candidates = []
    if suite is not None and task is not None:
        candidates.append(f"{suite}/{task}")
    if task is not None:
        candidates.append(str(task))
    for key in candidates:
        if key in tasks:
            entry = tasks[key]
            if not isinstance(entry, dict):
                raise ValueError(f"acceptance catalog entry {key!r} must be a dict")
            return entry, key
    return {}, None


def load_acceptance(
    source: str | Path | dict[str, Any] | None,
    *,
    suite: str | None = None,
    task: int | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load and validate a public acceptance contract or task catalog entry with provenance."""
    if source is None:
        return normalize_acceptance({}), None
    artifact: dict[str, Any] | None = None
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        data = path.read_bytes()
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid task-acceptance JSON in {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("task-acceptance file must contain a JSON object")
        entry, matched_key = _resolve_catalog_entry(payload, suite=suite, task=task)
        contract = normalize_acceptance(entry)
        artifact = {
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "schema": payload.get("schema", SCHEMA),
            "matched_key": matched_key,
        }
        return contract, artifact
    if isinstance(source, dict):
        entry, matched_key = _resolve_catalog_entry(source, suite=suite, task=task)
        contract = normalize_acceptance(entry)
        if source.get("schema") == CATALOG_SCHEMA:
            artifact = {"schema": CATALOG_SCHEMA, "matched_key": matched_key}
        return contract, artifact
    raise ValueError("task acceptance source must be a path, dict or None")


def acceptance_sha256(value: dict[str, Any]) -> str:
    """Deterministic digest of a normalized task-acceptance contract."""
    contract = normalize_acceptance(value)
    canonical = json.dumps(contract, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_acceptance(value: dict) -> list[str]:
    """Keep requirements and their source visible even in compact policy prompts."""
    contract = normalize_acceptance(value)
    lines = [
        "\nTask acceptance:",
        f"- verdict authority: {contract['verdict_authority']}; "
        "no verdict is supplied by this task card",
        f"- public requirement coverage: {contract['coverage']}"
        + (f" (source: {contract['coverage_source']})" if contract["coverage_source"] else ""),
    ]
    for requirement in contract["requirements"]:
        lines.append(f"- [{requirement['id']}; {requirement['kind']}] "
                     f"{requirement['condition']} (source: {requirement['source']})")
    if not contract["requirements"]:
        lines.append("- no supplemental public acceptance requirements supplied; "
                     "this does not mean there are no ordering, preservation or final-state rules")
    if contract["coverage"] != "complete":
        lines.extend([
            "- short task instructions often compress a multi-phase domain procedure "
            "(waiting/preconditions, primary action, supplemental/replacement item handling "
            "and placement, preserving untouched items, and arm/gripper reset); inspect step-0 "
            "images to account for all task-relevant object groups, supply stacks and target "
            "areas before fixing done_when",
            "- distinguish inferred steps and assumptions from supplied requirements; "
            "recording an assumption or narrower done_when never waives an unperformed task "
            "phase or justifies leaving task-relevant scene objects unused while budget "
            "remains. Unknown coverage alone is no reason to stop useful work",
        ])
    else:
        lines.append(
            "- distinguish inferred steps and assumptions from supplied requirements; "
            "recording an assumption or narrower done_when never waives a supplied requirement"
        )
    lines.extend([
        "- before finishing, review the whole original instruction and each public requirement: "
        "outcome, order, invariants and final state; cite current observation/action evidence "
        "and identify unmet or unknown items",
        "- a completed operation or planner goal is not task acceptance; "
        "your completion declaration remains a claim until the authority confirms it",
    ])
    return lines
