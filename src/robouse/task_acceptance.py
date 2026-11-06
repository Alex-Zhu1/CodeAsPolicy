"""Public task requirements, separate from planner claims and evaluator results."""

from __future__ import annotations

import copy

SCHEMA = "pyrualean.task_acceptance.v1"
KINDS = frozenset({"outcome", "order", "invariant", "final_state"})


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
    lines.extend([
        "- before finishing, review the whole original instruction and each public requirement: "
        "outcome, order, invariants and final state; cite current observation/action evidence "
        "and identify unmet or unknown items",
        "- distinguish inferred steps and assumptions from supplied requirements; "
        "do not invent hidden checks. Unknown coverage alone is no reason to stop useful work",
        "- a completed operation or planner goal is not task acceptance; "
        "your completion declaration remains a claim until the authority confirms it",
    ])
    return lines
