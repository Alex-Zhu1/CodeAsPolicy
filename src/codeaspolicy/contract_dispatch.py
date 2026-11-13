"""Carry policy contracts through prepared operations without weakening bounds."""

from __future__ import annotations

import copy


PREPARATIONS = {"intent_prepare": "motion", "local_prepare": "motion", "grasp_prepare": "grasp"}


def merge_constraints(inherited, explicit):
    """Combine independent bounds; conflicting definitions require a new plan.

    This intentionally does not guess whether two different orientation frames or
    contact exceptions are equivalent. Obstacles accumulate; per-arm maps merge.
    """
    result = copy.deepcopy(inherited)
    for key, value in explicit.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
        elif key == "keep_out":
            for obstacle in value:
                if obstacle not in result[key]:
                    result[key].append(copy.deepcopy(obstacle))
        elif result[key] == value:
            continue
        elif key in {"z_bounds", "max_tilt_deg", "corridor_m"} and (
            isinstance(result[key], dict) and isinstance(value, dict)
        ):
            for arm, bound in value.items():
                if arm in result[key] and result[key][arm] != bound:
                    raise ValueError(f"conflicting inherited constraint: {key}.{arm}")
                result[key][arm] = copy.deepcopy(bound)
        else:
            raise ValueError(f"conflicting inherited constraint: {key}")
    # Contact permissions use obstacle indices. Merging obstacle lists must not
    # silently redirect an explicit exception to a different inherited obstacle.
    contacts = (explicit.get("geometry") or {}).get("allowed_contacts", [])
    original = explicit.get("keep_out", [])
    merged = result.get("keep_out", [])
    for contact in contacts:
        for identity in contact["pair"]:
            if identity.startswith("keep_out:"):
                index = int(identity.split(":", 1)[1])
                if index >= len(original) or index >= len(merged) or original[index] != merged[index]:
                    raise ValueError("merging keep_out would change an allowed contact's obstacle; "
                                     "declare the complete obstacle/contact contract in plan")
    return result


def prepare(robo, tool, kwargs):
    contract = robo._action_contract
    if contract is None or tool not in PREPARATIONS:
        return kwargs
    inherited = contract.constraints
    if not inherited:
        return kwargs
    try:
        combined = merge_constraints(inherited, kwargs.get("constraints") or {})
    except ValueError as exc:
        contract.reject(tool, str(exc), "Replan with compatible bounds; inherited constraints cannot be overridden.")
    return dict(kwargs, constraints=combined)


def remember(robo, tool, kwargs, result):
    if tool not in PREPARATIONS or kwargs.get("preview") or result.get("status") == "rejected":
        return
    kind = PREPARATIONS[tool]
    identity = kwargs.get("grasp_id" if kind == "grasp" else "intent_id")
    # An old operation ID must not acquire a new contract through a repeated read.
    robo._contract_operations.setdefault((kind, identity), copy.deepcopy(kwargs.get("constraints") or {}))


def before_dispatch(robo, tool, kwargs):
    contract = robo._action_contract
    if contract is None:
        return
    if contract.evaluating:
        contract.reject(tool, "action_inside_contract_check", "Contract checks may only read public observations.")
    inherited = contract.constraints
    if inherited:
        if tool == "operation_resume":
            operation = kwargs.get("operation") or {}
            key = (operation.get("kind"), operation.get("id"))
        elif tool == "grasp_step":
            key = ("grasp", kwargs.get("grasp_id"))
        elif tool == "intent_execute":
            key = ("motion", kwargs.get("intent_id"))
        else:
            contract.reject(tool, "unsupported_constraint_dispatch",
                            "Use an intent or staged grasp that enforces the active geometry constraints.")
        frozen = robo._contract_operations.get(key)
        try:
            covered = frozen is not None and merge_constraints(inherited, frozen) == frozen
        except ValueError:
            covered = False
        if not covered:
            contract.reject(tool, "prepared_constraints_changed",
                            "Prepare a new operation under the current constraints; an old ID cannot adopt new bounds.")
    contract.before_action(tool, kwargs)
