"""Live public-evidence checks for action contracts, without simulator truth."""

from __future__ import annotations

import copy
import math


def _check(name, arm, passed, *, metric="hold_evidence", value=None, bound=None,
           delta=None, reason=None, remedy=None):
    # A local import lets monitor expose this helper without a circular import.
    from .monitor import ConstraintCheck

    return ConstraintCheck(name=name, kind="reactive", passed=passed, metric=metric,
                           value=value, bound=bound, delta=delta,
                           elements=(f"{arm}_gripper",), reason=reason, remedy=remedy)


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def check_hold(robo, reference, arm, *, space="spatial", max_drift_m=0.015,
               name="hold_retention"):
    """Read current hold evidence and require its requested dimensional coverage.

    ``space='image'`` accepts consistent wrist projections or observed 3D following.
    ``space='spatial'`` requires observed 3D following. Neither certifies object
    identity, force, unobserved rotations, or future retention. The host validates
    the live source; a dict's ``current`` flag alone never establishes freshness.
    Use in a zero-argument contract callback so every boundary re-reads evidence.
    """
    if arm not in ("left", "right"):
        raise ValueError("arm must be left or right")
    if space not in ("image", "spatial"):
        raise ValueError("space must be image or spatial")
    if not _number(max_drift_m) or max_drift_m <= 0:
        raise ValueError("max_drift_m must be finite and positive")

    def fail(reason, remedy="Obtain fresh, comparable object evidence before dependent motion."):
        return _check(name, arm, False, reason=reason, remedy=remedy)

    try:
        evidence = robo.observe_hold(reference, arm=arm)
        if not isinstance(evidence, dict):
            return fail("unknown: observe_hold returned no evidence dictionary")
        validator = getattr(robo, "perception_is_current", None)
        if not callable(validator):
            return fail("unknown: robot cannot validate the current perception source")
        if not validator(evidence) or evidence.get("current") is False:
            return fail("stale: hold evidence does not match the live observation source")
    except Exception as exc:
        return fail(f"evaluation_error: {type(exc).__name__}: {exc}",
                    "Repair observation access and obtain current hold evidence before moving.")

    if evidence.get("arm") != arm:
        return fail("unknown: hold evidence belongs to another arm")
    if evidence.get("baseline_ready") is not True:
        return fail("unknown: no usable post-closure comparison baseline",
                    "Repair the original capture or revise the view/grasp; do not erase drift.")
    state = evidence.get("state")
    if state not in ("following", "image_consistent") or evidence.get("needs_observation") is True:
        return fail(f"{state or 'unknown'}: {evidence.get('reason') or 'hold evidence insufficient'}")
    gripper = evidence.get("gripper")
    if not isinstance(gripper, dict) or gripper.get("obstructed_closure") is not True:
        return fail("conflicting_evidence: no current obstructed closure")

    if evidence.get("evidence_space") == "3d" and state == "following":
        features = evidence.get("features")
        if not isinstance(features, dict) or not features:
            return fail("unknown: following lacks measured 3D feature drift")
        drifts = []
        for feature in features.values():
            drift = feature.get("relative_change_m") if isinstance(feature, dict) else None
            if not _number(drift) or drift < 0:
                return fail("unknown: 3D feature drift is missing or invalid")
            drifts.append(drift)
        drift = max(drifts)
        passed = drift <= max_drift_m
        return _check(
            name, arm, passed, metric="feature_drift_m", value=drift, bound=max_drift_m,
            delta=max_drift_m - drift,
            reason=None if passed else "relative_motion: observed 3D drift exceeds the bound",
            remedy=None if passed else "Re-observe the object and revise its support or grasp.",
        )

    if space == "spatial":
        return fail("insufficient_evidence_space: image evidence cannot establish 3D following",
                    "Obtain current 3D feature evidence for this spatial prerequisite.")
    if evidence.get("evidence_space") != "wrist_image_2d" or state != "image_consistent":
        return fail("unknown: evidence does not establish wrist-image consistency")
    features = evidence.get("image_features")
    if not isinstance(features, dict) or len(features) < 2:
        return fail("unknown: at least two comparable wrist-image features are required")
    comparisons = []
    for feature in features.values():
        if not isinstance(feature, dict) or feature.get("camera") != f"cam_{arm}_wrist":
            return fail("unknown: image features lack the matching calibrated wrist camera")
        drift, bound = feature.get("displacement_px"), feature.get("comparison_bound_px")
        if not _number(drift) or drift < 0 or not _number(bound) or bound <= 0:
            return fail("unknown: image feature comparison is missing or invalid")
        comparisons.append((bound - drift, drift, bound))
    margin, drift, bound = min(comparisons)
    passed = margin >= 0
    return _check(
        name, arm, passed, metric="feature_drift_px", value=drift, bound=bound, delta=margin,
        reason=None if passed else "image_relative_motion: a wrist feature exceeds its bound",
        remedy=None if passed else "Re-observe the original features and revise the grasp if needed.",
    )


def check_fact(robo, fact_id, expected="yes"):
    """Reuse a current supervisor answer without issuing another visual request.

    A ledger's ``fresh`` flag is necessary but insufficient: its source must be
    registered by this robot and still match the live source and action revision.
    Unknown, unbound or stale evidence never satisfies either a yes or no check.
    This checks a fallible visual answer, not a task or physical-safety verdict.
    """
    from .monitor import ConstraintCheck

    if not isinstance(fact_id, str) or not fact_id.strip():
        raise ValueError("fact_id must be nonempty text")
    if expected not in ("yes", "no"):
        raise ValueError("expected must be yes or no")

    def result(passed, reason=None, *, unknown=False):
        return ConstraintCheck(
            name=fact_id, kind="reactive", passed=passed, metric="visual_fact_match",
            value=None if unknown else float(passed), bound=1.,
            delta=None if unknown else (0. if passed else -1.), elements=(fact_id,),
            reason=reason,
            remedy=None if passed else (
                "Declare this fact with watch and verify a current host observation before "
                "dependent motion; revise the action if the observed answer contradicts it."
            ),
        )

    def unknown(reason):
        return result(False, "unknown: " + reason, unknown=True)

    supervisor = getattr(robo, "_supervisor", None)
    if supervisor is None:
        return unknown("visual supervisor is unavailable")
    try:
        fact = supervisor.read_evidence().get("facts", {}).get(fact_id)
        if not isinstance(fact, dict):
            return unknown("visual fact has not been declared")
        if fact.get("fresh") is not True:
            return unknown("visual fact is stale or has not been observed")
        observation = fact.get("observation")
        if not isinstance(observation, dict):
            return unknown("visual fact has no observation source")
        source = observation.get("observation_id")
        registered = getattr(robo, "_observations", {}).get(source)
        validator = getattr(robo, "_observation_is_current", None)
        if not isinstance(registered, dict) or not callable(validator):
            return unknown("visual fact has no host-registered observation")
        registered = copy.deepcopy(registered)
        for key in ("session_id", "source_id", "observation_id"):
            if not observation.get(key) or observation[key] != registered.get(key):
                return unknown("visual fact observation identity does not match the host source")
        revision = observation.get("action_revision")
        if revision is None:
            revision = observation.get("state_version")
        if revision is None or revision != getattr(robo, "_action_revision", None):
            return unknown("visual fact predates the current action revision")
        if not validator(registered):
            return unknown("visual fact observation is stale, unaligned or unresolved")
        value = fact.get("value")
        if value not in ("yes", "no"):
            return unknown("current visual answer does not establish yes or no")
    except Exception as exc:
        return unknown(f"source validation failed: {type(exc).__name__}: {exc}")
    return result(value == expected,
                  None if value == expected else f"visual_fact_mismatch: observed {value}, expected {expected}")
