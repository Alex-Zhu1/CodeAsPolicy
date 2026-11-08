"""Planner-declared action prerequisites and persistent public invariants.

Declarations bind checks to public requirements; they do not establish that a
predicate measures the right physical property or certify collision safety.
"""

from __future__ import annotations

import copy
import inspect
import math

from .monitor import ConstraintCheck, proactive_constraints
from .task_acceptance import normalize_acceptance

_UNBOUND = object()


def _plain(value):
    """JSON evidence without live objects, callable representations or NaNs."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return None


class ActionContractPause(BaseException):
    """Return for evidence/replanning without ending the robot episode."""

    def __init__(self, report):
        if isinstance(report, str):
            report = {"reason": report, "passed": False}
        self.report = copy.deepcopy(report)
        super().__init__(report.get("reason", "action_contract_violation"))


class ActionContract:
    def __init__(self, robo, acceptance=None):
        self.robo = robo
        if isinstance(acceptance, list):
            acceptance = {"requirements": acceptance}
        public = normalize_acceptance({} if acceptance is None else acceptance)
        self._invariants = {
            requirement["id"]: requirement
            for requirement in public["requirements"] if requirement["kind"] == "invariant"
        }
        self._requires = {}
        self._ensures = {}
        self._preserves = {}
        self._constraints = {}
        self._checked = {}
        self._snapshot_revisions = {}
        self._configured = False
        self._configuration_error = None
        self._restore_pending = False
        self._evaluating = False
        self._evaluation_rejection = None
        self.last_report = None

    @property
    def evaluating(self):
        """Robot dispatch must reject stateful calls while this is true."""
        return self._evaluating

    @property
    def constraints(self):
        return copy.deepcopy(self._constraints)

    def _revision(self):
        return getattr(self.robo, "_action_revision", None)

    def _bindings(self, checks):
        if not isinstance(checks, dict):
            raise ValueError("requires, ensures and preserves must map names to checks")
        result = {}
        for name, check in checks.items():
            if not isinstance(name, str) or not name.strip() or len(name) > 160:
                raise ValueError("check names must be nonempty text of at most 160 characters")
            if isinstance(check, ConstraintCheck):
                # Re-registering the same frozen result is not new evidence.
                saved = self._snapshot_revisions.setdefault(id(check), (check, self._revision()))
                result[name] = (copy.deepcopy(check), saved[1])
            elif callable(check):
                try:
                    signature = inspect.signature(check)
                except (TypeError, ValueError):
                    # Some extension callables expose no signature; evaluate once
                    # at the boundary and report a bad invocation there.
                    pass
                else:
                    try:
                        signature.bind()
                    except TypeError as exc:
                        raise ValueError(f"Check {name!r} must accept no arguments") from exc
                result[name] = (check, None)
            else:
                raise ValueError(f"Check {name!r} must be a callable or ConstraintCheck")
        return result

    def configure(self, spec):
        """Replace stage prerequisites/geometry and merge persistent invariants."""
        if self.evaluating:
            self.reject("plan", "contract_change_during_check",
                        "Keep checks read-only; change the contract outside its evaluation.")
        self._configured = True
        try:
            if not isinstance(spec, dict) or set(spec) - {
                "requires", "ensures", "preserves", "constraints",
            }:
                raise ValueError("contract must contain only requires, ensures, preserves and constraints")
            requires = self._bindings(spec.get("requires", {}))
            ensures = self._bindings(spec.get("ensures", {}))
            preserves = self._bindings(spec.get("preserves", {}))
            constraints = spec.get("constraints", {})
            if not isinstance(constraints, dict):
                raise ValueError("constraints must be a dict")
            normalized = proactive_constraints(**constraints)
        except Exception as exc:
            # A caught invalid replacement must not silently authorize motion
            # under the preceding phase's prerequisites.
            self._configuration_error = f"{type(exc).__name__}: {exc}"
            raise
        self._requires = requires
        self._ensures = ensures
        self._preserves.update(preserves)
        self._constraints = copy.deepcopy(normalized)
        self._configuration_error = None
        self._restore_pending = False
        self._checked = {
            key: value for key, value in self._checked.items()
            if key[0] == "preserves" and key[1] not in preserves
        }
        return self.summary()

    def restore(self, summary):
        """Restore declarations only; saved results cannot restore executable evidence."""
        if self.evaluating:
            self.reject("plan", "contract_change_during_check",
                        "Restore contracts outside check evaluation.")
        if not isinstance(summary, dict):
            raise ValueError("contract summary must be a dict")
        names = {}
        for scope in ("requires", "ensures", "preserves"):
            values = summary.get(scope, [])
            if (not isinstance(values, list)
                    or any(not isinstance(name, str) or not name.strip() or len(name) > 160
                           for name in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"restored {scope} must contain distinct nonempty check names")
            names[scope] = values
        constraints = summary.get("constraints", {})
        if not isinstance(constraints, dict):
            raise ValueError("restored constraints must be a dict")
        normalized = proactive_constraints(**constraints)
        self._requires = {name: (_UNBOUND, None) for name in names["requires"]}
        self._ensures = {name: (_UNBOUND, None) for name in names["ensures"]}
        self._preserves = {name: (_UNBOUND, None)
                           for name in dict.fromkeys([*self._preserves, *names["preserves"]])}
        self._constraints = copy.deepcopy(normalized)
        self._configured = bool(summary.get("configured") or self._requires or self._ensures
                                or self._preserves or self._constraints)
        self._restore_pending = self._configured
        self._configuration_error = None
        self._checked = {}
        self.last_report = None
        return self.summary()

    @staticmethod
    def _failure(name, state, reason, remedy, *, scope, metric="predicate"):
        return dict(assumption=name, name=name, scope=scope, passed=False,
                    evidence_state=state, reason=reason, remedy=remedy,
                    metric=metric, value=None, bound=None, delta=None, elements=[])

    def _report(self, tool, stage, checks):
        checks = _plain(checks)
        failures = [check for check in checks if not check["passed"]]
        report = dict(
            schema="robouse.action_contract.v1", tool=tool, stage=stage,
            action_revision=self._revision(), passed=not failures,
            status="blocked" if failures else "checked", checks=checks, failures=failures,
            uncovered_invariants=sorted(set(self._invariants) - set(self._preserves)),
            coverage_meaning="Named checks are declared; physical safety is not certified.",
        )
        if failures:
            first = failures[0]
            report.update(failed_assumption=first["assumption"],
                          evidence_state=first["evidence_state"],
                          reason=first["reason"], remedy=first["remedy"])
        self.last_report = _plain(report)
        return copy.deepcopy(self.last_report)

    def reject(self, tool, reason, remedy, *, stage="before_action"):
        """Record a dispatch/contract rejection using the same pause protocol."""
        failure = self._failure(reason, "error", reason, remedy, scope="dispatch")
        if self.evaluating:
            self._evaluation_rejection = failure
        raise ActionContractPause(self._report(tool, stage, [failure]))

    def _check(self, name, binding, scope):
        check, registered_revision = binding
        if check is _UNBOUND:
            return self._failure(
                name, "unknown", "restored_check_unbound",
                "Explicitly bind a new current check with plan(contract=...) before moving.",
                scope=scope,
            )
        if isinstance(check, ConstraintCheck) and (
            registered_revision is None or registered_revision != self._revision()
        ):
            return self._failure(
                name, "stale", "static_check_stale",
                "Re-observe and register a new check, or use a callable reading current evidence.",
                scope=scope, metric="action_revision",
            ) | dict(value=self._revision(), bound=registered_revision)
        try:
            value = check() if callable(check) else check
            if self._evaluation_rejection is not None:
                return dict(self._evaluation_rejection, assumption=name, name=name, scope=scope)
            if type(value) is bool:
                return dict(assumption=name, name=name, scope=scope, passed=value,
                            evidence_state="passed" if value else "failed", metric="predicate",
                            value=value, bound=True, delta=None, elements=[],
                            reason=None if value else "predicate_false",
                            remedy=None if value else "Re-establish this assumption before dependent motion.")
            if isinstance(value, ConstraintCheck) and type(value.passed) is bool:
                # Evidence helpers use explicit status prefixes; retain their
                # distinction between a false predicate and unavailable evidence.
                prefix = (value.reason or "").partition(":")[0]
                evidence_state = {"unknown": "unknown", "stale": "stale",
                                  "evaluation_error": "error",
                                  "insufficient_evidence_space": "unknown"}.get(prefix, "failed")
                return dict(value.to_dict(), assumption=name, scope=scope,
                            evidence_state="passed" if value.passed else evidence_state,
                            reason=value.reason or (None if value.passed else value.metric),
                            remedy=value.remedy or (None if value.passed else
                                "Re-establish this assumption before dependent motion."))
            return self._failure(
                name, "unknown", "check_did_not_establish_truth",
                "Obtain current evidence and return bool or monitor.ConstraintCheck.", scope=scope,
            )
        except ActionContractPause:
            failure = self._evaluation_rejection or self._failure(
                name, "error", "check_interrupted",
                "Keep checks read-only and inspect their current evidence.", scope=scope,
            )
            return dict(failure, assumption=name, name=name, scope=scope)
        except Exception as exc:
            return self._failure(
                name, "error", f"{type(exc).__name__}: {exc}",
                "Repair the check or its evidence before dependent motion.",
                scope=scope, metric="evaluation_error",
            )

    def _evaluate(self, tool, stage):
        if self.evaluating:
            self.reject(tool, "reentrant_action",
                        "Checks may read observations but must not dispatch actions.", stage=stage)
        checks = []
        if stage != "after_action":
            if self._restore_pending:
                checks.append(self._failure(
                    "contract", "unknown", "restored_contract_requires_rebinding",
                    "Re-observe and explicitly configure this contract; saved checks are historical.",
                    scope="configuration",
                ))
            if self._configuration_error is not None:
                checks.append(self._failure(
                    "contract", "error", self._configuration_error,
                    "Configure a valid contract before moving.", scope="configuration",
                ))
            for name in sorted(set(self._invariants) - set(self._preserves)):
                checks.append(self._failure(
                    name, "missing", "invariant_not_declared",
                    "Bind a current check for this public invariant in contract.preserves.",
                    scope="preserves",
                ))
        self._evaluating = True
        self._evaluation_rejection = None
        try:
            groups = [("requires", self._requires)] if stage == "before_action" else []
            if stage == "verify_completion":
                groups.append(("ensures", self._ensures))
            groups.append(("preserves", self._preserves))
            for scope, bindings in groups:
                for name, binding in bindings.items():
                    outcome = self._check(name, binding, scope)
                    checks.append(outcome)
                    self._checked[(scope, name)] = dict(outcome, action_revision=self._revision(),
                                                        stage=stage)
        finally:
            self._evaluating = False
            self._evaluation_rejection = None
        report = self._report(tool, stage, checks)
        if not report["passed"]:
            raise ActionContractPause(report)
        return report

    def before_action(self, tool, kwargs=None):
        return self._evaluate(tool, "before_action")

    def after_action(self, tool, kwargs=None):
        return self._evaluate(tool, "after_action")

    def verify_completion(self, tool="plan"):
        """Check declared outcomes; a pass is not an environment success verdict."""
        return self._evaluate(tool, "verify_completion")

    def summary(self):
        checked = [dict(value, current=value["action_revision"] == self._revision())
                   for value in self._checked.values()]
        return _plain(dict(
            schema="robouse.action_contract.v1",
            configured=self._configured, evaluating=self.evaluating,
            requires=list(self._requires), ensures=list(self._ensures), preserves=list(self._preserves),
            declared=dict(requires=list(self._requires), ensures=list(self._ensures),
                          preserves=list(self._preserves)),
            checked=checked, constraints=self.constraints,
            required_invariants=copy.deepcopy(list(self._invariants.values())),
            uncovered_invariants=sorted(set(self._invariants) - set(self._preserves)),
            coverage_meaning="Declaration coverage only; no physical safety certification.",
            configuration_error=self._configuration_error, restore_pending=self._restore_pending,
            last_report=self.last_report,
        ))
