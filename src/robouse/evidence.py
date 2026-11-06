"""Episode-local visual evidence and declared phase prerequisites.

The planner names dependencies and invalidators; the host preserves them and
checks their evidence. This is not inference of physical action dependencies.
"""

from __future__ import annotations

import copy


class EvidenceLedger:
    MAX_FACTS = 64
    MAX_PHASES = 64

    def __init__(self):
        self.facts: dict = {}
        self.requirements: dict = {}

    def configure(self, phase, facts, requires, invalidates):
        """Validate the entire declaration before changing persistent records."""
        requires = {} if requires is None else requires
        known = set(self.facts) | set(facts)
        if len(known) > self.MAX_FACTS:
            raise ValueError(f"At most {self.MAX_FACTS} visual facts per episode")
        if (
            not isinstance(requires, dict)
            or len(requires) > 12
            or set(requires) - known
            or any(value not in ("yes", "no") for value in requires.values())
        ):
            raise ValueError("requires must map at most 12 known fact IDs to yes/no")
        if invalidates is not None and (
            not isinstance(invalidates, (list, tuple))
            or any(not isinstance(key, str) or key not in known for key in invalidates)
            or len(set(invalidates)) != len(invalidates)
        ):
            raise ValueError("invalidates must list distinct known fact IDs, or be None (all)")
        gates = copy.deepcopy(self.requirements.get(phase, {}))
        for key, expected in requires.items():
            if key in gates and gates[key]["expected"] != expected:
                raise ValueError(f"Conflicting prerequisite for {phase}: {key}")
            gates.setdefault(key, dict(expected=expected, reason="phase prerequisite"))
        self._validate_gates(phase, gates)
        pinned = {key for requirements in self.requirements.values() for key in requirements}
        for key, question in facts.items():
            old = self.facts.get(key)
            if old and old["question"] != question and key in pinned:
                raise ValueError(f"Fact {key} has a registered prerequisite; keep its question")

        for key, question in facts.items():
            old = self.facts.get(key)
            if old is None or old["question"] != question:
                self.facts[key] = dict(
                    question=question,
                    version=1 if old is None else old["version"] + 1,
                    value="unknown",
                    fresh=False,
                    observation=None,
                    probabilities=None,
                    unknown_streak=0,
                    source_id=None,
                    invalidated_by="not_observed" if old is None else "question_changed",
                    invalidated_by_action_id=None,
                )
        if gates:
            self.requirements[phase] = gates

    def _validate_gates(self, phase, gates):
        if len(gates) > 12:
            raise ValueError("At most 12 prerequisites per phase")
        if gates and phase not in self.requirements and len(self.requirements) >= self.MAX_PHASES:
            raise ValueError(f"At most {self.MAX_PHASES} phases with prerequisites")

    def defer(self, fact, *, until, reason, expected="yes"):
        """Register a persistent gate at the named future phase; never change truth."""
        if not isinstance(fact, str) or fact not in self.facts:
            raise ValueError("defer_fact requires a known fact ID")
        if not isinstance(until, str) or not 1 <= len(until) <= 200:
            raise ValueError("until must name a phase (1..200 characters)")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
            raise ValueError("Deferral needs a concrete reason (1..500 characters)")
        if expected not in ("yes", "no"):
            raise ValueError("expected must be yes or no")
        gates = copy.deepcopy(self.requirements.get(until, {}))
        if fact in gates and gates[fact]["expected"] != expected:
            raise ValueError(f"Conflicting prerequisite for {until}: {fact}")
        gates[fact] = dict(expected=expected, reason=reason)
        self._validate_gates(until, gates)
        self.requirements[until] = gates

    def invalidate(self, keys, reason, *, action_id=None):
        for key in self.facts if keys is None else keys:
            self.facts[key]["fresh"] = False
            self.facts[key]["invalidated_by"] = reason
            self.facts[key]["invalidated_by_action_id"] = action_id

    def observe(self, values, probabilities, observation):
        """Accept current evidence once per source, retaining uncertainty across cells.

        A new request or a second inference of the same image is not an
        independent observation. Repeated sources may restore applicability
        after a proven zero-effect action, but cannot change the recorded vote.
        """
        source = {key: observation.get(key) for key in (
            "observation_id", "source_id", "session_id", "step", "step_id",
            "state_version", "action_revision", "captured_at", "recorded_at",
            "source_freshness", "action_id",
        )}
        source_id = observation.get("source_id") or observation.get("observation_id")
        if not source_id:
            raise ValueError("Visual evidence needs an observation source identity")
        novel = {}
        for key, value in values.items():
            fact = self.facts[key]
            novel[key] = fact["source_id"] != source_id
            if not novel[key]:
                fact.update(fresh=True, invalidated_by=None, invalidated_by_action_id=None)
                continue
            fact.update(
                value=value, fresh=True, observation=copy.deepcopy(source),
                probabilities=dict(probabilities[key]), invalidated_by=None,
                invalidated_by_action_id=None,
                source_id=source_id,
                unknown_streak=fact["unknown_streak"] + 1 if value == "unknown" else 0,
            )
        return novel

    def blockers(self, phase):
        blocked = {}
        for key, gate in self.requirements.get(phase, {}).items():
            fact = self.facts[key]
            if not fact["fresh"] or fact["value"] != gate["expected"]:
                blocked[key] = dict(
                    expected=gate["expected"], value=fact["value"], fresh=fact["fresh"],
                    invalidated_by=fact["invalidated_by"], observation=fact["observation"],
                    invalidated_by_action_id=fact["invalidated_by_action_id"],
                )
        return blocked

    def snapshot(self):
        return copy.deepcopy(dict(facts=self.facts, phase_requirements=self.requirements))

    def summary(self):
        return copy.deepcopy(dict(
            facts={key: {k: v for k, v in fact.items() if k not in (
                "question", "probabilities",
            )} for key, fact in self.facts.items()},
            phase_requirements={phase: {key: gate["expected"] for key, gate in gates.items()}
                                for phase, gates in self.requirements.items()},
        ))
