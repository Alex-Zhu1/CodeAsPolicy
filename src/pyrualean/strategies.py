"""Small feedback loops extracted from the RSI relocalization/correction pattern.

These helpers contain no task geometry or robot-specific primitive calls.
Callbacks use the current robot's public API. Exceptions and host pauses
propagate to the outer model loop; no helper replays a failed code block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyResult:
    status: str  # verified | replan; never an episode-success verdict
    reason: str
    attempts: int
    observation: Any
    last_result: Any = None


@dataclass(frozen=True)
class GraspObservation:
    """Current object-specific evidence, plus optional public sensor/image data.

    holding is fresh gripper/contact evidence, not proof of object identity.
    object_held is a visual predicate for the intended object. Unknown is None.
    Only fresh observations with IDs and increasing state versions can verify
    a probe. Callers must not reuse a close result as a later sensor reading.
    """

    observation_id: str | None
    state_version: int | None
    holding: bool | None
    object_held: bool | None
    fresh: bool = False
    data: Any = None
    fact: str | None = None
    question_version: int | None = None

    def __post_init__(self):
        for name in ("holding", "object_held"):
            _verdict(getattr(self, name), name)
        if type(self.fresh) is not bool:
            raise TypeError("fresh must be bool")
        if self.observation_id is not None and not isinstance(self.observation_id, str):
            raise TypeError("observation_id must be a string or None")
        if self.state_version is not None and (
            type(self.state_version) is not int or self.state_version < 0
        ):
            raise ValueError("state_version must be a nonnegative integer or None")

    @classmethod
    def from_evidence(cls, ledger, fact, *, holding=None, data=None):
        """Read a detached evidence() snapshot; never promote unknown/stale to yes."""
        record = ledger.get("facts", {}).get(fact, {})
        source = record.get("observation") or {}
        value = record.get("value")
        fresh = record.get("fresh") is True
        return cls(source.get("observation_id"), source.get("state_version"), holding,
                   {"yes": True, "no": False}.get(value) if fresh else None,
                   fresh=fresh, data=data, fact=fact, question_version=record.get("version"))


@dataclass(frozen=True)
class GraspResult:
    status: str  # verified | failed | uncertain; never an episode-success verdict
    reason: str
    attempts: int
    observation: GraspObservation | None
    before_probe: GraspObservation | None = None
    last_result: Any = None


def _verdict(value, name):
    if value is not None and type(value) is not bool:
        raise TypeError(f"{name} must return bool or None")
    return value


def verify_grasp(observe, close, probe, compare, *, repair=None, max_attempts=2,
                 allow_action=None):
    """Close, test support with one small probe, then verify fresh paired evidence.

    close(attempt) returns the controller result; probe(closed, attempt) performs
    one bounded, safe diagnostic movement. observe() returns GraspObservation.
    compare(closed, after) checks whether the intended object followed the
    gripper and stayed stable relative to it, using permitted before/after data.
    It returns bool/None. A fresh object_held=True is also required to verify.

    Only a definite failed grasp can use repair(observation, next_attempt).
    Fresh contact can permit a small probe while visual support is unknown.
    Unresolved support after the probe, stale evidence and refused motion exit
    without repair.
    Optional allow_action(kind, attempt) must return bool and is checked before
    close/probe/repair, e.g. to reserve steps/time for handover and insertion.
    Callbacks must bound their own primitive costs; this is not a motion planner
    or a replacement for host timeouts, watch contracts and phase prerequisites.
    Exceptions and supervision pauses propagate without automatic replay.
    """
    _count(max_attempts, "max_attempts")
    current = before = last = None

    def permitted(kind, attempt):
        value = True if allow_action is None else allow_action(kind, attempt)
        if type(value) is not bool:
            raise TypeError("allow_action must return bool")
        return value

    def read():
        value = observe()
        if not isinstance(value, GraspObservation):
            raise TypeError("observe must return GraspObservation")
        return value

    def fresh(value):
        return value.fresh and bool(value.observation_id) and value.state_version is not None

    def result(status, reason, attempts):
        return GraspResult(status, reason, attempts, current, before, last)

    for attempt in range(1, max_attempts + 1):
        if not permitted("close", attempt):
            return result("uncertain", "phase_budget", attempt - 1)
        last = close(attempt)
        current = read()
        if getattr(last, "reached", None) is False:
            return result("failed", "close_not_reached", attempt)
        closed_holding = _verdict(getattr(last, "holding", None), "close.holding")
        failed = closed_holding is False or (
            fresh(current) and (current.holding is False or current.object_held is False)
        )
        if not failed:
            if not fresh(current):
                return result("uncertain", "stale_evidence", attempt)
            if not (current.holding is True or closed_holding is True
                    or current.object_held is True):
                return result("uncertain", "contact_unconfirmed", attempt)
            before = current
            if not permitted("probe", attempt):
                return result("uncertain", "phase_budget", attempt)
            last = probe(before, attempt)
            current = read()
            if getattr(last, "reached", None) is False:
                return result("failed", "probe_not_reached", attempt)
            if (not fresh(current) or current.observation_id == before.observation_id
                    or current.state_version <= before.state_version):
                return result("uncertain", "stale_evidence", attempt)
            if (current.fact, current.question_version) != (before.fact, before.question_version):
                return result("uncertain", "predicate_changed", attempt)
            follows = _verdict(compare(before, current), "compare")
            failed = (current.holding is False or current.object_held is False
                      or follows is False)
            if not failed:
                if current.object_held is True and follows is True:
                    return result("verified", "object_followed_and_held", attempt)
                return result("uncertain", "support_unconfirmed", attempt)
        if repair is None or attempt == max_attempts:
            return result("failed", "grasp_not_established", attempt)
        if not permitted("repair", attempt + 1):
            return result("failed", "phase_budget", attempt)
        last = repair(current, attempt + 1)
        if getattr(last, "reached", None) is False:
            return result("failed", "repair_not_reached", attempt)
        before = None


def _count(value, name):
    if type(value) is not int or not 1 <= value <= 32:
        raise ValueError(f"{name} must be an integer in 1..32")


def relocalize(observe, valid, *, prepare_view=None, max_attempts=3):
    """Observe up to max_attempts times; optionally change view between reads.

    observe() returns a NEW observation. valid(observation) must return bool
    or None (unknown). prepare_view(previous, attempt) runs before each repeat
    read, with attempt starting at 1. It may clear occlusion or select another
    camera. Callers must check status before using the returned observation.
    """
    _count(max_attempts, "max_attempts")
    observation = None
    for attempt in range(max_attempts):
        if attempt and prepare_view is not None:
            prepare_view(observation, attempt)
        observation = observe()
        verdict = valid(observation)
        if verdict is not None and type(verdict) is not bool:
            raise TypeError("valid must return bool or None")
        if verdict is True:
            return StrategyResult("verified", "localized", attempt + 1, observation)
        if prepare_view is None:
            return StrategyResult("replan", "observation_uncertain", attempt + 1, observation)
    return StrategyResult("replan", "observation_uncertain", max_attempts, observation)


def correct(observe, complete, act, *, max_attempts=3, error=None,
            min_progress=0.001, no_progress_limit=2):
    """Bounded correction or grasp attempts, verified after EVERY attempt.

    observe() reads current evidence; complete(observation) returns bool/None.
    act(observation, attempt) performs one bounded correction (attempt starts
    at 1). error(observation), when supplied, is a finite nonnegative residual
    in caller-defined units. Stop after no_progress_limit insufficient
    improvements. Unknown completion hands control back without another act.
    A holding predicate must establish the intended object, not just closure.
    """
    _count(max_attempts, "max_attempts")
    _count(no_progress_limit, "no_progress_limit")
    if not math.isfinite(min_progress) or min_progress < 0:
        raise ValueError("min_progress must be finite and nonnegative")
    observation, last, previous, stalled = observe(), None, None, 0
    for attempt in range(max_attempts + 1):
        verdict = complete(observation)
        if verdict is not None and type(verdict) is not bool:
            raise TypeError("complete must return bool or None")
        if verdict is True:
            return StrategyResult("verified", "postcondition", attempt, observation, last)
        if verdict is None:
            return StrategyResult("replan", "observation_uncertain", attempt, observation, last)
        if error is not None:
            residual = float(error(observation))
            if not math.isfinite(residual) or residual < 0:
                return StrategyResult("replan", "invalid_residual", attempt, observation, last)
            if previous is not None:
                improvement = previous - residual
                stalled = stalled + 1 if improvement <= min_progress else 0
                if stalled >= no_progress_limit:
                    return StrategyResult("replan", "no_progress", attempt, observation, last)
            previous = residual
        if attempt == max_attempts:
            return StrategyResult("replan", "attempt_limit", attempt, observation, last)
        last = act(observation, attempt + 1)
        observation = observe()


PROMPT = """# Optional local loops

Ordinary Python branches and bounded loops are valid. `strategy` is optional
(no import needed); extra arguments below are keyword-only:
- strategy.relocalize(observe, valid, prepare_view=None, max_attempts=3):
  observe() reads fresh evidence; valid(observation) returns bool/None.
  prepare_view(previous, attempt) changes the view before another observation.
- strategy.correct(observe, complete, act, max_attempts=3, error=None,
  min_progress=0.001, no_progress_limit=2): observe before and after each
  act(observation, attempt); complete(observation) returns bool/None;
  error(observation) optionally supplies residual error to detect no progress.
Both return status ('verified' or 'replan'), reason, attempts, observation and
last_result. 'verified' establishes only your predicate, not task success.
Inspect a 'replan' result before dependent work; yield for new model judgment.

strategy.verify_grasp(observe, close, probe, compare, repair=None,
max_attempts=2, allow_action=None) is an optional paired-evidence grasp check.
observe() returns strategy.GraspObservation(observation_id, state_version,
holding, object_held, fresh=False, data=None, fact=None, question_version=None).
close(attempt) closes once;
probe(closed_observation, attempt) makes a bounded diagnostic motion;
compare(before, after) returns bool/None for the intended object following the
gripper without slipping. This helper verifies only with fresh object-held
evidence and a positive comparison; holding alone is insufficient. Do not
invent evidence or reuse old holding measurements. It returns status
('verified'|'failed'|'uncertain'), reason, attempts, observation, before_probe
and last_result. repair(observation, next_attempt) handles definite failure
only; allow_action(kind, attempt) can reserve budget before close/probe/repair.
These are this optional helper's conditions, not a general motion permission.
Helpers do not change explicit prerequisites or execution-unknown guards.
Exceptions and host pauses propagate without replay.
"""
