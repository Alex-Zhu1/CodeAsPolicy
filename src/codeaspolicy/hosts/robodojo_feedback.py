"""Shared measured-progress decisions for native, timed and grasp motions.

This layer observes robot tracking, not contact force or object support. Callers
own geometry, dispatch, deadlines and hard invariants. A correction holds the
current target; it never advances the path or replays a completed waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import robodojo_kinematics as kin

MAX_CORRECTIONS = 2


@dataclass(frozen=True)
class TrackingLimits:
    position: float
    orientation: float
    position_ceiling: float | None = None
    orientation_ceiling: float | None = None
    joint: float | None = None
    progress_fraction: float = .1


def _errors(actual, expected):
    return (float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])),
            float(kin.rotation_error(actual[:3, :3], expected[:3, :3])))


class ProgressTracker:
    """One correction allowance per motion, retained across phases/checkpoints."""

    def __init__(self, max_corrections=MAX_CORRECTIONS):
        self.max_corrections = max_corrections
        self.corrections = 0
        self.last = {}

    def corrected(self):
        """Charge a corrective dispatch, separately from deciding to request it."""
        self.corrections += 1

    def summary(self):
        return dict(controller="measured_progress", correction_steps=self.corrections,
                    correction_limit=self.max_corrections, contact_inferred=False, **self.last)

    def assess(self, before, actual, expected, limits, *, can_correct=True,
               spare_steps=0, joints_before=None, joints_actual=None, joints_expected=None):
        """Return advance/correct/stop. Every failing dimension must make progress.

        Bounds scale with the caller's tolerance unless explicitly supplied.
        No correction is permitted within a moving timed spline: the caller
        must stop and replan that trajectory from the measured state instead.
        spare_steps excludes the unexecuted nominal path, preserving its budget.
        """
        errors, misses, stagnant, excessive = {}, [], False, False
        for arm, goal in expected.items():
            old = _errors(before[arm], goal)
            now = _errors(actual[arm], goal)
            tolerances = [limits.position, limits.orientation]
            ceilings = [limits.position_ceiling or 2 * limits.position,
                        limits.orientation_ceiling or 2 * limits.orientation]
            errors[arm] = dict(position_error_m=now[0], orientation_error_deg=now[1])
            if limits.joint is not None:
                old = (*old, float(np.max(np.abs(
                    np.asarray(joints_before[arm]) - joints_expected[arm]))))
                now = (*now, float(np.max(np.abs(
                    np.asarray(joints_actual[arm]) - joints_expected[arm]))))
                tolerances.append(limits.joint)
                ceilings.append(.15)
                errors[arm]["joint_error_rad"] = now[-1]
            failed = False
            for previous, current, tolerance, ceiling in zip(
                    old, now, tolerances, ceilings, strict=True):
                if not np.isfinite(previous) or not np.isfinite(current):
                    failed, excessive = True, True
                elif current > tolerance:
                    failed = True
                    excessive |= current > ceiling
                    stagnant |= previous - current < tolerance * limits.progress_fraction
            if failed:
                misses.append(arm)
        cause = ("within_tolerance" if not misses else
                 "outside_correction_bounds" if excessive else
                 "no_progress" if stagnant else
                 "continuous_trajectory" if not can_correct else
                 "correction_limit" if self.corrections >= self.max_corrections else
                 "insufficient_correction_budget" if spare_steps < 1 else "progressing")
        decision = "advance" if not misses else "correct" if cause == "progressing" else "stop"
        self.last = dict(decision=decision, cause=cause, errors=errors, missed_arms=misses)
        return decision
