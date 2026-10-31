"""Opt-in recovery instructions delivered only after a planner declares stuck."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StuckRecoveryConfig:
    instructions: str
    max_rounds: int = 2

    def __post_init__(self):
        if not self.instructions.strip() or len(self.instructions) > 12_000:
            raise ValueError("stuck recovery instructions must contain 1..12000 characters")
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise ValueError("stuck recovery rounds must be a positive integer")

    def record(self):
        return {
            "max_rounds": self.max_rounds,
            "instructions_sha256": hashlib.sha256(self.instructions.encode()).hexdigest(),
            "trigger": "finish(status=stuck)",
        }

    def prompt(self):
        return (
            "# Confirmed-stuck recovery\n\n"
            "When you confirm that the current plan is stuck, call finish(status='stuck', "
            "summary=...) with the concrete blockage and attempts made. This is the trigger, "
            "not an instruction to load the recovery skill beforehand. Read the tool result: "
            "episode_finished=false means the host recorded your stuck declaration and is "
            "now supplying a recovery skill. Apply it, inspect the current state, and continue "
            "in this same episode. Do not stop your response as if finish had succeeded. "
            f"At most {self.max_rounds} recovery rounds are available, without resetting "
            "any action, call, or time budget. Terminal environments and exhausted budgets "
            "cannot recover. A result with episode_finished=true is final. Success and "
            "failure finish requests remain terminal."
        )


def add_stuck_recovery_arguments(parser):
    parser.add_argument(
        "--stuck-recovery-skill",
        type=Path,
        default=None,
        help="skill text delivered after a confirmed stuck declaration (cells)",
    )
    parser.add_argument(
        "--stuck-recovery-rounds",
        type=int,
        default=2,
        help="maximum recovery dispatches per episode (default 2)",
    )


def stuck_recovery_config(args):
    if args.stuck_recovery_skill is None:
        return None
    return StuckRecoveryConfig(
        args.stuck_recovery_skill.read_text(encoding="utf-8"), args.stuck_recovery_rounds
    )
