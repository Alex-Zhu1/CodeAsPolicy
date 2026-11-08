"""Shared defaults for interactive play, including its CLI and host process.

These are launch settings, not task recipes or native simulator budgets.
"""

from argparse import ArgumentTypeError

ARM = "cells"
RUNTIME = "codex"
MODEL = "gpt-6-astra"
REASONING = "high"
BUDGET_S = 5400.0
MAX_TURNS = 100
IMAGES = "on-demand"
FEEDBACK = "pure"
PRIMITIVES = "no-vla"


def turn_budget(value: str) -> int | None:
    """Zero selects a wall-clock-only budget; negative counts are invalid."""

    try:
        count = int(value)
    except ValueError as exc:
        raise ArgumentTypeError("max-turns must be a nonnegative integer") from exc
    if count < 0:
        raise ArgumentTypeError("max-turns must be a nonnegative integer")
    return count or None
