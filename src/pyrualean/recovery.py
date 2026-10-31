"""Structured execution failures shared by code tools and agent runtimes.

These reports describe execution, not task success. Recovery belongs to the
planner: the host never replays a failed action or rolls the scene back.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from ._backend import ToolError


def failure_details(
    exc: BaseException, *, stage: str, filename: str | None = None
) -> dict[str, Any]:
    """Describe the original exception, retaining its generated-code location."""
    cause = exc
    if isinstance(cause, SyntaxError):
        kind = "syntax"
    elif isinstance(cause, ToolError):
        kind = "primitive"
    elif isinstance(cause, AssertionError):
        kind = "verification"
    elif isinstance(cause, ImportError):
        kind = "import"
    else:
        kind = "runtime"
    location = None
    if isinstance(cause, SyntaxError):
        location = {"file": cause.filename, "line": cause.lineno, "column": cause.offset}
    else:
        frames = traceback.extract_tb(cause.__traceback__)
        # Prefer the innermost workspace frame, including helpers defined in an
        # earlier cell. Errors inside the robot library point to its call site.
        root = Path(filename).parent if filename else None
        policy_frames = [
            f for f in frames if root is not None and Path(f.filename).is_relative_to(root)
        ]
        frame = (policy_frames or frames)[-1] if frames else None
        if frame is not None:
            location = {"file": frame.filename, "line": frame.lineno}
    return {
        "kind": kind,
        "stage": stage,
        "type": type(cause).__name__,
        "message": str(cause)[:2000],
        "location": location,
    }


def result_is_error(payload: dict[str, Any]) -> bool:
    """Execution failures must agree in MCP, runtime events and tool audits."""
    return bool(
        payload.get("error")
        or payload.get("exception")
        or payload.get("failure")
        or payload.get("status") in ("error", "timeout")
    )
