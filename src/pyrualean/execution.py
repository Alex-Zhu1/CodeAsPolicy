"""Limits for generated Python, without discarding its persistent namespace.

Tracing interrupts Python loops in policy/workspace frames, including helpers
from earlier cells. Native calls are checked when they return; a process-level
watchdog is still required for uncooperative native code. This is an execution
guard, not an isolation boundary against deliberately hostile Python.
"""

from __future__ import annotations

import io
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ._backend import EpisodeFinished


class ExecutionLimit(BaseException):
    """Yield to the planner; ordinary policy exception handlers must not retry."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_s: float = 120.0
    max_actions: int | None = None
    max_output_chars: int = 8000
    max_display_images: int = 8

    def __post_init__(self):
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("code timeout must be finite and positive")
        for name in ("max_actions", "max_output_chars", "max_display_images"):
            value = getattr(self, name)
            if value is None and name == "max_actions":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def record(self):
        return asdict(self)

    def prompt(self):
        actions = (
            f" At most {self.max_actions} motion attempts per code call."
            if self.max_actions is not None else ""
        )
        return (
            f"Each code call has a {self.timeout_s:g} s execution deadline.{actions} "
            f"Each output stream retains at most {self.max_output_chars} characters; "
            f"display accepts at most {self.max_display_images} images per call. "
            "A limit returns control to you without rolling back actions or assignments. "
            "Inspect current evidence and shorten the remaining work. Do not catch "
            "BaseException to suppress host interruptions. finish is final unless its "
            "result explicitly says episode_finished=false."
        )


class BoundedOutput(io.TextIOBase):
    """Bound memory while writing, not after StringIO has accumulated stdout."""

    def __init__(self, limit: int):
        self.limit = limit
        self.kept = 0
        self.omitted = 0
        self.parts: list[str] = []

    def writable(self):
        return True

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("output must be text")
        take = min(len(text), self.limit - self.kept)
        if take:
            self.parts.append(text[:take])
            self.kept += take
        self.omitted += len(text) - take
        return len(text)

    def getvalue(self):
        return "".join(self.parts)


class ExecutionBudget:
    def __init__(self, limits: ExecutionLimits, workspace: Path, robo):
        self.limits = limits
        self.root = str(workspace.resolve()) + "/"
        self.robo = robo
        self.actions = 0
        self.failure: ExecutionLimit | None = None
        self.deadline = 0.0
        self._files: dict[str, bool] = {}
        self._ticks = 0

    def check(self):
        if self.robo._stop_reason:
            raise EpisodeFinished(self.robo._stop_reason)
        if self.failure is not None:
            raise self.failure
        if time.perf_counter() >= self.deadline:
            self.failure = ExecutionLimit("timeout", "Code execution deadline expired")
            raise self.failure

    def before_action(self):
        self.check()
        if self.limits.max_actions is not None and self.actions >= self.limits.max_actions:
            self.failure = ExecutionLimit("action_limit", "Code motion-attempt limit reached")
            raise self.failure
        self.actions += 1

    def _trace(self, frame, event, arg):
        if event == "call":
            filename = frame.f_code.co_filename
            if filename not in self._files:
                self._files[filename] = str(Path(filename).absolute()).startswith(self.root)
            if not self._files[filename]:
                return None
            # Opcode events also cover a one-line `while True: pass`.
            frame.f_trace_opcodes = True
            self.check()
        self._ticks += 1
        if event in ("line", "return") or self._ticks % 128 == 0:
            self.check()
        return self._trace

    def __enter__(self):
        self.deadline = time.perf_counter() + self.limits.timeout_s
        self.previous_trace = sys.gettrace()
        self.previous_budget = self.robo._execution_budget
        self.robo._execution_budget = self
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous_trace)
        self.robo._execution_budget = self.previous_budget


def add_execution_arguments(parser):
    parser.add_argument("--code-timeout-s", type=float, default=120.0,
                        help="execution deadline per Python cell/program (default 120 s)")
    parser.add_argument("--code-max-actions", type=int, default=None,
                        help="optional maximum motion attempts per code call")
    parser.add_argument("--code-output-chars", type=int, default=8000,
                        help="maximum retained characters per stdout/stderr stream")


def execution_limits(args):
    return ExecutionLimits(timeout_s=args.code_timeout_s, max_actions=args.code_max_actions,
                           max_output_chars=args.code_output_chars)
