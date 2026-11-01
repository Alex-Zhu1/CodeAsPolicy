"""Best-effort, episode-local timing metadata shared by hosts and runtimes.

Callers must select metadata fields explicitly: never pass prompts, code, tool
arguments/results, credentials, arbitrary provider options or exception text.
The trace does not grant execution permission or change any control decision.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA = "pyrualean.interaction.v1"


class InteractionTrace:
    """Append timing events without making trace-write failures task failures.

    ``t`` is elapsed monotonic seconds in this trace's clock domain. The domain
    includes a unique origin: timestamps from separate traces must not be
    subtracted, even when they run in the same process. Failed writes are counted
    in :meth:`summary`; their exception messages are deliberately not retained.
    """

    def __init__(
        self,
        path: Path,
        *,
        session_id: str | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.path = Path(path).resolve()
        self.session_id = session_id if session_id is not None else uuid.uuid4().hex
        self._identity = uuid.uuid4().hex
        self.clock_domain = f"monotonic:{os.getpid()}:{self._identity}"
        self._clock = clock
        self._origin = clock()
        self._last_t = 0.0
        self._lock = threading.Lock()
        self._sequence = 0
        self._write_errors = 0

    def _next_id(self, prefix: str) -> str:
        self._sequence += 1
        return f"{prefix}_{self._identity}_{self._sequence}"

    def new_id(self, prefix: str) -> str:
        with self._lock:
            return self._next_id(prefix)

    def emit(self, event: str, **fields: Any) -> None:
        """Write one event using only caller-selected, non-sensitive metadata."""
        with self._lock:
            try:
                self._last_t = max(self._last_t, self._clock() - self._origin)
                record = {
                    **fields,
                    "schema": SCHEMA,
                    "event": event,
                    "event_id": self._next_id("event"),
                    "session_id": self.session_id,
                    "clock_domain": self.clock_domain,
                    "t": self._last_t,
                }
                line = json.dumps(record, ensure_ascii=False, allow_nan=False)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except Exception:
                # Instrumentation failure must not interrupt an in-flight tool.
                # Exception strings can contain paths, credentials or payloads.
                self._write_errors += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self.path),
                "schema": SCHEMA,
                "session_id": self.session_id,
                "clock_domain": self.clock_domain,
                "write_errors": self._write_errors,
            }


__all__ = ["InteractionTrace", "SCHEMA"]
