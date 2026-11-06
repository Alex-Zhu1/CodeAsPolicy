"""Durable execution identities; never replay Python to reconstruct a session.

The optional SQLite file lives outside the policy workspace. Each write commits
before dispatch/acknowledgement; a crash between them leaves an unresolved action.
Only one host may own a file at a time. Resumption requires the same backend
session, not merely the same task name or directory.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def plain(value):
    """Detached JSON data, without pickles or dynamically imported result types."""
    if dataclasses.is_dataclass(value):
        result = plain(dataclasses.asdict(value))
        if hasattr(value, "reached"):
            result["reached"] = bool(value.reached)
        return result
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError(f"Execution record cannot serialize {type(value).__name__}")


def fingerprint(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


class ExecutionJournal:
    def __init__(self, path: Path | None, *, session_id: str, binding: dict, resume=False):
        self._lease = None
        self.failed = False
        self.db = None
        if resume and path is None:
            raise ValueError("resume requires an execution_state_path")
        try:
            if path is not None:
                path = Path(path).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._lease = path.with_suffix(path.suffix + ".lock").open("a+b")
                fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                exists = path.exists()
                if exists != resume:
                    raise ValueError("Existing execution state requires resume=True; "
                                     "a new episode needs a new execution_state_path")
            self.db = sqlite3.connect(str(path) if path else ":memory:",
                                      check_same_thread=False)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS records "
                            "(kind TEXT, key TEXT, payload TEXT, PRIMARY KEY(kind,key))")
            previous = self.get("meta", "binding")
            expected = dict(session_id=session_id, **binding)
            if resume:
                if previous != expected:
                    raise ValueError("Execution state belongs to a different backend session, "
                                     "task, or budget configuration")
            else:
                self.put("meta", "binding", expected)
        except BaseException:
            self.close()
            raise

    def get(self, kind, key):
        row = self.db.execute("SELECT payload FROM records WHERE kind=? AND key=?",
                              (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind):
        return {key: json.loads(payload) for key, payload in self.db.execute(
            "SELECT key,payload FROM records WHERE kind=? ORDER BY key", (kind,))}

    def put(self, kind, key, value):
        self.check()
        try:
            payload = json.dumps(plain(value), ensure_ascii=False, allow_nan=False)
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                                (kind, key, payload))
        except BaseException:
            self.failed = True
            raise

    def check(self):
        if self.failed or self.db is None:
            raise RuntimeError("Execution journal unavailable; no new actions may be dispatched")

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self._lease is not None:
            self._lease.close()
            self._lease = None
