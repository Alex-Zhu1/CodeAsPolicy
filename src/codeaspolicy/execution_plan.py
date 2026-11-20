"""A small planner-authored progress note, separate from task facts and controls."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path


class ExecutionPlan:
    def __init__(self, directory: Path, session_id: str, task_card_path: Path | None = None):
        self.directory = directory
        self.session_id = session_id
        self.revision = 0
        self.latest: dict | None = None
        self.card_reference = None
        if task_card_path is not None:
            self.card_reference = dict(
                path=str(task_card_path.resolve()),
                sha256=hashlib.sha256(task_card_path.read_bytes()).hexdigest(),
            )

    def read(self):
        return copy.deepcopy(self.latest)

    @staticmethod
    def validate(*, goal, next_action, done_when, evidence, status):
        values = dict(goal=goal, next_action=next_action, done_when=done_when, evidence=evidence)
        for name, value in values.items():
            if not isinstance(value, str) or len(value) > 1000:
                raise ValueError(f"plan {name} must be text of at most 1000 characters")
        if not goal.strip() or not done_when.strip():
            raise ValueError("plan needs a goal and done_when completion criterion")
        if status not in ("pending", "in_progress", "done", "blocked"):
            raise ValueError("plan status must be pending/in_progress/done/blocked")

    def update(self, *, goal, next_action, done_when, evidence, status, context, contract=None):
        self.validate(goal=goal, next_action=next_action, done_when=done_when,
                      evidence=evidence, status=status)
        values = dict(goal=goal, next_action=next_action, done_when=done_when, evidence=evidence)
        record = dict(
            schema="codeaspolicy.execution_plan.v1", author="astra", kind="planner_note",
            session_id=self.session_id, revision=self.revision + 1,
            updated_at=time.time(), time_domain="unix_seconds",
            task_card=self.card_reference, context=context, status=status, **values,
        )
        if contract is not None:
            record["contract"] = copy.deepcopy(contract)
        data = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        # Keep revisions outside the policy workspace, just like the execution audit.
        latest = self.directory / "execution_plan.json"
        temporary = self.directory / "execution_plan.tmp"
        temporary.write_text(data, encoding="utf-8")
        with (self.directory / "execution_plan.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(data)
        temporary.replace(latest)
        self.latest, self.revision = record, record["revision"]
        return self.read()

    def summary(self):
        if self.latest is None:
            return None
        result = {key: copy.deepcopy(self.latest[key]) for key in (
            "kind", "revision", "goal", "next_action", "done_when", "evidence", "status", "context",
        )}
        if "contract" in self.latest:
            result["contract"] = copy.deepcopy(self.latest["contract"])
        return result


PLAN_PROMPT = """\
# Execution plan

Maintain a brief goal note alongside work (no planning-only turns):
`plan(goal="...", next_action="...", done_when="...", evidence="", status="in_progress")` (`plan()` reads it; fields <= 1000 chars; status: `pending`/`in_progress`/`done`/`blocked`). Budget ~60–70% for primary execution and ~30–40% reserve for fixes, release verification, and homing. Before finishing, account for all step-0 task objects against acceptance criteria with fresh visual evidence.

## Executable action contracts

Optional `plan(..., contract={...})` binds zero-argument read-only callbacks returning `bool` or `monitor.ConstraintCheck` (`requires` before actions, `preserves` before/after, `ensures` at `done`/`finish`; unknown/errors block; static checks expire on motion). When public invariant IDs are supplied on the card, bind all of them in `preserves`.

## Local work groups

Batch nearby items sharing one setup into one approach, compact in-place actions (tip, flip, slide, press) with local checks, and one clean vertical/axial retreat—do not relocate items unnecessarily or vacate existing members to open an interior gap. If a neighbour shifts while budget remains, re-stand or re-align it rather than stopping early.
"""
