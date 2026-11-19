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

Keep a short current-goal note with the predefined Python helper:
`plan(goal="...", next_action="...", done_when="...", evidence="", status="in_progress")`.
Call it alongside useful work when the goal, evidence, or approach changes (no planning-only turn needed). Track primary execution budget (~60–70%) vs contingency/cleanup reserve (~30–40% for alignment fixes, release verification, and homing). Each update replaces the note; `plan()` reads it across turns. Status is `pending`/`in_progress`/`done`/`blocked` (`done` covers your stated goal only). Use `action_history(limit=10, stateful_only=True, since_plan=False)` to inspect recent actions and outcomes. Plan text is a planner note, never benchmark fact or motion permission. Before finishing, review the task instruction and acceptance criteria, account for all task-relevant objects visible at step 0, and cite fresh observation evidence. Keep fields brief (<= 1000 chars); notes survive code errors.

## Executable action contracts

`plan(..., contract={...})` enforces zero-argument read-only callbacks returning `bool` or `monitor.ConstraintCheck` (`requires` before actions, `preserves` before/after, `ensures` at `status="done"`/finish; unknown/errors block; static checks expire on motion). Bind all public invariant IDs in `preserves`. New contracts replace `requires`/`ensures`/`constraints` and merge `preserves`. `constraints` inherits `execute_motion` bounds.

## Local work groups

When nearby objects share a setup, reuse one approach, compact local actions with feedback, and one departure. Prefer in-place actions (tip, flip, slide, press) over pick-carry-place when relocation is unnecessary, and do not vacate existing group members to open an interior gap. After a low contact stroke, lift straight up or reverse along the contact axis before any lateral sweep, and verify neighbours remain undisturbed. Record member states and active intent IDs in `plan()`; inspect effects after contact, reconcile interrupted actions before resuming, and do not abandon useful remaining work solely from cost estimates.
"""
