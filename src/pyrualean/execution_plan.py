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

    def update(self, *, goal, next_action, done_when, evidence, status, context):
        values = dict(goal=goal, next_action=next_action, done_when=done_when, evidence=evidence)
        for name, value in values.items():
            if not isinstance(value, str) or len(value) > 1000:
                raise ValueError(f"plan {name} must be text of at most 1000 characters")
        if not goal.strip() or not done_when.strip():
            raise ValueError("plan needs a goal and done_when completion criterion")
        if status not in ("pending", "in_progress", "done", "blocked"):
            raise ValueError("plan status must be pending/in_progress/done/blocked")
        record = dict(
            schema="pyrualean.execution_plan.v1", author="astra", kind="planner_note",
            session_id=self.session_id, revision=self.revision + 1,
            updated_at=time.time(), time_domain="unix_seconds",
            task_card=self.card_reference, context=context, status=status, **values,
        )
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
        return {key: copy.deepcopy(self.latest[key]) for key in (
            "kind", "revision", "goal", "next_action", "done_when", "evidence", "status", "context",
        )}


PLAN_PROMPT = """\
# Execution plan

Keep a short current-goal note with the predefined Python helper:
`plan(goal="...", next_action="...", done_when="...", evidence="", status="in_progress")`.
Use it in the same code call as useful work, and update when the goal, evidence
or approach changes; no separate planning-only turn is required. Each update
replaces the current note. `plan()` reads it across cells/programs. Status is
pending/in_progress/done/blocked; done refers only to your stated goal.
These are your planning claims, not benchmark facts, verified evidence or motion
permission. They never replace the task instruction, watch prerequisites or finish.
Keep text brief (at most 1000 characters per field). The host saves the latest
note and revision history with observation/action context; earlier notes survive
code failures. An unchanged note is not refreshed by later motion.
"""
