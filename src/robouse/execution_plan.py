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
            schema="pyrualean.execution_plan.v1", author="astra", kind="planner_note",
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
Use it in the same code call as useful work, and update when the goal, evidence
or approach changes; no separate planning-only turn is required. Each update
replaces the current note. `plan()` reads it across cells/programs. Status is
pending/in_progress/done/blocked; done refers only to your stated goal.
The text fields are planning claims, not benchmark facts or motion permission.
They never replace the task instruction, watch prerequisites or finish.
Before declaring the whole task complete, review the task card's acceptance
requirements and original instruction, including ordering, preservation and final
state. When acceptance coverage is unknown or partial, account for all task-relevant
object groups visible in step 0 (such as primary targets, actor/opponent items,
separate supply/replacement stacks and destination slots) and every physical phase
implied by the domain action. Record unmet/unknown items and assumptions in your
note; a narrower done_when or recorded assumption never discharges an unperformed
phase or justifies finishing while task-relevant scene objects remain unused and
budget remains. Cite observation/action evidence, and distinguish your completion
claim from the environment's verdict. Missing public criteria alone do not require
stopping useful work or an extra planning-only turn.
Keep text brief (at most 1000 characters per field). The host saves the latest
note and revision history with observation/action context; earlier notes survive
code failures. An unchanged note is not refreshed by later motion.

## Executable action contracts

plan(..., contract={...}) works without Valen. requires maps names to checks before
actions; preserves checks before/after; ensures checks at status="done"/success finish.
Bind all public invariant IDs in preserves. Checks: zero-argument read-only callbacks
returning bool or monitor.ConstraintCheck from current evidence; unknown/errors block.
Static checks expire on attempted motion; rebind after restart.
New contracts replace requires/ensures/constraints, merge preserves; omission retains.
constraints inherits execute_motion bounds into motion/local/grasp; conflicts or old IDs
missing bounds reject. Observations/failure/stuck remain available; repair assumptions
or choose independent work. Preserve protections. Compare approach/contact/exit for
preservation, evidence, both arms' clearance, held extent, future reachability and cost.
Preview coupled poses; re-observe after interaction. Arrival does not prove task completion.

## Plan local work groups

When nearby objects or controls share a useful setup, consider a local work
group. Choose its members, required effect per member, dependencies, surroundings
to preserve, and an exit condition from current observations. Compare the cost
of candidate manipulation schemes by the required final object state: distinguish
objects that only need an in-place state or orientation change (such as tipping
face up, flipping, sliding or pressing) from objects that must be relocated to a
different zone. When an object or group can achieve its required final state in
place, act on it in place with local contact strokes rather than picking up,
carrying, placing and then tipping/acting on each member. Do not vacate or
relocate existing group members merely to open an interior gap for a later
supply/replacement item; place an incoming or replacement item at the designated
open end or target slot in its required standing/resting orientation. Compare the
cost of a shared approach and local transitions with repeatedly leaving and
returning. Prefer one approach, a compact sequence of local actions with feedback,
and one departure when geometry and task ordering permit. Keep useful positioning
between members; reposition only as far as the next action and clearance require,
and keep default free/approach arrival tolerances on intermediate shifts/retreats
so local repositioning is not slowed by unnecessary precision windows. After a
low contact stroke, always lift clear vertically or reverse back along the
contact axis before sweeping laterally across neighboring objects, and check/repair
any disturbed neighbors before leaving the group.

A group may use several strokes, grasps or directions. Simultaneous contact and
a single straight trajectory are optional, not the definition of grouping.
Choose contacts that can produce the intended effect on each affected member;
proximity alone does not establish common contact geometry. Split the group or
change approach when access, ordering, support, occlusion or interference demand
it. Parallel work is useful only for independent actions with compatible routes.

Record concise member identities, pending/observed-complete/uncertain states,
evidence references, and the active action/operation ID in the existing plan
fields. A completed motion is not a completed member. Inspect effects after
contact or a dependency-changing action, then continue only the unfinished work
whose prerequisites are known. Related actions may share a code call when their
feedback suffices; return for visual judgment when it is needed. Grouping never
licenses a blind loop over stale targets or continuing after a failed prerequisite.

After interruption, reconcile the active action/operation before retrying it;
where supported, continue the same ID's known remaining work. Reobserve uncertain
or displaced members. Preserve completed members only while their evidence is
still valid; do not replay the whole group or treat the plan note as execution
permission. Leave the local area when the group is verified, the next task needs
a new setup, or a concrete obstruction requires replanning. Budget estimates
alone do not justify abandoning useful remaining work.
"""
