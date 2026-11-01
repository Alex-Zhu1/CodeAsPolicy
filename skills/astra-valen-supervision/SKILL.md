---
name: astra-valen-supervision
description: Write robot-control Python with Astra and Valen visual evidence in PyRUA-Lean. Use when choosing manipulation checks, composing single-arm or dual-arm actions, handling uncertain observations, or reconciling completion evidence.
---

# Astra–Valen supervision

Astra receives a robot manual, callable `robo` methods and current observations,
then writes the task strategy in Python. Choose roles, targets, ordering, release
timing, checks and bounded corrections from the task and scene. This skill provides
reasoning guidance and reusable patterns; it does not prescribe a task state machine.
Valen answers local visual questions and can trigger review. Astra combines that
fallible evidence with images, measurements and action results to choose the next
program. The host executes actions and enforces explicit dependencies and resource
limits. Ordinary grasp and transport do not require a Valen approval by default.

The packaged [runtime core](../../src/pyrualean/knowledge/astra_valen_supervision.md)
is injected when monitoring is enabled and frozen with its hash for the run.
Changing this repository skill alone does not update that core. Read
[contracts and examples](references/contracts.md) for `watch`/evidence semantics,
[dual-arm code](../../docs/guide/coordination.md) when composing coordinated moves,
and the [supervision reference](../../docs/visual-supervision.md) for host limits.
Repository links assume this skill remains in its repository location.

## Plan the program from the actual task

Extract acceptance conditions and constraints from the task instruction and
permitted documentation. Identify what is observable, what requires a quantitative
measurement, and who provides the final verdict. Do not invent cleanup, symmetry,
ordering or stability-vote requirements. Keep hidden evaluator truth and object
poses out of policy inputs and Valen evidence in restricted benchmark runs.

A phase is a useful planner grouping when dependencies or expected facts change,
not a mandatory sequence of approach/grasp/transport/release. A short program may
contain several actions and conditional branches. Select model-return boundaries
where new judgment is needed; avoid requiring another model turn for every small
step or known numerical correction.

For two arms, decide which acts, supports or observes, and when those roles change.
Use `move_to`/`set_gripper` for sequential actions and `move_both` for genuine
simultaneous motion. They compose directly. Simultaneity alone does not maintain
relative pose; request `keep_relative` only when needed. Read actual capabilities:
this RoboDojo executor does not certify full arm-arm or object collision clearance.
Targets and corrections must come from current observations, not a copied task recipe.

## Select the evidence that this action needs

Use controller state for arrival and numerical error. A gripper's `holding` or
opening is not proof of intended-object support. Use current images, permitted
measurements and suitable probes to establish the relevant relationship. The
optional `strategy.verify_grasp` helper is useful when its callbacks can supply
that evidence; other justified verification code is valid. During a supported
handover, verify receiver support before giving up donor support, then inspect
what happened after release. A release onto a surface may need different checks.

Ask Valen atomic questions about visible relationships with clear object/arm
identities. Do not ask it to certify hidden contact, force, millimetres or broad
"safety/success". `unknown` is unresolved evidence; probabilities are not calibrated
accuracy. Preserve raw winners and host threshold abstention reasons.

When monitoring is enabled, call `watch` before motion in each cell:

- `facts` names visual questions; `alerts` handles explicit contradictions.
- `critical` defaults to `[]`; opt in to a review for sustained unknown on selected facts.
- `requires` explicitly demands fresh matching Valen evidence before every motion in that named phase. Use it only for a necessary hard visual dependency, not as the default transport pattern.
- `completion` requests planner review of candidate completion, not an automatic task verdict.

`clear` only means no pause rule fired. Inspect `evidence()` for freshness,
question versions and observation IDs before dependent actions. Motion invalidates
old evidence by default; narrow `invalidates` only with a causal reason. Registered
gates persist across cells/watch calls, and editing an evidence snapshot changes
no host fact. Defer a later prerequisite with `defer_fact(..., until=..., reason=...)`
and establish it before that boundary. Do not change phase names to bypass it.

Use explicit verification for an explicit missing prerequisite or a
support/release decision that needs evidence. A normal approach without these
dependencies needs no extra pre-action visual call; automatic endpoint monitoring
already updates facts. Reuse current results rather than rechecking every primitive.
On a backend with source-aware snapshots, call `view = robo.snapshot()` and then
`checked = verify(view["observation_id"])` when needed. Request
`snapshot(refresh=True)` only when a new read-only capture is necessary.
`verify()` evaluates the recorded snapshot without refreshing it or moving. Inspect `values`,
`blocked` and the observation reference. An unsupported backend, stale ID or
misaligned image/state source cannot establish current evidence through this API.
Use the snapshot's recorded `step` when reading its images or camera calibration.

Unknown streaks and explicit phase gates survive cells and phase changes.
A continuous critical unknown episode requests review only once. In the next cell,
Astra inspects the report and scene, records the reasoning for its next action,
then continues, corrects or seeks useful evidence. This handoff does not change
Valen's unknown to yes and does not require Valen to approve the new program.
A new known source ends the unknown episode; a later unknown episode may trigger
again. Alert/completion conditions also notify once while continuously matching;
new explicit counterevidence rearms them, and unknown alone does not. Re-reading
one source cannot supply another confirmation, change its accepted vote or clear
uncertainty. Explicit `requires`/`defer_fact` gates remain enforced across reviews.
A deliberately changed observation may help, but avoid spending actions solely to
obtain another vote. Explicit verification has
its own per-cell allowance (`verification_calls` / `verification_limit`), counting
failures separately from the motion count.

## Keep both feedback loops

Inspect `reached`, error, actual endpoint and consumed native steps. A recoverable
`DualMove` rejection/partial result may select a bounded branch in the generated
program. A zero-step IK refusal can likewise return without producing a visual
vote. A recoverable partial dual-arm move is visually checked at its actual
endpoint before returning; Valen can still pause that correction loop. Other
unclassified partial failures require outer review. Returning normally does not
establish grasp or task success.

Choose a retry bound, a progress measure and a concrete correction hypothesis.
Preserve planner phase allocations across cells; `move_both(max_steps=...)` bounds
one call, while episode limits remain cumulative. Prefer a changed hypothesis
when observations or attempts stop adding useful information. Optional strategy
helpers factor out loops; they do not choose task geometry or enforce a fixed recipe.

Code errors, critical execution faults, expired budgets and visual reviews return
the live scene to Astra. Do not catch control interruptions. Determine which calls
executed and which assignments never completed before generating follow-up code;
never replay a failed cell as if it were rolled back. Distinguish controller
outcomes from `monitor_error` (image/protocol/service failure). Do not lower a
threshold or manufacture no-op votes to get past uncertainty. Review the actual
evidence need: ordinary Valen abstention is not an unknown execution outcome.

Persistent capture-capability, source or service failures belong to the host.
After a relevant bounded recovery such as one explicit refresh, report an unchanged
blocker with `finish(status="failure")`; do not spend manipulation retries, repeated
localization or robot motions trying to fix the protocol.

Inspect `robo.action_receipt()` and `handoff.unresolved_actions` after an incomplete
or uncertain execution. A known stopped partial result can support a bounded
correction; an unknown/still-running action first needs `robo.reconcile(action_id)`.
That query never replays the action. Only a matching receipt establishing execution
completion and effect certainty clears the execution blocker. Images, visual yes,
an edited receipt copy or a new cell cannot clear it. After reconciliation, capture
the current scene, verify the remaining prerequisites, and write the remaining work
in a new cell. Receipt reconciliation and visual verification discharge different
obligations; neither means the task is complete.

Only after confirming stuck call `finish(status="stuck")` with attempts and blockers.
If the enabled dispatcher returns `episode_finished=false`, apply the returned
[recovery skill](../astra-stuck-recovery/SKILL.md) in the same episode. A local
rejection, pause or unknown alone does not load recovery. `episode_finished=true`
is terminal. Preserve native/task verdict, visual verification, Astra assessment
and execution status separately when reporting completion.

For supervision tuning, use saved and labelled evidence with the
[offline evaluator](../../docs/guide/valen-evaluation.md). A lower unknown rate alone
is not an accuracy gain; don't change the live task's acceptance criteria to improve it.
