---
name: astra-valen-supervision
description: Plan and execute robot tasks with Astra and Valen visual supervision. Use when decomposing manipulation tasks, writing phase-specific watch contracts, diagnosing uncertain visual judgments, or reconciling completion evidence in PyRUA-Lean.
---

# Astra–Valen supervision

Turn the user's goal into observable, dependency-aware milestones. Astra plans
and interprets evidence; Valen answers visual questions; the host applies pause
rules; the environment supplies its own verdict when available. Keep these roles
and their evidence separate.

This skill supplies planning instructions, not a runtime enforcement mechanism.
Read [contracts and examples](references/contracts.md) when writing a task brief
or Python contracts. Read the repository's [supervision API](../../docs/visual-supervision.md)
for current limits and execution semantics. Repository links assume this skill
remains in `skills/astra-valen-supervision/`.
When you confirm that the current plan is stuck, call `finish(status="stuck")`
with the blockage and attempts made. If confirmed-stuck recovery is enabled, the
host dispatches [Astra stuck recovery](../astra-stuck-recovery/SKILL.md) afterward.
Read `episode_finished`: false means apply the returned skill and continue in the
same episode; true means stop. Do not pre-activate recovery on every Valen pause.

## 1. Establish the acceptance contract

Before moving, extract from the task instruction, permitted documentation, and
available observations:

- Required objects, relationships, final states, and any ordering constraints.
- Constraints to preserve while manipulating other objects.
- Observable evidence for each requirement, including which view can show it.
- Quantitative requirements needing permitted measurements rather than RGB guesses.
- Required cleanup: release, gripper state, arm pose, or other task-specific state.
- Available acceptance authority, action/step/time limits, and unresolved ambiguity.

Do not substitute a plausible-looking arrangement for an unspecified target.
Resolve consequential ambiguity from permitted context; ask only if it blocks a
meaningful choice. Continue independent observation while clarification is pending.
Do not impose home/open, symmetry, or a particular construction order unless the
task or execution constraints require it.

If benchmark rules are explicitly supplied or their use is authorized, translate
them into target constraints and record their source. Do not silently consult a
hidden evaluator or hidden object poses in a restricted benchmark run. Disclose
extra task knowledge in experiment records; a clarified run is a different prompt
condition. Never feed the benchmark verdict to Valen as visual evidence.

## 2. Plan phases around dependencies and evidence

Describe each milestone with: prerequisites, intended change, preserved facts,
exit evidence, expected visibility, and recovery scope. Order milestones by their
dependencies; use a simple sequence when sufficient. Separate approach, grasp,
transport, release, and inspection when the facts expected to hold change.

Maintain a compact evidence ledger in the plan or persistent cell variables:
`fact ID -> question, source, status, observation/step, dependencies, invalidators`.
These are planner records, not additional host API fields. A movement that can
change a fact invalidates its old evidence. Recheck affected dependents after a
repair; do not infer current truth from an earlier yes.

Carry still-required, established facts into later phase contracts. Enter a
repair phase explicitly when changing a previously established relationship;
retain unaffected constraints and re-arm the repaired relationship at the
appropriate endpoint. Do not require a future postcondition throughout approach
or transport. Arm a release postcondition before the release action, then inspect
the returned evidence before starting dependent work.

Use short cells ending at evidence boundaries. In this host, `clear` means no
pause rule fired, NOT that all facts are yes. A single critical unknown can yield
`clear`. Astra must gate milestone advancement on the required evidence itself.

## 3. Ask Valen about visible facts

Use one concrete relationship or state per question, identified by visible object
attributes and a clear reference object or arm. Split contact, orientation,
centering, and identity when they can fail independently. Examples:

- "Is the red cup inside the blue tray?"
- "Is the red cup upright?"
- "Is the drawer front flush with the cabinet?"

Avoid historical references such as "the other two" unless identities are clear
in the supplied images. Avoid "correct", "safe", "done", broad scene quality,
future intentions, and asking RGB images to certify precise millimeters, force,
or hidden contact. If a required fact cannot be observed, mark the evidence gap
and choose a supported observation or measurement; do not invent confidence.

Choose a small set of relevant checks for each phase. Explicitly define:

- `facts`: questions Valen answers.
- `alerts`: which confident contradictions require review, with the correct
  polarity. For a required invariant, typically alert on `no`.
- `critical`: facts whose sustained uncertainty blocks dependent progress.
- `completion`, if useful: a candidate for Astra review, never automatic success.

`critical` does not alert on a confident no; `alerts` does not catch unknown.
Use both where both matter. Match rules to expected phase behavior: "object held"
is appropriate for verified transport, not after intentional release.

Inspect actual evidence views. A home-position wrist camera may show no target;
multiple views can add occlusion and irrelevant context. The current host sends
all declared cameras and at most one previous endpoint: `watch()` has no camera
selection argument. Camera filtering or extra Valen-only replay requires an
explicit harness/offline change, not an invented helper option. For development,
compare camera subsets on saved evidence before changing live supervision.

## 4. Execute and recover without erasing uncertainty

Declare `watch()` before motion in every cell. Read the returned action result,
step IDs, supervision report, and relevant images before continuing after a pause.
The action that triggered a review already executed; an errored action may have
partially executed. Replan from the actual state instead of replaying a whole cell.
Do not catch or bypass supervision interruptions.

Diagnose uncertainty in this order:

1. **Evidence/transport failure:** missing images, stale endpoint, malformed
   response, timeout, or `monitor_error`. Check artifacts and state before retrying.
2. **Explicit model uncertainty:** raw highest-probability choice is `unknown`.
3. **Host abstention:** raw highest choice is yes/no, but its probability is below
   the configured threshold. Inspect all probabilities; a weak no is not agreement.
4. **Perceptual/question ambiguity:** inspect visibility and object identity;
   split compound questions or replace ambiguous references without weakening the
   acceptance condition. Compare equivalent variants on the same saved frame.

The current decoder uses the maximum candidate probability, not the response's
separate `confidence` field. Preserve the raw result and decoded result when
reporting uncertainty. Probabilities are not calibrated accuracy estimates.

For each retry, state what evidence will improve: e.g. a meaningful retreat that
exposes contact, a supported view change, or a clearer identity description. Use
the run's recovery budget. If none is specified, two unproductive local attempts
prompt a progress review. If the current plan is stuck, explicitly declare it
with `finish(status="stuck")`; the enabled host then dispatches recovery rather
than finalizing. Apply that returned procedure to reclassify dependencies and
consider alternative strategies. A local retry count alone does not prove the
whole task impossible. An unresolved current prerequisite still blocks actions
that depend on it.

Do not lower thresholds, empty `critical`, redeclare phases, or start new cells
merely to avoid pauses. A documented dependency review may narrow `critical` and
remove alerts on future goals for an independent phase: keep deferred facts
unresolved in the ledger, name the action boundary that needs them, and restore
and establish them before crossing it. This reschedules checks without weakening
task acceptance. New cells and every `watch()` call reset evidence streaks;
carry unresolved facts in the planner ledger across these resets. Read-only
`show()` does not invoke Valen, and an unchanged/no-op action is not a new vote.
Do not spend robot actions just to manufacture confirmations.

If tuning supervision is part of the task, keep it separate from execution:
freeze evidence, label failures and successes, vary one factor at a time, and
compare misses, false alarms, and abstentions. A higher yes rate alone is not an
improvement. Record threshold, camera, and question changes as new configurations.

## 5. Close with distinct verdicts

Declare confirmed stuck before invoking the recovery procedure. With the opt-in
host dispatcher, `finish(stuck)` can return `episode_finished=false` and the skill
instructions; continue from that response. Without the dispatcher, or when it
returns `episode_finished=true`, finish is terminal. A local pause alone is not
a confirmed stuck declaration.

When the scene appears complete, inspect every required final relationship and
required cleanup state using sufficiently recent, uninvalidated evidence. Observe
stability over distinct relevant endpoints only when the task requires it; do not
invent a universal number of votes. Stop issuing actions once the environment
has ended. A primitive ending early can reflect task success as well as limits or
failure: inspect the actual termination reason if exposed.

Report separately:

| Field | Meaning |
|---|---|
| Native/task verdict | Authoritative evaluator result, if available and permitted; otherwise pending/unavailable |
| Visual verification | Confirmed, contradicted, or unresolved; name remaining facts |
| Astra assessment | Planner judgment with its evidence and limitations |
| Execution status | Running, ended, error, or budget exhausted; not task success by itself |

In this repository, `result.json.status`, `environment_success`, and
`finish.status` answer different questions. `finish` records Astra's self-report;
it does not override the native verdict. Native true plus visual unknown means
"environment passed; visual verification incomplete", not environment failure.
Native false plus a convincing image is still an evaluator failure. If no native
verdict is exposed before `finish`, report the available visual evidence honestly
and reconcile after finalization; do not assume the planner can read that file live.

Do not rewrite original result fields to make them agree. Keep contradictory
reports and explain their sources. For user tasks without an evaluator, report
which agreed acceptance conditions are verified and which remain unresolved.

## Deliverables when preparing a run

Produce a task-specific brief and phase contracts using the linked template,
with evidence/repair budgets and completion authority. Record the actual prompt,
model/monitor configuration, question versions, and evidence artifacts. Include
the supervision instructions and trigger protocol in the initial prompt. Configure
`--stuck-recovery-skill` so the recovery procedure is supplied only after confirmed
stuck. Verify the saved prompt and dispatch audit. Merely storing skill files in
the repository does not enable this behavior.
