---
name: astra-stuck-recovery
description: Replan robot tasks after Astra explicitly confirms stuck and the host dispatches recovery for a live episode. Reassess causal prerequisites, defer checks to the phases that need them, and choose bounded progress or information-gathering actions before ending an episode.
---

# Astra stuck recovery

Treat a stalled phase as a planning problem before treating it as episode failure.
Reconstruct the actual state, challenge the current decomposition, and choose a
next action that either advances the task or resolves a consequential uncertainty.

Use this skill **after** Astra calls `finish(status="stuck")` and the enabled host
returns `status="stuck_confirmed", episode_finished=false`. A pause or repeated
unknown alone does not activate this skill. It complements
[Astra–Valen supervision](../astra-valen-supervision/SKILL.md). See
[recovery cases and contracts](references/recovery-cases.md), when accessible, for examples of
reclassifying conditions and restoring them later.

With `--stuck-recovery-skill` enabled, the host records the confirmed blockage
and dispatches these instructions without finalizing the episode. Read the tool
result: `episode_finished=true` is terminal (including exhausted recovery rounds
or episode budgets). Without this option, `finish` remains terminal. A skill
cannot reopen an ended episode or refund its budgets. If it has already ended, analyze
the case for a separately authorized new attempt. Do not automatically launch a
new episode, alter the evaluator, or resume from assumed state.

## 1. Reconstruct the blockage

Make a short recovery record from the latest images, primitive results, monitor
reports, and the persistent evidence ledger:

- What physically happened, what was attempted, and what never executed.
- Current milestone, remaining steps/calls/time, and any termination signal.
- Each unresolved or contradicted fact, raw model probabilities, and decoded
  result. Distinguish an empty grasp, a poor view, a host error, and abstention.
- The causal reason each fact was made critical. Mark assumptions as assumptions.
- Approaches already tried and the evidence or progress each produced.

A monitor pause is not a rollback. A successful Python call is not a successful
grasp. Stop issuing actions if the environment is terminal; otherwise reason from
the observed state rather than rerunning the failed cell.

## 2. Reclassify conditions relative to the next candidate action

For each condition ask: **If this were false, would this particular action fail,
damage established progress, or violate a task constraint?** Use scene geometry
and task dependencies to justify the answer; do not classify by confidence alone.

| Classification | Treatment |
|---|---|
| Needed for this action, or an invariant it could disturb | Retain the appropriate critical check and contradiction alert; resolve or repair before dependent progress |
| Required only by a later milestone | Defer the blocking check, record a reactivation boundary, and continue independent work |
| Required final outcome | Keep the requirement; intermediate approach need not satisfy it. Verify before completion |
| Planner-invented assumption with no task or causal basis | Remove it as a gate and record why it was unnecessary |
| Essential but not observable in current evidence | Seek another permitted observation/measurement or an action avoiding that dependency; uncertainty alone does not justify dropping it |

The classification can change with the proposed action. An uncertain lock pose
may be irrelevant while grasping a distant key, but decisive for insertion or a
motion passing close to the lock. Known false and unknown are different: a failed
current prerequisite requires recovery, while a not-yet-achieved future goal can
remain false during earlier work.

Changing `critical` is permitted when this analysis changes the phase contract.
Do not lower the confidence threshold or relabel unknown as yes to obtain a pass.
Preserve task requirements, actual execution constraints, and still-relevant
invariants. These changes need no extra permission within an already authorized
robot task, but do not expand that task or its execution permissions.

## 3. Re-infer a way forward

Generate a few materially different candidates, rather than arbitrary variations
of the last command. Useful alternatives include:

- Complete an independent prerequisite first, or reorder independent milestones.
- Change grasp location, approach direction, or manipulating hand after a miss.
- Move to a reversible intermediate pose that exposes the object or frees a hand.
- Obtain a useful view or measurement before a contact-sensitive action.
- Undo only the affected part of a failed subassembly, preserving the rest.

For each candidate state its prerequisites, expected progress or information,
possible effect on preserved facts, approximate budget cost, and observation that
would decide the next branch. Prefer feasible, reversible actions with a concrete
benefit and modest cost. Do not spend motion steps merely to solicit new votes.

"Closer to the goal" means an observable improvement such as a verified grasp,
an exposed opening, or reduced alignment error—not just another successful API
call. Partial progress does not relax the final acceptance criteria.

## 4. Revise the contract and execute one recovery phase

For every deferred fact record:

```text
fact/question version:
last evidence and unresolved/false status:
why the next action does not depend on it:
which actions remain prohibited until it is established:
reactivate before which milestone/action:
what new evidence can establish it:
```

Keep deferred facts in the ledger. They may remain in `facts` as noncritical
observations if useful, or be omitted from this phase's questions if irrelevant.
Review `alerts` as well as `critical`: an alert on the deferred future goal can
still interrupt this phase. Retain alerts for true current invariants.

List the next phase's active prerequisites and guards explicitly. An empty
`critical` list is only justified if this action truly has no uncertainty-sensitive
visual prerequisites; document the available controls/evidence and use relevant
questions. Never empty it as a blanket response to a pause.

Write the changed plan and its deferral reasons before the next motion. After a
pause, do this in the next cell; do not catch the interruption. Execute a short
phase, then inspect the actual result and update the ledger. Check that it made
physical progress or supplied the promised information.

Before entering a deferred fact's dependency boundary, restore its appropriate
contract and establish it. For example, alignment may be unknown during transport;
arm it at the final alignment endpoint and inspect the evidence **before insertion**.
Do not wait until after the dependent action to restore a prerequisite. A repair
may intentionally change an invariant, but must restore it before dependent work.

The current host does not enforce deferred-fact deadlines or carry uncertainty
across cells. Astra must do so in its plan and ledger; a new `watch()` resets
streaks but does not resolve an earlier uncertainty.

## 5. Decide whether to continue or terminate

Respect existing episode and task budgets. Repeated attempts with no useful
state/evidence change trigger a new causal review, not a reset of retry counts.
Two unproductive local attempts are a useful review trigger, **not** a universal
instruction to end the episode. Choose a bounded next attempt based on its
hypothesis, cost, and remaining budget; do not repeat the same failed approach
without a changed reason to expect progress.

Continue when at least one feasible candidate can produce meaningful progress or
information without requiring an unresolved current prerequisite. If a fact only
blocks later work, explain that it is deferred and proceed with independent work.

After this recovery review, call `finish(stuck)` again only when no feasible
supported next action remains:
remaining candidates depend on unresolved essential conditions, repeated
alternative approaches bring no progress, necessary capability/input is missing,
or the available budget cannot support a meaningful next attempt. Record what
was achieved, the remaining blockers, alternatives tried, and deferred obligations.
If success or a terminal budget/error condition is established, report that actual
state instead. Preserve native, visual, planner, and execution verdicts separately.

## Run preparation

For a supervised run, pass this skill file via `--stuck-recovery-skill` and choose
`--stuck-recovery-rounds` (default two dispatches). Include the supervision skill
and the confirmed-stuck protocol in the initial prompt; deliver this recovery
procedure only after the declaration. The host records `stuck_recovery.jsonl` and
keeps the scene, Python namespace, and existing budgets. No actions are replayed.
Remove contradictory task-brief rules that force finalization after a local
retry count. Do not silently change archived prompts/results from prior attempts.
