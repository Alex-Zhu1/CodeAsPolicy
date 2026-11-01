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

The `play` CLIs configure this skill's packaged
[runtime core](../../src/pyrualean/knowledge/astra_stuck_recovery.md) by default for
cells. `--stuck-recovery-skill FILE` overrides that core; `--no-stuck-recovery`
disables dispatch. The host records the confirmed blockage and dispatches recovery
without finalizing the episode. Read the tool
result: `episode_finished=true` is terminal (including exhausted recovery rounds
or episode budgets). With recovery disabled, `finish` remains terminal. A skill
cannot reopen an ended episode or refund its budgets. If it has already ended, analyze
the case for a separately authorized new attempt. Do not automatically launch a
new episode, alter the evaluator, or resume from assumed state.

## Generate a revised program

Recovery changes Astra's reasoning and generated Python; it does not run a preset
robot workflow. Reconsider roles, target geometry, action ordering and evidence
boundaries. Compose single-arm/gripper calls with move_both when synchronized
motion is useful. A known recoverable result may be handled by bounded code before
stuck is confirmed; critical faults and visual pauses still return to the model.
Choose the next sequence from the actual scene, without an operation object or
mandatory grasp/transfer/release recipe. Keep planner retry allocations across
cells; per-call move_both limits do not reset native episode usage.

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
Ordinary visual review hands judgment back to Astra. Combine images, measurements
and action results; a Valen unknown alone does not prove a missed grasp or unknown
execution. In the next cell, proceed from that assessment without rewriting the
Valen record. Only an explicitly registered `requires`/`defer_fact` gate continues
to demand a matching fresh visual answer.

## 2. Reclassify conditions relative to the next candidate action

For each condition ask: **If this were false, would this particular action fail,
damage established progress, or violate a task constraint?** Use scene geometry
and task dependencies to justify the answer; do not classify by confidence alone.

| Classification | Treatment |
|---|---|
| Needed for this action, or an invariant it could disturb | Assess it using the available evidence; retain useful review triggers and resolve or repair any real dependency before progress |
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

Reserve native steps/time for the unfinished task before allocating a repair
phase, and preserve that allocation across cells. Read `motion_rejected` or
`motion_incomplete` as controller outcomes, not Valen faults. For grasp repair,
the [bounded verification helper](../../docs/guide/strategies.md#verify-a-grasp)
can close, probe support and compare fresh observations. Its uncertain result
requires replanning; its repair callback only retries definite failures within
the chosen limit. Mechanical `holding=True` alone does not establish a stable
grasp. Receiver support must be established before donor release in a handover.

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

Inspect the host ledger with `evidence()`. Register a known deferred fact using
`defer_fact(fact_id, until="dependent phase", reason="...", expected="yes")`.
The exact phase name becomes a persistent prerequisite; changing cells, omitting
the fact from a watch, or passing `requires={}` does not remove it. This call does
not change truth or remove an earlier gate. Use stable phase names and retain the
causal reason in the plan. Facts may remain in `facts` as noncritical
observations if useful, or be omitted from this phase's questions if irrelevant.
Review `alerts` as well as `critical`: an alert on the deferred future goal can
still interrupt this phase. Retain alerts for true current invariants.

List the next phase's actual prerequisites and guards. `critical` defaults to `[]`;
choose it only when sustained Valen uncertainty merits a review. It is not a list
of all physical dependencies or a substitute for Astra's assessment. A continuous
unknown episode triggers once, while new known evidence permits a later recurrence
to trigger again. Use `requires` only for a deliberately hard visual condition;
ordinary transport must not acquire a `held=yes` gate just because it uses a gripper.

Write the changed plan and its deferral reasons before the next motion. After a
pause, do this in the next cell; do not catch the interruption. Execute a short
phase, then inspect the actual result and update the ledger. Check that it made
physical progress or supplied the promised information.

Before entering a deferred fact's dependency boundary, restore its appropriate
contract and establish it. For example, alignment may be unknown during transport;
arm it at the final alignment endpoint and inspect the evidence **before insertion**.
Do not wait until after the dependent action to restore a prerequisite. A repair
may intentionally change an invariant, but must restore it before dependent work.

The host carries evidence and registered phase prerequisites across cells. In
pause mode it blocks motion in the dependent phase until the fact has fresh
matching evidence. Establish it in the preceding observation/repair phase, not
after the dependent action. It cannot infer undeclared dependencies or detect
misnamed phases. Astra remains responsible for causal classification, justified
`invalidates` declarations, and final acceptance. A new `watch()` does not resolve
an earlier uncertainty or discard a registered gate. Normal review triggers and
explicit gates have different obligations: do not turn every review into a new gate.

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

For a CLI cells run, the bundled core is already configured. Pass this longer skill
file via `--stuck-recovery-skill` to override it, or use `--no-stuck-recovery` to
disable dispatch. Choose `--stuck-recovery-rounds` (default two dispatches).
Monitoring automatically includes
the supervision core; verify it and the confirmed-stuck protocol in the initial
prompt. Deliver this recovery
procedure only after the declaration. The host records `stuck_recovery.jsonl` and
keeps the scene, Python namespace, and existing budgets. No actions are replayed.
Remove contradictory task-brief rules that force finalization after a local
retry count. Do not silently change archived prompts/results from prior attempts.
