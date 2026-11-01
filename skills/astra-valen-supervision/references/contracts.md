# Task brief and phase contracts

Use this reference when preparing a supervised task. Adapt the template to the
task; the fields below are planning records, not new `watch()` arguments.

## Code composition

A task brief is planning context, not a required runtime state machine. Astra may
combine several motions, checks and bounded corrections in one Python call.
Single-arm/gripper methods compose with `move_both` on supported hosts; see
[dual-arm examples and feedback](../../../docs/guide/coordination.md). The planner
chooses release timing and dependencies. Astra normally combines image, state and
action evidence itself. Declare `requires` only when a fresh matching Valen answer
must be a hard prerequisite before every action in that phase; the common boundary
enforces it for either interface. It is not a default grasp/transport declaration.

`DualMove.recoverable` allows a programmed correction, not automatic permission to
continue dependent task work. A recoverable partial endpoint still goes through
Valen; uncertainty or contradiction can pause it. Preflight rejection creates no
new image evidence. Critical faults and expired budgets return to the model.

## Task brief template

```text
Goal and source:
Required final facts (object identities + relations/states):
Permitted acceptance rules and measurements:
Constraints to preserve:
Known ambiguities / unobservable requirements:
Acceptance authority and when its verdict becomes available:
Action, step, wall-time, and recovery budgets:

For each milestone:
  prerequisites and evidence references:
  intended change:
  facts valid throughout this phase:
  postconditions to arm at the transition endpoint:
  relevant views / occlusions:
  alerts and critical facts:
  exit evidence needed before dependent work:
  repair scope and facts invalidated by repair:

Completion:
  required final evidence and cleanup:
  native verdict / visual verdict / planner assessment / execution status:
```

Keep fact IDs stable when meaning stays the same. When a rewrite changes meaning,
record a new question version; do not combine its temporal votes with the old one.
An evidence entry can be as small as:

```text
cup_in_tray: unresolved; step 42; raw yes=.67/no=.21/unknown=.12;
host unknown at threshold .8; wrist view occluded;
dependent drawer-closing phase blocked; next inspect after withdrawing gripper.
```

## Example: place a cup, then close a drawer

Suppose the user's goal is to put the red cup upright inside a blue tray and
close the drawer below it without disturbing the cup. A suitable dependency
sequence is inspect identities -> grasp cup -> transport -> release in tray ->
verify placement -> close drawer while preserving placement -> final inspection.
The real task and geometry determine whether this sequence is feasible.

For a PyRUA-Lean `cells` run, the following are valid contract declarations; actual
motion targets must be planned from the current robot and observations. These
are separate phases/cells, not a script to execute top-to-bottom without evidence.

**Approach/grasp:** do not yet require "held". Monitor any established scene
invariants and inspect the grasp/lift result before arming transport.

**Transport, after a verified grasp:**

```python
watch(
    "carry red cup to blue tray",
    {"held": "Is the red cup held between the right gripper fingers?"},
    alerts=[{"name": "cup lost", "when": {"held": "no"}}],
)
# Astra has already assessed the grasp using available evidence. A confident
# contradiction requests review; ordinary unknown is returned as information.
# Code may contain several meaningful actions and result-driven branches.
```

**Release, once the cup has been lowered into position:**

```python
placement = {
    "cup_in_tray": "Is the red cup inside the blue tray?",
    "cup_upright": "Is the red cup upright?",
}
watch(
    "release red cup in blue tray",
    placement,
    alerts=[
        {"name": "cup outside tray", "when": {"cup_in_tray": "no"}},
        {"name": "cup tipped", "when": {"cup_upright": "no"}},
    ],
)
# Release, then inspect the endpoint. Withdraw only as appropriate to the
# actual state and report; a pause may have skipped the intended withdrawal.
```

Do not start drawer work merely because the report says `clear`. Assess the
placement using images and permitted measurements. If fingers obscure the cup, resolve the visibility gap
with a justified retreat before relying on the placement.

**Closing the drawer:** carry `placement` into the approach and closing phases.
A partly closed drawer is expected during approach; require `drawer_closed` only
at the endpoint intended to complete closure. If a long closing primitive needs
intermediate monitoring, split it at meaningful endpoints.

```python
final_facts = {
    **placement,
    "drawer_closed": "Is the drawer front flush with the cabinet?",
}
watch(
    "complete drawer closure",
    final_facts,
    alerts=[
        {"name": "cup outside tray", "when": {"cup_in_tray": "no"}},
        {"name": "cup tipped", "when": {"cup_upright": "no"}},
        {"name": "drawer still open", "when": {"drawer_closed": "no"}},
    ],
    completion={
        "name": "placement and closure candidate",
        "when": {key: "yes" for key in final_facts},
        "confirmations": 1,
    },
)
# Perform the intended final closing primitive. Completion requests review and
# can unwind this cell, so do not assume later cleanup statements executed.
```

A flush front is a visual proxy for closure, not evidence that a hidden latch
engaged. If latching is part of the user goal, obtain an appropriate permitted
measurement or report that requirement unverified.

## Question design across task types

| Requirement | Useful visual questions | Separate evidence needed |
|---|---|---|
| Stacking | Is the short board resting on the two white supports? Is the board horizontal? | Exact clearances/angles need measurement |
| Sorting | Is the red item inside the left bin? | Hidden items may need another view |
| Insertion | Is the plug aligned with the socket opening? Is the plug seated against the socket? | Electrical connection cannot be inferred from appearance |
| Opening/closing | Is the drawer front flush with the cabinet? | Latch/lock state may be hidden |
| Transport | Is the yellow block held between the left gripper fingers? | A correct endpoint does not certify the entire trajectory |

Do not ask these indiscriminately. Select only requirements of the actual task,
and use visible identifiers that remain distinguishable in the selected evidence.

## Current host constraints

Consult [the implementation](../../../src/pyrualean/supervision.py) if the API
changes. At the time this skill was written:

- `watch(phase, facts, alerts=[], critical=None, completion=None, requires=None,
  invalidates=None)` accepts 1–12
  distinct questions and at most eight alert rules. Each question is at most
  500 characters; fact IDs are at most 64; phase names are at most 200.
- Rule `when` maps fact IDs to yes/no and is a conjunction. To stop when either
  of two invariants is false, use separate rules; one rule with both set to no
  would require both to fail simultaneously.
- `critical` defaults to `[]`. Explicitly listed facts request review once when a
  continuous unknown episode reaches the configured limit; this is not a persistent
  pre-motion gate. A new known source rearms the fact for a later unknown episode.
- The default threshold is .8; the default unknown window is two distinct
  endpoints. These are configuration defaults, not universal calibration advice.
- A single low-confidence answer may return `clear`. Missing data or malformed
  probabilities produce `monitor_error`, not an ordinary model unknown.
- New cells preserve fact/version uncertainty and explicit phase gates. After a
  normal visual review, Astra can generate subsequent actions from its combined
  evidence without waiting for a Valen yes. An alert/completion condition notifies
  once while continuously matching; new explicit counterevidence rearms it, and
  unknown alone does not. The same source cannot add a confirmation or change its
  stored vote. Phase changes do not reset the per-cell
  action or separate verification budget (default eight each).
- `evidence()` retains fact versions, observations, freshness and prerequisites
  across cells. `requires` and `defer_fact(..., until=phase, reason=...)` register
  persistent pre-motion gates. Unknown, contradictory or stale required facts
  block dependent actions in pause mode, even after a `clear` monitor result.
  Resetting a watch cannot erase gates. A motion invalidates all old facts by
  default; a narrower `invalidates` list must be justified by the planner.
- Executed endpoints trigger checks, including recoverable partial dual-arm
  motion; `show()` does not. Zero-step IK/preflight refusals can return to code
  without a visual vote. Critical faults and unclassified partial motion require
  review. A monitor timeout or pause does not undo the preceding physical action.
- `pause` interrupts the cell; `shadow` only records reviews. A completion rule
  does not call `finish`. Single-arm primitives are checked at their endpoints; move_both also
  exposes bounded native segments to the same supervisor.

If an essential contract exceeds the limits, restructure its phases/checkpoints
or change the host in a separately validated implementation. Do not merge
independent questions into an ambiguous sentence or drop mandatory checks just
because the contract is too large.

## Uncertainty audit using saved artifacts

Use `supervision/events.jsonl`, the matching `*-request.json` and
`*-response.json`, and the exact image paths referenced by the request.

1. Check fact IDs, probability sums, raw `choice`, and argmax. Reapply the
   configured threshold to distinguish explicit unknown from host abstention.
   Inspect `monitor_error` independently.
2. Inspect all camera views and current/previous frame ordering. Do not assume
   the wrist views show the target or that earlier images represent current state.
3. Hold the endpoint fixed and compare original versus equivalent atomic
   questions. Separately compare relevant camera subsets. Preserve each packet
   and response; do not change live robot actions as part of a replay.
4. Label results against visible evidence. If wording flips the answer, treat
   that as instability to investigate. A rewrite that drops orientation or contact
   requirements is not an equivalent acceptance test.
5. Test proposed changes on varied successes, failures, and occlusions before
   adopting them. Replayed frames do not provide independent stability votes.

The repository's [offline replay example](../../../examples/replay_supervision.py)
can evaluate a phase contract without robot actions. It does not establish model
accuracy, and loading a different camera subset remains an explicit experiment.

## Reconcile contradictory completion reports

```text
Execution: ended at step 847.
Native verdict: success, score 100.
Visual verification: incomplete; several relationships remain unknown.
Astra self-report: failure because it could not visually verify all conditions.
Conclusion: benchmark passed; the visual verifier did not certify completion.
```

This illustrates separate authorities, not expected numbers for another task.
The reverse case—Astra says success but native verdict is false—must be reported
as native failure and investigated against the actual task requirements. Where
no authoritative verdict exists, do not invent one or equate elapsed budget with
failure of the physical task; describe the achieved and unresolved conditions.
