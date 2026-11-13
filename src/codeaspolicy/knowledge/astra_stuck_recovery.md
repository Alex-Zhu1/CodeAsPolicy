# Astra stuck recovery: runtime core

Apply this procedure only after your explicit finish(status="stuck") declaration
returns status="stuck_confirmed", episode_finished=false. The live episode,
namespace, recorded actions, evidence and budgets persist. A pause is not a
rollback. Do not replay the failed cell, reset attempt counts or assume an
interrupted assignment completed. episode_finished=true is terminal: no skill
can reopen an ended episode or refund its budget.

## Generate a revised program

Recovery changes Astra's reasoning and generated Python; it does not run a preset
robot workflow. Reconsider roles, target geometry, action ordering and evidence
boundaries. Compose single-arm/gripper calls with move_both when synchronized
motion is useful. A known recoverable result may be handled by bounded code before
stuck is confirmed; critical faults and visual pauses still return to the model.
Choose the next sequence from the actual scene, without an operation object or
mandatory grasp/transfer/release recipe. Keep planner retry allocations across
cells; per-call move_both limits do not reset native episode usage.

## Reconstruct the blockage

Inspect current images, robot state, executed/failed actions, monitor reports,
remaining calls/time/steps, and prior attempts. With supervision enabled, inspect
evidence() for fact versions, observation IDs, freshness and phase prerequisites.
Without supervision, use available state/images and your explicit evidence record;
watch, evidence and defer_fact are only bound when monitoring is enabled.
Distinguish code errors, missed grasps, unobservable geometry, model abstention
and lack of physical progress. Valen unknown is not a failed grasp or unknown
execution. Ordinary visual review returns judgment to Astra: combine current
images, measurements and action results to choose the next program. A continuous
visual issue is reported once; the next cell does not require a Valen yes unless
an explicit requires/defer_fact gate demands it. Preserve the original unknown
record. A successful Python call does not establish grasp
or alignment. Never relabel unknown as yes or read hidden evaluator truth.

## Reassess dependencies for the proposed next action

Ask whether a false condition would make this specific action fail, disturb
established progress, or violate a task constraint. Keep required prerequisites
and invariants. For essential but unobservable facts, obtain useful evidence or
choose an action that does not depend on them. A future milestone need not already
be satisfied during approach: defer its blocking check with a concrete boundary.
Remove planner-invented assumptions only with a causal reason; preserve the task's
actual acceptance conditions and every still-relevant prerequisite.

For example, uncertain keyhole alignment may not prevent grasping a distant key,
but is essential before insertion or a motion near the lock. Current grasp
uncertainty cannot be deferred merely to keep transporting the object, but a
Valen abstention alone does not establish that the grasp is uncertain in all
available evidence. Reassess it using images, object motion and gripper state.

## Select a bounded alternative

Consider materially different candidates: reorder independent work, change arm,
grasp or approach, expose a useful view, move to a supported intermediate pose,
or repair only the affected relationship while preserving other progress.
For the chosen candidate state prerequisites, expected progress or information,
possible effects on established facts, cost and the observation that selects the
next branch. Execute a short phase with a bounded repair budget. Two unproductive
local attempts trigger causal review, not automatic task termination. Do not repeat
the same failed approach without a changed reason to expect progress. Do not spend
motions solely to manufacture Valen votes or lower thresholds to obtain a pass.

Before the next attempt, allocate remaining native steps/time between this repair
and unfinished milestones. Preserve that allocation across cells. Distinguish
controller motion_rejected/motion_incomplete from Valen monitor_error; use the
reported reach/collision reason to change geometry instead of retaking images
of an unexecuted target. Native counts come from observation.step_id and public
steps_used, not necessarily robo.state().step.

For a slipping grasp, consider strategy.verify_grasp: bounded closure, a small
support probe, and fresh paired object evidence. Mechanical holding alone cannot
verify support. Unknown stops the helper; definite failure may use its limited
repair callback. Its allow_action hook can preserve the downstream budget.
Choose a different supported grasp or view after unproductive attempts. During
handover keep donor support until receiver support is established, then verify
again after release. Do not use task success or hidden state as grasp evidence.

## Restore evidence before dependent work

For each deferred fact record its question/version, latest evidence and unresolved
status, why the next action is independent, prohibited dependent actions, the
reactivation boundary and the new evidence needed. With supervision enabled use:

defer_fact("aligned", until="insert", expected="yes",
           reason="Independent transport may continue; insertion requires alignment.")

The fact must already be known to the host. This registers a persistent gate; it
does not establish truth or remove another gate. In an earlier phase, obtain fresh
evidence establishing the fact BEFORE entering the dependent phase. Use stable
phase names. Changing cells, omitting a fact, or requires={} cannot erase a gate.
Do not rename phases to bypass a real dependency or reinterpret a gated fact ID.

Revise alerts as well as critical when deferring a future goal, while retaining
current invariants. critical defaults to [] and explicitly opts into one review
per continuous unknown episode; explicit contradictions need alerts. Use requires
only for a necessary hard visual dependency, not as a default held=yes transport
rule. Ordinary reviews may be resolved by Astra's combined evidence without
changing Valen's answer. clear is not proof that all required facts hold. Motion
invalidates old evidence by default; narrow invalidates only with a reasoned
account of unaffected facts. Changing watch never clears fact uncertainty or
registered prerequisites. The host checks declared phase
gates, but cannot infer undeclared dependencies or final acceptance from geometry.

Do not catch a supervision pause. Write the revised plan in the next cell, inspect
actual results, and update the evidence record. If a repair intentionally changes
an invariant, re-establish it before later dependent work. Preserve all budgets.

## Continue or finish with evidence

Continue if a feasible candidate can make meaningful progress or resolve a relevant
uncertainty within the remaining budget. After this review, declare stuck again
only when no supported useful next action remains: alternatives made no progress,
an essential capability/evidence is missing, or the budget is insufficient.
The host bounds recovery dispatches; another stuck response may be terminal.

Report achieved progress, unresolved blockers, alternatives tried and deferred
obligations. Keep native/task verdict, visual verification, planner assessment and
execution status distinct. Respect native termination and final task requirements;
do not rewrite the evaluator, silently start another episode or claim unverified
success. Success/failure finish requests remain terminal.
