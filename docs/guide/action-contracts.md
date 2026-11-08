# Plans, evidence and motion constraints

A useful manipulation plan accounts for the whole task: what must remain unchanged,
what evidence permits the next interaction, which poses leave subsequent actions
reachable, and how the robot will exit. Reaching one TCP target does not establish
any of those properties. The interfaces below connect these planning decisions to
execution without embedding task names, fixed coordinates or simulator object poses.

## Stage contracts

The existing `plan` helper accepts an optional executable `contract` in both cells
and programs. It works independently of the Valen monitor mode.

| Field | Evaluation | Lifetime |
| --- | --- | --- |
| `requires` | Before each action dispatch | Current stage |
| `preserves` | Before and after each action dispatch | Persistent across stages and cells |
| `ensures` | `plan(status="done")` and `finish(status="success")` | Current stage |
| `constraints` | Motion compilation and native execution guards | Frozen into prepared operations |

The first three fields map names to zero-argument, read-only functions returning a
Python `bool` or `monitor.ConstraintCheck`. Read current observations inside these
functions. `None`, invalid results, exceptions and missing evidence block the
dependent action. A precomputed `ConstraintCheck` expires after an attempted
action; rebinding that same result does not refresh it. Checks cannot dispatch
another action or change the plan while being evaluated.

Public task-card requirements with `kind="invariant"` must have matching names in
`preserves` before any action. Naming a check establishes declaration coverage,
not that the predicate actually measures the requirement. There is no automatic
translation of natural-language requirements into reliable physical detectors.
Ordering and final-state criteria still need appropriate stage predicates.

For example, after observing the task and implementing its checks:

```python
plan(
    goal="Transfer the retained object into the observed receiving region",
    next_action="Approach the receiving region with both routes clear",
    done_when="Current observations establish the required object state",
    contract={
        "requires": {"receiving_region_clear": check_receiving_region},
        "preserves": {"preserve_surroundings": check_surroundings},
        "ensures": {"object_at_destination": check_destination},
        "constraints": observed_bounds,
    },
)
```

The names and callbacks above are task-derived inputs, not built-in detectors.
Use the task card's actual invariant IDs. Construct `observed_bounds` from public,
current evidence and a justified geometric envelope. Refresh the relevant bounds
and prepare a new operation when their assumptions change.

Supplying a new contract replaces `requires`, `ensures` and `constraints`, while
merging `preserves`. Omitting `contract` retains the current one. Persistent checks
cannot be removed by changing a goal label or passing an empty contract. A failed
persistent invariant also blocks corrective motion: first establish valid current
evidence and an appropriate check; this API does not infer a safe repair exception.
Unmet prerequisites belonging only to a later stage need not block independent work.

Geometry bounds flow into `execute_motion`, `align_local` and every staged grasp
phase, including native open/close ticks. An explicit action cannot override an
inherited bound with a conflicting one. A previously prepared operation cannot
acquire new bounds by reusing its ID. Direct primitives that cannot enforce active
bounds are refused; use an intent or staged grasp under those bounds instead.

On pause, read-only observations remain available. Feedback records the failed
assumption, evidence state and metric details when available. Re-establish the
assumption or choose independent work before moving again. Completed action receipts
remain completed even when an after-action check fails. After a policy-session
restart, saved check declarations require fresh executable bindings; serialized
results never restore live evidence. Host operation recovery has its own session
and receipt restrictions.

## Match evidence to the decision

`monitor.check_hold(robo, reference, arm="right", space="image")` checks current
object-following evidence from the original post-closure baseline. It requires
comparable features and obstructed closure. `space="spatial"` requires measured 3D
following; wrist-image consistency alone cannot pass that stronger check. Neither
proves object identity, contact forces or future retention. Install a hold check
after capture/probe evidence exists, as a prerequisite for the work depending on it.

With visual supervision enabled, `monitor.check_fact(robo, fact_id, expected="yes")`
reuses an already declared and verified visual fact. It makes no additional VQA
request. Host registration, observation identity, source freshness and action
revision must agree. Unknown, stale and unavailable facts fail the check, including
when `expected="no"`. The answer remains fallible visual evidence.

## Explicit collision envelopes

RoboDojo accepts a `geometry` field inside motion constraints:

```python
geometry = {
    "arms": {
        "left": [
            {"name": "gripper", "xyz": left_local_center,
             "radius_m": left_radius, "uncertainty_m": measurement_margin},
            {"name": "held_object", "xyz": object_local_center,
             "radius_m": object_radius},
        ],
        "right": [
            {"name": "gripper", "xyz": right_local_center,
             "radius_m": right_radius},
        ],
    },
    "clearance_m": required_clearance,
}
```

Sphere centers are in their arm's TCP frame. Supply envelopes that conservatively
cover the relevant gripper geometry throughout its entire opening range and the
held object's extent. The host does not obtain these from privileged object state.
Inflation for uncertainty and clearance is explicit.

Checks cover obstacles in `keep_out`, both represented arms (including an arm not
commanded in the current move), and translation/rotation between consecutive poses.
For represented arms, these envelopes replace the legacy TCP-point `keep_out`
check: include a TCP-centered sphere if that point must also be covered. Undeclared
arms retain the legacy point check. `z_bounds`, tilt and corridor bounds still apply
to TCP poses.

Intentional contact can be declared with `allowed_contacts`, for example a pair
`["left:pad", "right:pad"]` or `["left:pad", "keep_out:0"]`, plus an explicit
`max_penetration_m`. Exceptions apply only to the named pair and penetration bound.
They do not disable other checks or establish that the contact is mechanically safe.

The sweep model interpolates TCP translation and shortest-path orientation between
supplied poses, with conservative subdivision for offset spheres. It does not
certify unsampled servo/FK motion, undeclared robot links, same-arm self-collision,
contact dynamics or unobserved geometry. Inspect the returned `geometry_coverage`.
These are declared-envelope guards, not a full-robot collision certificate.

## Compare short sequences before execution

`execute_motion(preview=True, following=[...])` evaluates several pose legs without
moving or reserving the intent ID:

```python
candidate = robo.execute_motion(
    "candidate-transfer", left=approach_left, right=approach_right,
    holding={"left": "retain", "right": "empty"},
    constraints=observed_bounds, preview=True,
    following=[{"left": interaction_pose}, {"left": exit_pose}],
)
```

The host starts from measured joint state and rolls hypothetical joint state through
the legs, so each later leg starts from the previous leg's planned endpoint. Up to
32 following legs are allowed. This is one continuous route through waypoints;
intermediate stops and stage transitions are not simulated. Every leg uses the same interaction, holding and
constraint declarations. Directions, local alignment and `corridor_m` are not
supported in a multi-leg preview. This interface is preview-only: execution still
uses observed-state intents and checks between actions.

Compare candidates against invariants, geometric margins, both arms' future access,
evidence quality and action cost. A shorter feasible first leg alone is not enough.
The preview reports nominal boundaries and explicitly lists contact, grasp changes,
object motion and task success as unverified. It cannot simulate a handoff or prove
that a grasp will succeed. Re-observe after those interactions and preview the next
group under the resulting evidence and geometry. When a check fails, revise the
dependent sequence rather than repeatedly changing a target coordinate.
