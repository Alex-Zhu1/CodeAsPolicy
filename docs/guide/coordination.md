# Write dual-arm strategy in Python

PyRUA-Lean supplies a robot manual, executable `robo` methods and observations.
Astra chooses arm roles, target positions/orientations, ordering, release timing
and bounded corrections in generated Python. Skills provide manipulation knowledge;
Valen supplies visual evidence for the checks Astra declares. Neither supplies a
fixed handover/carry/recovery workflow.

`robo.move_both()` adds true simultaneous execution to the existing primitives.
It is available on RoboDojo's RPent and official XPolicyLab hosts, including RGB.
The generic CLI automatically includes it in the generated model API reference.
Other adapters keep their existing methods; synchronization is not emulated by
threads or silently replaced with two sequential moves.

## The code interface

| Call | Meaning |
| --- | --- |
| `robo.move_to(xyz, arm=..., ...)` | Move one arm; the other holds |
| `robo.set_gripper(arm, command)` | Change a gripper, independently of the next motion |
| `robo.move_both(left=..., right=..., ...)` | Plan both paths, then advance them on one native action clock |
| `robo.observe()` | Both TCP poses/openings, image step, native count and observation version |
| `robo.pose(xyz, quat)` | World TCP pose, metres, unit quaternion `[w,x,y,z]` |
| `obs.left.shifted(dx, dy, dz)` | World translation from this fixed observation, preserving orientation |
| `robo.motion_capabilities()` | Live synchronization, geometry, timing and force capability coverage |
| `robo.motion_profile(velocity, acceleration, jerk)` | Per-joint command limits in rad/s, rad/s², rad/s³ |

Each `move_both` target is an xyz sequence (retain current orientation), a `Pose`,
or `None` (hold that arm at the current pose). A Pose is an absolute target, not a
live instruction to follow the other arm. Existing `move_to` pitch/yaw/roll and
signed gripper commands retain their documented meanings.

Single-arm `Move.status` is `completed`, `rejected` (zero native actions), or
`partial`. `planned_path` describes the selected geometry; `path` is None until
native actions execute and does not by itself prove arrival. Check `reached` and
`stop_reason`. `ik_error` only describes geometry refusals; timing support, budget
and trajectory validation failures have distinct `stop_reason` codes and `note`
detail. A budget refusal is not evidence of an unreachable pose. Monitored
single-arm partial moves still require outer review before dependent code runs.

`grippers={"left": "keep", "right": "close"}` sets commands from the start of the
synchronized move. `keep` repeats the last command, retaining squeeze; only when
no command has been recorded does it use the measured opening. A program can
choose to open during motion, close both grippers together, or use separate
`set_gripper` calls. If release needs new evidence, separate the actions and inspect
that evidence first. No universal release sequence is imposed by the executor.

## Simultaneous movement

Assuming the scene permits this displacement and any necessary support has already
been established, Astra can generate:

```python
obs = robo.observe()
result = robo.move_both(
    left=obs.left.shifted(dz=0.03),
    right=obs.right.shifted(dz=0.03),
    keep_relative=True,
    max_steps=30,
)
print(result.reached, result.left.position_error_m, result.right.position_error_m)
```

With monitoring enabled, declare the appropriate `watch` contract before motion.
The displacement is an illustrative parameter, not a default task strategy.
Sequential cooperation is ordinary Python: move one arm, inspect its result, then
move the other. Single-arm, gripper and dual-arm calls can be freely composed;
there is no operation name, plan handle, lease or close call to manage.

Simultaneity alone does not preserve the arms' relationship. `keep_relative=True`
checks the initial left-to-right TCP transform on sampled paths and after each
native action, using `position_tolerance_m` and `orientation_tolerance_deg`.
It does not establish object support or implement force control. Both paths are
checked before either arm moves; there is no implicit near-pose or sequential
fallback. On tracking failure, both stop advancing together.

## Timed and continuous trajectories

The updated XPL client reports its physical timestep and substeps per native
action. When `motion_capabilities()["speed_control"]` is true, `move_to` and
`move_both` time-parameterize the IK path automatically. The default joint limits
are 1 rad/s, 4 rad/s² and 40 rad/s³; they are planning defaults, not hardware ratings.
`motion=robo.motion_profile(...)` selects other positive finite limits.

Timing first removes redundant IK samples within 0.0001 rad per joint. This
prevents tiny IK residuals from multiplying spline segments and duration. Real
foldbacks and phase boundaries remain. Every emitted physics substep is still
validated against joint/table geometry, and line moves and dual-arm paths retain
their 3 mm / 1 degree path checks. If simplification prevents path validation,
the planner retries with the original knots; it never widens geometry limits.

Choose an explicit motion phase from task intent and observed clearance:

For observed clear empty-hand travel or straight transport after a broad stable
grasp is verified, `motion="native"` uses the original
waypoint executor and simulator interpolation. It retains geometric checks,
gripper KEEP, action receipts and resume. A measured tracking miss stops both
native and timed moves before later waypoints or endpoint settling. Native
execution provides no C2 or derivative-limit
guarantee. Its preview reports `planned_steps`; timed profiles additionally
report `trajectory`. Keep smooth profiles for grasping and contact. For fragile
grasps, `path="line"` preserves orientation along the route; a cheaper joint
swing can rotate the object even when its final orientation is unchanged.

| `motion` | Intended use | Joint velocity / acceleration / jerk limits |
| --- | --- | --- |
| `"transit"` | Clear empty-hand travel to a pre-grasp region | 3 / 18 / 180 |
| `"approach"` | Final approach before grasp/contact | 1 / 6 / 90 |
| `"carry"` | Transport after verifying the held object | 1.8 / 10 / 120 |
| `"precision"` | Placement, insertion and fine alignment | 0.45 / 3 / 45 |

Units are rad/s, rad/s² and rad/s³. These are simulator planning choices, not
hardware ratings or contact-force guarantees. A phase is not inferred from an
open gripper, and transit does not add object/inter-arm collision checking.
Choose a pre-grasp region from current observations, then slow for the final
approach; verify the grasp before carrying. Split at decisions that need new
evidence. Each phase still accelerates and decelerates smoothly.

Omitting `motion` keeps the existing 1 / 4 / 40 defaults. Explicit phases use
the table above. For a long descent, apply slower limits only near contact:

```python
approach = robo.motion_window(far="transit", near="approach", distance_m=0.02)
placement = robo.motion_window(far="carry", near="precision", distance_m=0.015)
departure = robo.motion_window(far="carry", near="approach", distance_m=0.015, at="start")
estimate = robo.move_to(grasp_target, motion=approach, path="line", preview=True)
```

These windows share one C2 trajectory and introduce no intermediate stop.
The distance measures the nominal TCP path, not detected contact or clearance;
choose it from current geometry. A segment crossing the boundary uses slower
limits in full, and pure orientation moves use slower limits throughout.
Dual-arm distance sums the larger TCP displacement per segment on the common
clock; it is not an independent contact window for each arm. Returned
`trajectory.window.regions` reports derivative bounds and time for each region.
Near limits cannot exceed far limits. Joint/table/path checks still apply.

Verify grasp with a short lift before a longer carry. After release, move away
slowly until clear, then use transit. Join already-decided route segments with
`move_both(via=...)` to avoid repeated starts/stops, but retain observation and
release decision boundaries. Before grasping, allocate steps to the remaining
operation, including receiving-arm positioning and return. Select a mutually
reachable handoff region. Each preview starts from live state; summing unrelated
previews does not simulate a future action sequence.

Both move methods support `preview=True`: the return value is a read-only
planning dictionary, not `Move`/`DualMove`. No arm/gripper command is dispatched
and no executable plan is retained. For example:

```python
estimate = robo.move_to(pregrasp, arm="left", motion="transit", preview=True)
print(estimate)  # accepted, trajectory.planned_steps, limiting_factor, remaining budget
if estimate["accepted"] and estimate["planned_steps"] <= travel_allowance:
    result = robo.move_to(pregrasp, arm="left", motion="transit")
    print(result.reached)
```

`pregrasp` and `travel_allowance` come from the current task plan. Reserve steps
for the remaining manipulation and cleanup. An over-budget trajectory retains
its estimate in `trajectory`; limits are never silently raised to fit a budget.
Execution replans from live state, so a preview is not permission or evidence of
object/support conditions. Single-arm settling can use additional steps, bounded
by `settle_reserve_steps` and the native episode budget.

Use `via` to connect targets in one call, when the whole route can proceed without
an intermediate observation, release or contact decision:

```python
caps = robo.motion_capabilities()
if caps["continuous_trajectory"]:
    obs = robo.observe()
    result = robo.move_both(
        right=obs.right.shifted(dz=0.06),
        via=[{"right": obs.right.shifted(dz=0.03)}],
        motion=robo.motion_profile(velocity=0.8, acceleration=3.0, jerk=30.0),
        max_steps=200,
    )
    print(result.reached, result.trajectory)
```

Each `via` mapping names either or both arms. xyz retains the preceding waypoint's
orientation; an omitted arm holds its preceding pose. All waypoints are planned
before either arm moves. Joint position, velocity and acceleration are continuous
(C2), with zero endpoint velocity/acceleration. Intermediate knots have no inserted
settle pause; reversals can require zero joint velocity. Redundant straight joint
samples are removed for timing, while original geometry remains checked. Local
durations and shared knot derivatives are fitted against subdivided Bernstein
derivative bounds, so one constrained section does not dilate the entire route.
Sampled blending stays within 3 mm / 1 degree
of IK path segments; joint limits, sampled table clearance and requested relative
pose are checked again at every emitted substep. Full object/arm-arm collision
coverage remains unavailable.

A measured start joint up to 0.002 rad outside a planning limit may hold or move
back inward; the allowed envelope is bounded by that initial reading and cannot
grow further out. Larger violations still reject, as do table-clearance failures.
This accommodates small homing errors without expanding the target joint limits.

The client supplies joint position **and velocity** targets at each physical
substep, replacing the native arms' 80% interpolation / 20% hold sequence for
timed actions. It preserves native gripper processing, support-arm actions and
episode accounting. One native action still costs one step; slower limits can
exceed a call or episode budget and reject before movement. `trajectory` reports
planned duration and derivative bounds; `steps_used` reports consumed actions.
Tracking failure interrupts the unexecuted suffix, with the actual endpoint returned.

Continuity is in simulation time within one call. Network/model/visual-review waits
have no wall-clock continuity guarantee. Separate calls start/end at rest;
interruptions end the continuity guarantee. `home` retains geometric stepping.
Near a recorded release, it first withdraws behind that release plane along the
negative finger-pointing direction, preserving orientation. An interrupted
withdrawal resumes from the measured remaining distance. An unreachable or
incomplete withdrawal returns `Home.stop_reason` without starting the swing.
After a held object is released, the return stages above the starting TCP at
the cleared height before finishing the joint return. This protection persists
after a separate finger withdrawal until that arm is homed or receives a new close command.
Ordinary home retains the original lift/joint route. An unreachable or incomplete
staging line stops the call. This geometry does not detect objects elsewhere
along the return route.
Legacy RPent or old XPL clients without verified substep support keep their previous
geometric moves and reject explicit timed profiles or timed `via` before dispatch.
Explicit `motion="native"` also works on these hosts.

The current adapter has no force observation, target-force, force-threshold stop,
gripper-force or impedance API. Their capability flags remain false; gripper
obstruction and tracking lag cannot be used as force measurements. The task card
includes a startup capability snapshot, and the live query reports current support.
Both RoboDojo hosts assemble cards with the same code: native task text is copied
verbatim, initial TCP/angles are derived from recorded step-0 state, and the full
capability snapshot uses the live API's provider. Sources and configured budget
fallbacks are labelled. Missing initial measurements are unknown, not zero-valued
poses or closed grippers. Astra receives this card; it does not author it.

## Task records and a short execution plan

`card.json` keeps the complete startup card: the verbatim instruction, unrounded
public arm measurements, full capability snapshot, field sources, observation and
capture identifiers, clock domains, image provenance and SHA-256 digests. On the
native host it also references and hashes the recorded observation artifact.
Recorded index `0` is distinct from the native action count; the card records the
initial count and the remaining budget at card creation separately. Missing
identifiers stay unknown, and a configured budget fallback is explicit. The card
does not include hidden object poses or benchmark success labels.

The startup prompt uses a compact view: instruction, initial arm state, budget,
key motion/force/collision capabilities and observation alignment. Full capability
JSON and audit hashes stay in `card.json`. This is a startup snapshot; later code
queries `robo.state()` and `robo.motion_capabilities()` for current data.

In interactive `play` cells and programs, Astra can keep one current-goal note
using the predefined Python helper, alongside useful observation or action code:

```python
robo.state()
plan(goal="Check the grasp before lifting",
     next_action="Inspect the wrist image",
     done_when="The image supports proceeding with the lift",
     evidence="Not checked yet", status="in_progress")
previous = plan()  # latest note, even across program namespaces
```

Each update replaces the current note and appends a revision. Status is `pending`,
`in_progress`, `done` or `blocked`; `done` describes only that goal. These are
planner-authored claims, not verified evidence or task completion. They do not
change motion permission, watch prerequisites, retry budgets or `finish`.

The episode directory saves `execution_plan.json` and `execution_plan.jsonl`
after the first update, including the task-card hash and the last known
observation/action context. Updates survive a later code error, but subsequent
movement never refreshes an old note automatically. The helper itself does not
capture images or execute native actions. Use it in an existing code call; no
extra planning turn is required. Standalone generated policies do not receive
this interactive helper. The task card remains the host's input record.

## Local correction and model replanning

`DualMove` contains `status`, `reached`, `left`/`right` actual poses and errors,
`steps_used`, `stop_reason`, `recoverable`, `observation`, `remaining_steps` and
`checked`. Status is `completed`, `rejected` (no native action), or `partial`.
Arrival is motion success, not task success or proof that an object stayed held.

A recoverable result returns to generated Python. For example, if Astra has
already derived a bounded set of viable alternatives from current observations:

```python
# candidate_targets contains (left_target, right_target) alternatives chosen by Astra.
# repair_steps_left is a persistent planner allocation, initialized once for this phase.
for left_target, right_target in candidate_targets[:2]:
    if repair_steps_left <= 0:
        break
    result = robo.move_both(left_target, right_target, max_steps=min(30, repair_steps_left))
    repair_steps_left -= result.steps_used
    print(result)
    if result.reached or not result.recoverable:
        break
# Only continue dependent work after checking result and its required evidence.
# If neither candidate works, end this cell so Astra can inspect and revise it.
```

Alternatives must account for the actual partial endpoint and object state; an
automatic retry of the same failed target is not a correction strategy.
`recoverable` means local code may consider a correction, not that it must retry.
Preflight rejection produces no visual vote. A recoverable partial move supplies
a real endpoint to Valen before returning, so a declared alert/uncertainty review
can still interrupt the program. An IK refusal with known zero native progress
from the existing single-arm controller can also return for local handling.
Other unclassified single-arm partial failures still require outer review.

Controller exceptions, violated execution invariants, expired budgets and visual
review requests unwind the code block. The tool's `coordination` feedback preserves
both arm states and consumed native actions even if an assignment never completed.
Do not catch `CoordinationPause` or `SupervisionPause`; write revised code from the
actual scene on the next turn. Previously executed actions are never rolled back.

## Checks chosen by Astra

For a handover where the donor is the only existing support, establish receiver
support before opening the donor. A useful contract at that boundary is:

```python
# receiver_support was defined and observed during receiver grasp/probing.
watch(
    "donor release",
    {"receiver_support": "Is the object supported by the right gripper?"},
    requires={"receiver_support": "yes"},
)
robo.set_gripper("left", robo.OPEN)
```

The host enforces declared `requires` at the common motion boundary, including
single-arm and dual-arm calls. Freshness and question versions persist across
cells. Astra chooses which facts are prerequisites, which actions affect them,
and what final evidence is needed. The runtime does not infer a donor/receiver
role from a gripper command. Use the existing general watch mechanism for task-specific dependencies.

Valen answers atomic visual questions. It does not choose distances, reorder the
program or generate corrections. An optional `strategy.verify_grasp` helper can
support a suitable evidence-driven probe; custom Python is equally valid when
other observations or geometry call for a different verification method.

## Bounds and capability limits

`max_steps` and `timeout_s` bound each call (defaults 60 native actions / 120 s).
They include planning and checkpoint waits. `checkpoint_steps` defaults to 4,
maximum 8, before returning through the existing supervisor. Episode budgets,
code time/output/action limits and registered prerequisites remain in force across
primitive calls. Calling `move_both` again does not reset the episode counter.
Planner-specific retry/phase allocations belong to the generated program and must
be preserved across cells; `DualMove.remaining_steps` is only this call's allowance.

The current host checks sampled arm/table geometry, IK, joint/TCP step limits and
requested relative transforms. It does not check full arm-arm, self, held-object
or obstacle collision volumes. `require_inter_arm_clearance=True` returns a
rejection before motion. Nominal Cartesian spacing permits 1 mm / 0.5 degree IK
interpolation error; it is not a speed setting or continuous collision guarantee.
An interrupted sequence stops sending further commands and retains the last
native gripper commands. An unresponsive native RPC still needs the existing
outer process watchdog.

This replaces the earlier mandatory `coordination → plan → execute` surface.
Preflight and segmented execution remain internal implementation details.
