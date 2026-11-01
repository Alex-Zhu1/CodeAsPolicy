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
| `robo.motion_capabilities()` | The host's actual synchronization and geometric-check coverage |

Each `move_both` target is an xyz sequence (retain current orientation), a `Pose`,
or `None` (hold that arm at the current pose). A Pose is an absolute target, not a
live instruction to follow the other arm. Existing `move_to` pitch/yaw/roll and
signed gripper commands retain their documented meanings.

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
