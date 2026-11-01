# Reusable local feedback loops

Generated cells and online `run_program` programs receive a predefined
`strategy` module. These helpers factor out patterns already present in
RSI's `general_pickup.py` (clear occlusion, relocalize, bounded retries) and
`push_T.py` (observe the result, measure residual, correct a bounded number of
times). They contain no task coordinates, object truth or simulator access.

| Helper | Inputs | Verified condition | Handoff to planner |
| --- | --- | --- | --- |
| `strategy.relocalize` | `observe()`, `valid(observation)`, optional `prepare_view(previous, attempt)`, `max_attempts` | A fresh observation passes the supplied validity predicate | No valid observation within the attempt limit; no alternative view callback |
| `strategy.correct` | `observe()`, `complete(observation)`, `act(observation, attempt)`, `max_attempts`; optional residual `error(observation)` | The observed postcondition is true, including after the final allowed attempt | Unknown evidence, exhausted attempts, invalid residual, or repeated lack of progress |

Both work with any adapter whose documented public API can implement the
callbacks. They never enable hidden primitives under `no-vla`. They return
`StrategyResult(status, reason, attempts, observation, last_result)`:

- `status='verified'` establishes only the caller's local predicate.
- `status='replan'` requires the model to inspect the result before dependent
  motion. Print the result, request a current view, and end the cell.
- Primitive exceptions, execution limits and supervisor pauses propagate;
  helpers do not automatically repeat failed controller calls.

Predicates must return Python `bool` or `None` (unknown). Wrap a NumPy boolean
with `bool(...)`. Attempt limits must be integers in 1..32. In `correct`, the
residual must be finite and nonnegative; smaller means better. Progress must
exceed `min_progress` (default 0.001, in the residual's units), otherwise the
stagnation count increases. `no_progress_limit` defaults to two corrections.
`correct.attempts` counts actions; `relocalize.attempts` counts observations.

## Relocalize with another camera

For LIBERO, use segmentation from the currently selected view. This example
selects another camera after an invalid observation; it does not move an arm
to a hard-coded pose.

```python
view = {"camera": "agentview"}

def change_view(previous, attempt):
    view["camera"] = "wrist"

located = strategy.relocalize(
    lambda: robo.segment("black bowl", camera=view["camera"]),
    lambda s: bool(s.found and s.world_xyz is not None),
    prepare_view=change_view,
    max_attempts=2,
)
print(located)
if located.status == "replan":
    robo.show("agentview")
# End this cell; use located.observation only after checking its status.
```

## Correct a reachable approach target

For LIBERO, `target` below must already be a scene-derived, reachable approach
position. A cell-local loop checks fresh proprioception after each short move.
This verifies the end-effector position, not grasp success or collision safety.

```python
target = np.asarray(target, dtype=float)

def remaining(state):
    return float(np.linalg.norm(np.asarray(state.eef_pos) - target))

approach = strategy.correct(
    robo.state,
    lambda state: remaining(state) <= 0.01,
    lambda state, attempt: robo.move_to(target.tolist(), max_steps=20),
    max_attempts=3,
    error=remaining,
    min_progress=0.002,  # metres
)
print(approach)
if approach.status == "replan":
    robo.show("agentview")
# Return to the planner before issuing a dependent grasp.
```

For grasping, use the same `correct` loop with fresh observation of the
intended object being held as the completion predicate. A gripper close command
returning normally is insufficient. Adapters expose different evidence: some
return `holding`, while others require gripper state and image checks. Return
`None` when object identity or holding cannot be established locally; let the
outer model inspect the camera images. Each `act` callback must itself be short
and must check `reached` before a dependent close/lift.

## Verify a grasp

`strategy.verify_grasp(observe, close, probe, compare, *, repair=None,
max_attempts=2, allow_action=None)` packages closure and support verification.
It does not supply robot geometry, a grasp pose, or a visual detector.

| Callback | Responsibility |
| --- | --- |
| `close(attempt)` | One closure at an already reached, observed grasp pose; return its controller result |
| `observe()` | Return a `strategy.GraspObservation` from current public sensor/image evidence |
| `probe(closed, attempt)` | One small, bounded diagnostic move; return its controller result and check reachability |
| `compare(before, after)` | Return bool/None for the intended object following the gripper and remaining stable relative to it |
| `repair(observation, next_attempt)` | Optional changed grasp/approach after a definite failure, never a replay of a failed cell |
| `allow_action(kind, attempt)` | Optional bool budget check before each `close`, `probe`, or `repair` |

Read visual hold evidence from the persistent ledger:

```python
sample = strategy.GraspObservation.from_evidence(
    evidence(), "left_holds_key", holding=None,
    data={"wrist_image": robo.image("cam_left_wrist")},
)
```

`holding` is a fresh public contact measurement when available, otherwise None.
Do not reuse the closure's `holding` flag after a lift. `data` can carry permitted
images, measured object/gripper positions or keypoints for the comparison callback.
The same identity must be observed before and after the probe; a constant `True`
comparison or gripper motion alone does not establish that the object followed it.
The helper requires fresh visual `object_held=True` **and** a positive paired
comparison. It rejects reused observation IDs, non-increasing state versions,
changed fact/question versions and stale ledger facts. It never edits host evidence.

It returns `GraspResult(status, reason, attempts, observation, before_probe,
last_result)`. Status is `verified`, `failed` or `uncertain`. Both failed and
uncertain require review before transport/handover; local verification is not
task success. Fresh contact can permit a small diagnostic probe while visual
support is unknown. Unresolved support after the probe exits without repair.
Only definite failure can invoke the optional repair callback, up to
`max_attempts` closures (default two).
Exceptions and host pauses propagate, preserving the actual scene and budgets.

Callbacks still need `watch` contracts. Keep the diagnostic probe small enough
that an unconfirmed grasp cannot become unverified transport. Retain persistent
hold prerequisites for later transport and donor-release phases. On a supervisor
pause, inspect which callbacks executed before writing new code; the interrupted
helper does not automatically resume and must not be blindly restarted.

Use `allow_action` to reserve steps/time for later phases. Accumulate actual
`steps_used` from controller results and use conservative cost bounds for the
next callback. `robo.state().step` may count observation records instead of native
control steps; use tool feedback `observation.step_id` for the episode count.
The hook is checked before callbacks; it cannot preempt a callback that spends
more than its allocation. Host execution deadlines and primitive limits still apply.

The helpers are injected by `ArmToolkit`, not into the offline `generate`/`run`
protocol. RSI's 54 fixed programs remain unchanged. Its task-specific geometry
has not been promoted to a generic online strategy library.
