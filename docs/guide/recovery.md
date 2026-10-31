# Closed-loop execution and code recovery

The online `cells` arm already has the same basic loop as computer use:
the planner sees observations, generates code, the host executes it against
`robo`, and the planner receives the result. Code is an action tool. Its output
must describe both Python execution and what actually happened to the robot.

```text
Task + current observations
          |
          v
Planner generates one short phase
          |
          v
Host validates and compiles the whole cell
          |                         |
          | valid                   | invalid: no cell execution
          v                         |
Execute primitives + check outcomes |
          |                         |
          +---- failure/pause ------>+--> failure + state + evidence
          |                                  |
          | normal return                    v
          v                         Planner repairs/replans
Verify progress in the scene        from the current scene
          |                                  |
          +---- next phase <------------------+
          |
          v
Finish or stop at the existing episode budgets
```

There are two feedback loops. Within a cell, code branches on primitive results
and can perform bounded local corrections. Between cells, the model interprets
errors and observations, then changes the code or plan. Optional
[visual supervision](../visual-supervision.md) adds checks at primitive endpoints
and can return control before the rest of a cell executes.

## What happens when generated code is wrong

| Failure | Detection | Recovery |
| --- | --- | --- |
| Invalid Python, including invalid top-level `return`/`yield` | Compile the entire cell before executing its prefix | Fix the reported source location; no part of that cell ran |
| Blocked Python introspection | Existing sandbox preflight | Use the documented API; the cell/program was not executed |
| Missing import or program entry point | Program loader / Python exception | Inspect the load error; module code may already have run |
| Undefined variable, wrong API arguments, arithmetic error | Runtime exception | Earlier assignments and actions persist; repair only the remaining work |
| Controller/service exception | Primitive ledger plus `ToolError` | Treat the action as attempted with potentially partial effects; inspect current state |
| `Move.reached == False`, failed grasp or missing mask | Generated code checks returned fields; a failed `assert` becomes a `verification` report | Re-observe and change target, parameters or strategy before the dependent action |
| Code returns normally but the scene is wrong | Planner's observation or optional visual supervisor | Replan from visual evidence; normal execution does not establish task success |
| Wall-clock/step/call/program limit | Existing host and robot guards | Finish; repair does not reset budgets |

Compilation is not a proof that the code is correct. API mistakes, bad geometry,
and incorrect expectations can still appear only during execution. Assertions
only check conditions the policy author supplied. The host does not silently
interpret every false primitive result as a Python failure: false results are
also legitimate inputs to a local recovery branch.

## Failure feedback

Code failures return the existing `exception` text plus two structured fields,
including with `feedback=pure`:

- `failure`: `kind`, `stage` (`validation`, `load`, `execution`), exception `type`,
  bounded `message`, and `location` when available. Kinds include `syntax`,
  `sandbox`, `import`, `runtime`, `primitive`, `verification` and `timeout`.
- `recovery`: `next_action`, `retry_allowed`, remaining `calls_left`,
  `stop_reason`, `same_failure_count`, motion attempt/completion counts,
  simulator steps consumed, `last_motion`, current `state`, and persistence
  instructions. `calls_left=null` means no call-count limit was configured;
  the episode's wall-clock and simulator limits still apply. `retry_allowed`
  is advisory: a later deadline or terminal observation can still stop work.

An acknowledged primitive call is counted by `completed_motion_calls` even when
its measured result is false. A failed controller call counts as an attempt;
zero recorded simulator steps do not prove it had no physical effect.
`namespace_may_have_changed` is conservative for runtime/load errors. Program
namespaces are fresh on the next attempt; cell namespaces and workspace files
persist. Neither mode rolls the scene back.

Identical source, exception kind/type and source line share a failure counter
for the episode (it is not reset by an observation-only cell). At three
occurrences, the report recommends `change_approach_or_finish`. This is advice,
not a new automatic retry mechanism or hard retry cap. The existing episode
budgets enforce the limit. With `max_programs=1`, the first failure recommends
`finish` and grants no second attempt.

The report follows the robot adapter's public state contract. It does not add a
live benchmark success signal to robots, such as RoboDojo, that judge only the
final scene. Image delivery still follows `images=on-motion|on-demand|none`;
with `on-demand`, use an observation cell to request fresh views if the failed
cell did not reach `robo.show()`. Supervisor pauses retain their existing rule
of attaching the exact inspected evidence, even with normal images disabled.

Large stdout cannot hide the failure/recovery fields. Oversized replies omit
bulky state and last-action details while retaining a valid JSON diagnostic;
inspect state/images again when those details are needed. The full structured
failure and recovery report is also recorded in `tool_turns.json` by `play`.

## Repair from the current scene

Suppose a cell executes an approach and then raises `NameError`:

```python
approach = robo.move_to([0.05, -0.10, 0.60])
assert approach.reached, "approach did not reach target"
print(grasp_result)  # undefined: the approach has already executed
```

The host reports one motion attempt, the last motion and current state. The
planner fixes the remaining work instead of replaying the approach. When an
exception or supervision pause interrupts an assignment, its destination may
still contain a value from an earlier cell: re-read state/results rather than
assuming the assignment completed. Recovery code must verify its own outcome
before continuing to the next task phase.

This is why blindly retrying an entire Python block is incorrect for robot
control. Scene changes, gripper state, object displacement and prior Python
assignments are not transactional.

## Runtime behavior and scope

MCP `isError`, `tool_turns.json`'s `ok`, and direct OpenAI runtime events agree on
execution errors. A supervisor pause remains a request for review, not a Python
error or task-success verdict.

The direct `openai` runtime stops executing the rest of a model reply after a
tool failure or supervisor pause. Pending tool IDs receive explicit `skipped`
results, including a prematurely queued `finish`, so the next model decision
can incorporate the error. Skipped calls do not consume toolkit execution
slots. A successful `finish` also prevents further queued actions. Codex and
Claude own their external agent loops; the common prompt requests one dependent
code call at a time, and MCP delivers the error, but this host cannot cancel
already queued requests inside those clients.

`generate` followed by offline execution and the fixed PyRUA-RSI programs still
have no model running during evaluation. They can recover only through their
own code. Online model repair requires `play --arm cells` or a multi-attempt
`program` run; it must not be added silently to a single-shot benchmark.

The optional supervisor checks endpoints, not every control step. Hard
preemption of uncooperative code remains the host process's responsibility.
The new recovery reports do not add collision checking, automatic rollback,
guaranteed visual verification, or a real-robot emergency controller.

## Validation

`tests/test_recovery.py` exercises compile-before-motion, persistent state after
partial execution, primitive failures, postcondition failures, repeated errors,
single-shot budgets, program import/SystemExit failures, and MCP error flags.
The scripted runtime tests verify that the model receives the failure, queued
actions are skipped, and a later repair runs without replaying prior motion.
These tests use a fake backend; they establish execution behavior, not measured
task-success improvements or visual-model accuracy.
