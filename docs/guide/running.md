# Running Episodes

`pyrualean play` runs one episode: it boots the robot, gives the agent the task and serves its code until the
episode is over.

The default launch uses `cells`, Codex with `gpt-6-astra` at `high` reasoning,
100 code calls and a 5400-second wall-clock ceiling. Images are requested on demand,
feedback is `pure`, and the primitive set is `no-vla`. Both CLI entry points and
Python's `play_episode` share these launch settings.

Valen monitoring and confirmed-stuck recovery are off by default. A normal run
does not require a Valen service. To opt in, start the
[Valen service](../visual-supervision.md#run) and pass `--monitor-mode pause` for
review interruptions or `--monitor-mode shadow` for reports only. The service
address is `http://127.0.0.1:18001/predict` (`--valen-url` overrides it). Supervision
instructions load only when monitoring is enabled. Enable bundled recovery with
`--stuck-recovery`, or supply `--stuck-recovery-skill <file>` for a custom procedure;
its body is supplied only after confirmed stuck. `--stuck-recovery-rounds` sets
the limit without enabling recovery. The `program` arm rejects either feature.

Executable [action contracts](action-contracts.md) remain active independently
of Valen: public task invariants require check bindings before motion. The planner
declares stage prerequisites, completion checks, geometry and multi-stage previews.

```bash
pyrualean play --arm cells --suite libero_spatial_swap --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --budget-s 7200 --out runs/demo --cuda-device 0
```

`--robot` picks the robot (`libero`, the default, `robotwin` or `robocasa`) and `--suite`, `--task`, `--seed` the
task instance; each [robot page](../robots/index.md) says what they mean there. Always pass `--seed`: `play` sends
`0` when it is left out, whatever the robot's own default. RoboDojo runs through XPolicyLab
([RoboDojo](../robots/robodojo.md)).

## Cells or programs

| `--arm` | What one LLM call runs | What carries over |
|---|---|---|
| `cells` | a cell in a persistent namespace with `robo`, like a notebook | variables, functions, imports, files |
| `program` | a complete program, `def run(robo): ...`, in a fresh namespace | files in the workspace, and the conversation |

`--max-programs N` caps the number of programs on the `program` arm; `1` is a single shot whose result the agent
still sees.

## What the agent sees after a call

| Option | Values |
|---|---|
| `--images` | `on-demand` (default): images requested by the code with `robo.show(camera)` or `display()`. `on-motion`: current camera images after every call that moved the robot. `none`. Failure/pause reports can include recorded evidence; see [recovery](recovery.md). |
| `--feedback` | `pure` (default): code output and execution diagnostics. `rich`: adds the robot's state and the primitives it called. |

## Budgets

| Option | Meaning |
|---|---|
| `--max-turns N` | A decision budget: at most N code calls (default `100`; `finish` is not counted). The agent knows calls are limited, not how many. `0` removes the call-count limit; the wall-clock and native action budgets still apply. |
| `--budget-s S` | A wall-clock budget in seconds, thinking included (default `5400`). With a call-count limit it is a safety ceiling: set it well above what the calls take, so that outcomes do not depend on how fast the provider is (the paper's runs used `7200`). |

## VLA policies, guides and shared servers

- `--primitives no-vla` is the default: VLA primitives disappear from the API reference and the
  knowledge text, and calling one is refused. Use `--primitives full` when the adapter supports a VLA.
- `--guides` gives the agent RPent's three LIBERO operating guides in its workspace (optional reading);
  `--guides-dir <dir>` gives it the `*.md` files of another directory.
- `--vla-endpoint host:port` (and `--sam3-endpoint` on LIBERO-PRO) attach the episode to running model servers
  instead of booting its own; the simulator is always the episode's own.

## A single shot

The strictest setting: one reply, one program, no feedback.

```bash
pyrualean card --suite libero_object_swap --task 2 --seed 0 --out runs/card --cuda-device 0
pyrualean generate --card runs/card/card.json --out runs/gen --model gpt-6-astra
pyrualean run --policy runs/gen/policy.py --suite libero_object_swap --task 2 --seed 0 \
    --out runs/code --cuda-device 0
```

`card` boots only the simulator and saves the task card; `generate` asks the model for `policy.py`
(`--transfer` tells it the program will run on other seeds, so it must not rely on pixels); `run` runs the file
once, with a hard timeout, and a crash or timeout counts as a failed episode.
