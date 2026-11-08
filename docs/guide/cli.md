# Command-Line Reference

## Commands

| Command | What it does |
|---|---|
| `pyrualean api` | Print the API reference the agent reads (LIBERO-PRO's robot). `--primitives no-vla` leaves the VLA out. |
| `pyrualean prompt` | Print the prompt of a single shot (`generate`); `--card` renders it for a task card. A `play` episode saves the prompt its model read as `prompt.txt` in the run directory. |
| `pyrualean play` | Run one episode with an agent ([Running Episodes](running.md)). |
| `pyrualean card` | Boot only the simulator and save a task card. |
| `pyrualean generate` | Ask the model for a whole program from a task card (single shot). |
| `pyrualean run` | Run a program file for one episode. |
| `pyrualean summarize` | Tabulate run directories ([Output Files](outputs.md)). |
| `pyrualean valen-eval` | Compare saved visual checks, export human labels, or replay question/crop variants without a robot ([Valen Evaluation](valen-evaluation.md)). |

## The task instance

`full` exposes the primitives supported by the chosen adapter. Official
`robodojo_xpl` and `robodojo_xpl_rgb` have no VLA service and automatically use
the effective `no-vla` set in both prompts and execution. An explicit `full`
cannot enable missing capabilities; `result.json` records the effective set.

`play`, `card` and `run` take the same options:

| Option | Meaning (default) |
|---|---|
| `--robot` | `libero`, `robotwin` or `robocasa` (`libero`). `play` also takes `robodojo_xpl_rgb`, which starts RoboDojo's own client itself (it needs `ROBODOJO_SOURCE_ROOT` and one environment with both sides); RoboDojo's official evaluation runs through XPolicyLab ([RoboDojo](../robots/robodojo.md)) |
| `--suite` | LIBERO suite, RoboTwin task config or RoboCasa split |
| `--task` | LIBERO task index, or the robot's task name |
| `--seed` | the scene seed (`0`; always pass it, since the robots' own defaults are not used) |
| `--out` | the run directory |
| `--cuda-device` | the GPU the episode's servers use |
| `--max-episode-steps` | the simulator step limit (`10000`) |
| `--rpent-root`, `--python` | the RPent checkout and the interpreter it is installed in |

## `play`

| Option | Meaning (default) |
|---|---|
| `--arm` | `cells` (default) or `program` |
| `--runtime` | `codex`, `claude` or `openai` (`codex`; [Agent Configuration](agents.md)) |
| `--model`, `--reasoning` | the model (`gpt-6-astra`) and its reasoning effort (`high`) |
| `--base-url`, `--api-key-env` | an OpenAI-compatible endpoint, and the variable holding its key |
| `--max-turns` | the decision budget (`100` code calls); `0` removes the call-count limit |
| `--budget-s` | the wall-clock budget, in seconds (`5400`) |
| `--code-timeout-s` | execution deadline per cell/program, including loading (`120` seconds) |
| `--code-max-actions` | optional maximum motion attempts per code call |
| `--code-output-chars` | maximum retained characters per stdout/stderr stream (`8000`) |
| `--monitor-mode` | `pause`, `shadow` or `off` (default) |
| `--valen-url` | existing Valen service (`$PYRUALEAN_VALEN_URL` or `http://127.0.0.1:18001/predict`) |
| `--monitor-timeout-s`, `--monitor-threshold` | visual request budget (`30` seconds) and choice threshold (`0.8`) |
| `--monitor-unknown-limit`, `--monitor-max-actions` | consecutive critical unknowns (`2`) and per-cell successful actions before review (`8`) |
| `--stuck-recovery` | enable the bundled recovery core for cells; recovery is off by default |
| `--stuck-recovery-skill` | enable recovery with a custom skill file; delivered only after confirmed stuck |
| `--stuck-recovery-rounds` | maximum recovery dispatches within the episode (`2`); setting this alone does not enable recovery |
| `--no-stuck-recovery` | explicitly disable stuck recovery; cannot be combined with either enabling flag |
| `--images` | `on-motion`, `on-demand` or `none` (`on-demand`) |
| `--feedback` | `rich` or `pure` (`pure`) |
| `--primitives` | `full` or `no-vla` (`no-vla`) |
| `--max-programs` | the most programs on the `program` arm |
| `--guides`, `--guides-dir` | operating guides in the agent's workspace |
| `--vla-endpoint`, `--sam3-endpoint` | running model servers to attach to |
| `--codex-home`, `--codex-bin`, `--claude-cli` | the runtimes' homes and executables |

Both `pyrualean play` and `python -m pyrualean.play` use the defaults above;
`play_episode` uses the same launch settings when its optional arguments are omitted.
Executable [action contracts](action-contracts.md) are active in the execution loop:
public task invariants require check bindings before motion. Stage prerequisites,
completion checks, geometric constraints and multi-stage previews depend on the
planner's declarations.

Valen monitoring and confirmed-stuck recovery are independently off by default.
Enable monitoring with `--monitor-mode pause` or `shadow`, and recovery with
`--stuck-recovery` or a custom `--stuck-recovery-skill`. Only enabled monitoring
requires a running Valen service sharing the image filesystem. Its skill core is
included in the initial prompt; the recovery body is delivered only after
`finish(status="stuck")` when the host permits continuation. `program` rejects
enabling either feature. Python callers opt in with `MonitorConfig` and
`StuckRecoveryConfig`; see [visual supervision](../visual-supervision.md).

These defaults apply to `play`. `generate`, `prompt`, `run` and XPolicyLab's
deployment configuration retain their own settings. Robot, task, seed, GPU and
native benchmark action budgets are selected separately.

## Environment variables

| Variable | Meaning |
|---|---|
| `RPENT_ROOT` | the RPent checkout, when RPent is not installed as a package |
| `PYRUALEAN_RUNTIME` | default of `--runtime` |
| `PYRUALEAN_PRIMITIVES` | default of `--primitives` |
| `PYRUALEAN_CODEX_BIN`, `PYRUALEAN_CODEX_HOME` | defaults of `--codex-bin`, `--codex-home` |
| `PYRUALEAN_CLAUDE_CLI` | default of `--claude-cli` |
| `PYRUALEAN_OPENAI_EXTRA_BODY` | default of `--openai-extra-body` |
| `PYRUALEAN_VALEN_URL` | default Valen `/predict` address; explicit `--valen-url` takes precedence |
| `PYRUALEAN_CODEX_RETRIES` | retries of a request from Codex (`play`, `generate`) to a `--base-url` endpoint (`10`) |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | the endpoint and key of `--runtime openai` when `--base-url` / `--api-key-env` are not given |
| `PYRUALEAN_OPENAI_RETRIES`, `PYRUALEAN_OPENAI_TIMEOUT_S` | retries (`10`) and request timeout in seconds (`900`) of `--runtime openai` |
| `CODEX_AUTO_COMPACT_TOKEN_LIMIT` | Codex's compaction threshold for the episode ([Agent Configuration](agents.md#codex)) |

Environment variables supply CLI defaults. Explicit `--runtime` and `--primitives`
take precedence, including `--runtime codex` and `--primitives full`; the outer
CLI forwards the selected values to the host process.

Each robot page lists the variables of its stack: checkpoints, assets and rendering.

## Development

```bash
pip install -e '.[dev]'
pytest          # offline tests against scripted backends
ruff check .
```
