# Output Files

Every episode writes one run directory (`--out`).

| File | What it holds |
|---|---|
| `result.json` | The verdict (`environment_success`), the arm, how the agent finished, token usage and LLM calls, and the settings of the run. |
| `episode.mp4` | The episode video (RPent robots; on RoboDojo the run directory keeps each step's images under `steps/`, and RoboDojo records the videos). |
| `workspace/` | Every cell or program the agent ran, and the files it wrote. |
| `tool_turns.json` | Each tool call: tool, timing, execution `ok`, motion, simulator steps, sandbox flags, observation IDs/action revisions/image provenance, and structured `failure` / `recovery` reports when generated code fails. |
| `calls.jsonl` | Every primitive call: arguments, outcome, wall time and simulator steps. |
| `interaction_events.jsonl` | Host interaction timings and local call IDs: queueing, complete-cell validation, execution, feedback preparation, and available direct-OpenAI request/retry timings. |
| `prompt.txt`, `card.json`, `card/*.png` | The prompt and the task card the model received. |
| `codex_events.jsonl` / `claude_events.jsonl` / `openai_events.jsonl` | The runtime's own record of the conversation (with `--runtime openai`, also `messages.json`). |
| `states.json`, per-step images | The robot stack's own records (RPent robots). |
| `policy.py`, `policy_output.txt` | The program and what it printed (single shot). |
| `host.log` | The log of the robot host that ran the episode: where to look when `play` reports that it failed. |

On RoboDojo, RoboDojo's evaluation client keeps its own records as well ([RoboDojo](../robots/robodojo.md#view-results)).

## Model-facing feedback

The model receives a compact view of the host records. Normal replies retain
program output, action IDs/results and observation/image sources; repeated action
lists and the default `check_prerequisites` wording are omitted. An omitted normal
handoff is not an action permit. Pending actions and stop conditions keep their
explicit restrictions, and the runtime still enforces them.

Failure feedback summarizes `recovery.last_motion`, including its outcome and
receipt, rather than repeating every native receipt. Serialized text is bounded
to 9,000 characters with structured omission markers when needed. Query IDs are
never shortened into unusable IDs; partial/unknown outcomes and execution controls
take priority over optional detail. Complete results, receipts and provenance
remain in the host records. Inspect actual runtime-delivered output when auditing
client-side truncation; a host record is not proof of what reached the model.

Default perception repr shows task geometry, limitations and a short source
reference. Full calibration, measurement assumptions and image provenance remain
available through the result's `provenance`; printing it does not refresh a source.

## Interaction timing

`play` writes `interaction_events.jsonl` alongside the runtime's existing logs.
`result.json.interaction` records its schema, session, clock domain, write-error count,
supported timing capabilities, and selected runtime configuration. The configuration distinguishes
requested settings from those submitted by the runtime; it does not claim provider confirmation.
These settings describe the runtime's parameter mapping; request/attempt events show whether it
was actually reached (for example, an episode can fail during host startup).
For the direct OpenAI runtime, the recorded reasoning setting comes from `openai_options.extra_body`,
which can differ from the general `--reasoning` argument.

The sidecar contains timing metadata, not generated code, images, prompts or credentials.
Trace-write failures are counted without interrupting robot execution. `local_call_id` links the
sidecar with `tool_turns.json`; returned observation IDs link a completed tool to its feedback.
MCP invocation IDs and provider tool-call IDs remain distinct.

Cells record `validation_started`, `code_validated`, `execution_started`, `execution_finished`
and `tool_result_ready`. `code_validated` means the existing sandbox checks and complete-cell
compilation passed; it does not claim the proposed decision-deadline and evidence-admission contract
is implemented. `feedback_ready` means the transport payload is prepared, not that the model received it.
Program mode currently has outer tool timings, without the cells-only validation/execution breakdown.
Codex and Claude supply host/MCP timings; provider request and first-token timestamps remain unavailable
through these hooks. Direct OpenAI additionally records HTTP attempts and retry waits; it does not
record first-token timing either.

After installing the package, summarize a trace without calling a model or robot:

```bash
python -m pyrualean.evaluation.latency runs/demo/interaction_events.jsonl --format interaction --output /tmp/interaction-summary.json
python -m pyrualean.evaluation.latency /path/to/rollout.jsonl --format codex-rollout --output /tmp/rollout-summary.json
```

Summaries preserve missing or ambiguous intervals and report sample counts with distributions.
Per-call missing-stage reasons distinguish uninstrumented program stages from execution stages
not reached after an explicit validation or setup failure; unexplained absence remains unknown.
Codex rollout intervals are explicitly labeled as client-log proxies; they are not TTFT or pure
reasoning time. Only compatible timestamps within the same clock domain are subtracted.
The original logs and historical `result.json` are not rewritten by analysis.
`--output` must name a new file; omit it to print the report to standard output.

## Tables

```bash
pyrualean summarize runs/demo runs/other           # one row per episode
pyrualean summarize --json runs/demo runs/other    # the same rows, then the totals as JSON: success rate, statuses, mean calls and tokens
```

Give `summarize` run directories, not a folder that also holds other things.

A crash, a timeout or an episode that did not boot counts as a failed episode.
