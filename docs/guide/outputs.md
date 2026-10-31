# Output Files

Every episode writes one run directory (`--out`).

| File | What it holds |
|---|---|
| `result.json` | The verdict (`environment_success`), the arm, how the agent finished, token usage and LLM calls, and the settings of the run. |
| `episode.mp4` | The episode video (RPent robots; on RoboDojo the run directory keeps each step's images under `steps/`, and RoboDojo records the videos). |
| `workspace/` | Every cell or program the agent ran, and the files it wrote. |
| `tool_turns.json` | Each tool call: tool, timing, execution `ok`, motion, simulator steps, sandbox flags, and structured `failure` / `recovery` reports when generated code fails. |
| `calls.jsonl` | Every primitive call: arguments, outcome, wall time and simulator steps. |
| `prompt.txt`, `card.json`, `card/*.png` | The prompt and the task card the model received. |
| `codex_events.jsonl` / `claude_events.jsonl` / `openai_events.jsonl` | The runtime's own record of the conversation (with `--runtime openai`, also `messages.json`). |
| `states.json`, per-step images | The robot stack's own records (RPent robots). |
| `policy.py`, `policy_output.txt` | The program and what it printed (single shot). |
| `host.log` | The log of the robot host that ran the episode: where to look when `play` reports that it failed. |

On RoboDojo, RoboDojo's evaluation client keeps its own records as well ([RoboDojo](../robots/robodojo.md#view-results)).

## Tables

```bash
pyrualean summarize runs/demo runs/other           # one row per episode
pyrualean summarize --json runs/demo runs/other    # the same rows, then the totals as JSON: success rate, statuses, mean calls and tokens
```

Give `summarize` run directories, not a folder that also holds other things.

A crash, a timeout or an episode that did not boot counts as a failed episode.
