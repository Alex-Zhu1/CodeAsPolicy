# Valen execution supervision

The `cells` arm can ask Valen to inspect the camera images after each motion
primitive. Astra writes the Python and phase-specific visual checks; Valen answers
atomic questions; the host evaluates rules. A review pauses the current cell and
returns evidence to Astra without ending the episode. This is an experimental
endpoint monitor, not trajectory collision checking or a physical safety controller.

## Planning skill

Use [Astra–Valen supervision](../skills/astra-valen-supervision/SKILL.md) to prepare
task-specific acceptance criteria, phase contracts, uncertainty recovery, and
completion reports. Its [contract examples](../skills/astra-valen-supervision/references/contracts.md)
cover placement and drawer closure as well as other manipulation relationships.

After Astra explicitly confirms stuck with `finish(status="stuck")`, the optional
host dispatcher supplies [Astra stuck recovery](../skills/astra-stuck-recovery/SKILL.md).
It reassesses causal dependencies, selects alternative actions,
and permits documented deferral of future-phase checks while retaining current
prerequisites. Deferred facts must be established before their dependent action;
the host does not enforce these deadlines. Two unproductive local repairs trigger
a progress review, not automatic task termination.

For a planner with repository access, request: "Read
`skills/astra-valen-supervision/SKILL.md` and apply it to this task." In a separate
episode runtime, include the supervision instructions and resulting task brief
in the actual Astra prompt, and verify the saved `prompt.txt`. The current CLI
does not discover or inject this folder automatically. The skill is planning
guidance; it does not add host-enforced phase gates or change monitor thresholds.

Enable confirmed-stuck recovery explicitly for `--arm cells`:

```bash
--stuck-recovery-skill skills/astra-stuck-recovery/SKILL.md --stuck-recovery-rounds 2
```

The initial prompt describes the trigger protocol. The skill body is delivered
only after `finish(status="stuck", summary="...")`, along with current state and
available camera images. `episode_finished=false` means the confirmed blockage
was recorded and Astra should apply recovery in the same episode. No motion is
replayed; variables, scene, monitor rules, and budgets are preserved. A regular
Valen pause does not dispatch this skill. The OpenAI runtime also defers later
calls in the same batch until the planner has read this response.

Recovery dispatches are bounded (two by default). Terminal environments,
exhausted episode budgets, or exhausted recovery rounds finalize the stuck
request with `episode_finished=true`. Success/failure requests remain terminal.
Without the option, stuck remains terminal as before. The audit is stored in
`stuck_recovery.jsonl` and `result.json`; the frozen skill is saved alongside it.
An ended episode cannot be reopened by loading a skill. Archived run prompts
remain unchanged.

## Run

Start the existing GPT-as-Jev Valen service in its model environment, choosing an
available GPU and port. This repository does not install or load model weights:

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/valen-env/bin/python \
    /path/to/GPT-as-Jev/hybrid_rollout/libero/valen_server.py \
    --model /path/to/Valen-2B --port 18001
```

Add these options to an existing `pyrualean play --arm cells ...` command:

```bash
--monitor-mode shadow --valen-url http://127.0.0.1:18001/predict
```

Use `--monitor-mode pause` to stop the remaining cell actions when a review is
due. `off` is the default and preserves the original execution path. Both the
outer `pyrualean play` CLI and `python -m pyrualean.play` accept the options.
Codex, Claude, and OpenAI runtimes share the same supervised tools. The `program`
arm is rejected when monitoring is enabled.

The service must be able to read the run's image paths on the same filesystem.
The client uses the existing `/predict` protocol: `state.messages` contains the
images and their observation IDs, and `questions` contains choice questions with
`yes`, `no`, and `unknown` candidates. It does not send simulator object truth,
benchmark success, planned numerical actions, or alert rules to Valen. Requests
are synchronous, without automatic retries or redirects. HTTP(S) endpoints are
supported; environment proxy settings are ignored for this model connection.

For the XPolicyLab adapter, add a `monitor` mapping to `deploy.yml`:

```yaml
monitor:
  mode: shadow
  url: http://127.0.0.1:18001/predict
  timeout_s: 30
  threshold: 0.8
  unknown_limit: 2
  max_actions: 8
```

Python callers can pass `monitor=MonitorConfig(...)` to `play_episode`.

## Phase contracts

The generated Astra prompt documents the `watch` helper. A new cell has no active
contract. Before moving, Astra declares checks that apply after **every** primitive
in the current phase:

```python
# This phase assumes grasp and lift have already been verified.
watch(
    "transport mouse",
    {"held": "Is the computer mouse held between the right gripper fingers?"},
    alerts=[{"name": "lost mouse", "when": {"held": "no"}}],
    critical=["held"],
)
robo.move_to([0.1, -0.1, 0.95], arm="right", gripper=robo.CLOSE)
# A second movement executes only after the first endpoint's monitor result.
```

These coordinates illustrate the interface, not a reusable task policy. Questions
must identify objects and arms from the actual scene. Enter a different phase
before releasing the object: loss of grip is expected during release.

`watch(phase, facts, alerts=[], critical=None, completion=None)` accepts:

| Field | Meaning |
| --- | --- |
| `phase` | Nonempty name of the current manipulation phase |
| `facts` | 1–12 distinct atomic visual questions, keyed by short IDs |
| `alerts` | Up to eight rules: `name`, `when` mapping fact IDs to `yes`/`no`, optional `confirmations` |
| `critical` | Fact IDs whose sustained uncertainty requests review; defaults to all facts |
| `completion` | Optional rule with the same shape; requests Astra review, never calls `finish` |

All predicates of a rule must match. `confirmations` defaults to one and counts
consecutive distinct executed primitive endpoints. Unknown matches neither yes
nor no and clears a matching streak. Phase changes and new cells clear streaks
and the previous image reference. The per-cell action budget survives phase changes.
Do not repeatedly redeclare the same phase if it needs temporal confirmation.
There is no automatic “previously held” arming: Astra must declare the transport
phase only after the prerequisite was checked. Completion rules with one
confirmation provide a candidate, not evidence of sustained stability.

## Review and recovery

In `pause` mode, the current cell stops on:

- A matching alert or completion rule.
- Consecutive unknown results for a critical fact (default two endpoints).
- The per-cell action limit (default eight successful primitives).
- Missing phase contract, action error, unavailable images, malformed model
  response, stale/non-increasing step evidence, or a model timeout.

With the default unknown window, one unknown endpoint permits the next action.
Use `--monitor-unknown-limit 1` when a critical unknown should pause immediately.
The threshold defaults to 0.8; any diffuse prediction becomes unknown. Valen's
candidate probabilities are not calibrated correctness estimates.

`shadow` mode records these conditions and continues. A missing contract is
recorded, but cannot produce meaningful Valen judgments until Astra supplies
questions. Shadow mode is useful for collecting evidence; it does not prevent
bad actions. It still adds synchronous model latency.

The tool's `supervision` field includes the current contract, recent reports,
review reason, and audit location. On pause it also contains the last attempted
action and current robot-state summary. Exact inspected images accompany a pause
even when `images=none` or `on-demand`, and the report is returned even with
`feedback=pure`. Full records remain on disk.

The interruption derives from `BaseException` so ordinary `except Exception`
blocks cannot swallow it. A host latch blocks subsequent movement in that cell,
even if a bare handler catches the interruption. The next tool call resets the
latch and requires a fresh contract. The scene and Python variables persist;
assignments interrupted by supervision may not have completed. Astra must inspect
the report and replan from the current state instead of replaying the old cell.
This cooperates with the existing policy sandbox; it is not an adversarial-code
security boundary.

A long `move_to`, `home`, or learned primitive can execute many control steps
before its endpoint is checked. Mid-primitive interruption requires separate
integration into the backend's control loop. There is no background monitor
running while Astra thinks, and no simulator steps are advanced to obtain more
votes from an unchanged scene. Read-only calls do not trigger a check.

## Evidence and offline replay

`result.json` records configuration under `monitor`. The `supervision/` directory
contains append-only `events.jsonl`, immutable camera snapshots, and each request
and response. Timings include image preparation and model communication. Valen
receives the current endpoint and at most one previous endpoint from the same
phase, using all declared robot cameras (six images maximum for RoboDojo).

To inspect recorded evidence without starting a simulator, prepare a JSON manifest
with consecutive endpoints **within one phase**. Steps are actual executed control
counts, not image-directory indices or wall-clock time:

```json
[
  {"step": 21, "images": {"cam_head": "/absolute/path/first-head.png"}},
  {"step": 26, "images": {"cam_head": "/absolute/path/second-head.png"}}
]
```

Include the wrist cameras too when available. A contract file contains the
arguments to `watch`, for example:

```json
{
  "phase": "transport mouse",
  "facts": {"held": "Is the mouse held between the right gripper fingers?"},
  "alerts": [{"name": "lost mouse", "when": {"held": "no"}}]
}
```

```bash
python examples/replay_supervision.py --manifest /path/to/manifest.json \
    --contract /path/to/contract.json --start-step 0 \
    --valen-url http://127.0.0.1:18001/predict --out runs/valen-replay
```

Replay always uses shadow mode and writes `replay.json`; it never issues robot
actions. It validates interface compatibility and yields judgments for manual
labeling, not a measured failure-detection accuracy or improved task success.
Evaluate false alerts, missed failures, unknown rate, and latency on labeled
episodes before drawing conclusions about the model's usefulness.

## Tests

```bash
python -m pytest tests/test_supervision.py tests/test_arms.py tests/test_play.py tests/test_cli.py -q
```

Fixtures cover interruption and recovery, phase transitions, uncertainty, evidence
windows, malformed results, action errors, and the HTTP protocol. The HTTP test
opens a temporary loopback port; the remaining tests require neither GPU nor a
model service. They establish control-flow behavior, not visual accuracy.
