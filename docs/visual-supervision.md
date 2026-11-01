# Valen execution supervision

The `cells` arm can ask Valen to inspect the camera images after each motion
primitive. Astra writes the Python and phase-specific visual checks; Valen answers
atomic questions; the host evaluates rules. A review pauses the current cell and
returns evidence to Astra without ending the episode. Astra then combines images,
measurements and action results to choose the next program; an ordinary review
does not require Valen to approve that program. This is an optional, experimental
endpoint monitor, not trajectory collision checking or a physical safety controller.

## Planning skill

Use [Astra–Valen supervision](../skills/astra-valen-supervision/SKILL.md) to prepare
task-specific acceptance criteria, phase contracts, uncertainty recovery, and
completion reports. Its [contract examples](../skills/astra-valen-supervision/references/contracts.md)
cover placement and drawer closure as well as other manipulation relationships.

After Astra explicitly confirms stuck with `finish(status="stuck")`, the configured
host dispatcher supplies [Astra stuck recovery](../skills/astra-stuck-recovery/SKILL.md).
It reassesses causal dependencies, selects alternative actions,
and permits documented deferral of future-phase checks while retaining current
prerequisites. Register deferred facts with `defer_fact`: in pause mode the host
checks them before motion in the named dependent phase. Two unproductive local repairs trigger
a progress review, not automatic task termination.

Enabling either `shadow` or `pause` automatically includes the packaged
[planning skill core](../src/pyrualean/knowledge/astra_valen_supervision.md) in the
actual Astra prompt, together with the task card and monitor API. The core asks
Astra to derive task-specific phases and acceptance conditions. It is frozen as
`supervision_skill.md`; `monitor.planning_skill` in `result.json` records its hash.
The longer repository skill and examples remain reference material; arbitrary
skill folders are not discovered. Verify `prompt.txt` when comparing runs. The
skill gives planning guidance; host checks enforce only the dependencies actually
registered through the API, without changing monitor thresholds.

The `play` CLIs configure the bundled
[recovery core](../src/pyrualean/knowledge/astra_stuck_recovery.md) by default for
`cells`. It is independent of the monitor switch. Override its text or round limit:

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
With `--no-stuck-recovery`, stuck is terminal. The audit is stored in
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

`pyrualean play` and `python -m pyrualean.play` default to `cells`, monitoring
`off`, and bundled stuck recovery. Normal execution does not require a Valen service
or `watch` declarations. Opt in with `--monitor-mode pause` to request cell
interruptions, or `--monitor-mode shadow` to collect reports. The service address is
`http://127.0.0.1:18001/predict`; set `PYRUALEAN_VALEN_URL` or `--valen-url` to
override it (the flag wins). The CLI connects to an existing service rather than
starting model weights. To collect observations without pausing, use:

```bash
--monitor-mode shadow --valen-url http://127.0.0.1:18001/predict
```

When explicitly enabled, `pause` stops the remaining cell actions when review is due.
Use `--monitor-mode off --no-stuck-recovery` to disable both features. Turning off
the monitor alone still leaves the bundled recovery skill configured. Both the
outer `pyrualean play` CLI and `python -m pyrualean.play` forward explicit opt-outs.
Codex, Claude, and OpenAI runtimes share the same supervised tools. The `program`
arm defaults to both features off and rejects explicitly enabling either.

When migrating an existing supervised run, pass `--monitor-mode pause` explicitly
to retain monitoring. Questions no longer become critical implicitly. Use
`critical=[...]` for a one-time uncertainty review, and `requires` / `defer_fact`
only when a fresh matching visual answer must remain a hard precondition across
cells. An existing explicit gate is still enforced; ordinary review does not remove it.

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

The defaults above are CLI defaults. Python callers still explicitly pass
`monitor=MonitorConfig(...)` and/or `stuck_recovery=StuckRecoveryConfig(...)` to
`play_episode`; the XPolicyLab adapter's deployment configuration is unchanged.

## Phase contracts

The generated Astra prompt documents the `watch` helper. A new cell has no active
contract. Before moving, Astra declares checks that apply after **every** primitive
in the current phase:

```python
# Astra has assessed grasp/lift using images, state and object motion.
watch(
    "transport mouse",
    {"held": "Is the computer mouse held between the right gripper fingers?"},
    alerts=[{"name": "lost mouse", "when": {"held": "no"}}],
)
robo.move_to([0.1, -0.1, 0.95], arm="right", gripper=robo.CLOSE)
# A second movement executes only after the first endpoint's monitor result.
```

These coordinates illustrate the interface, not a reusable task policy. Questions
must identify objects and arms from the actual scene. Enter a different phase
before releasing the object: loss of grip is expected during release.

`watch(phase, facts, alerts=[], critical=None, completion=None, requires=None,
invalidates=None)` accepts:

| Field | Meaning |
| --- | --- |
| `phase` | Nonempty name of the current manipulation phase |
| `facts` | 1–12 distinct atomic visual questions, keyed by short IDs |
| `alerts` | Up to eight rules: `name`, `when` mapping fact IDs to `yes`/`no`, optional `confirmations` |
| `critical` | Fact IDs whose sustained uncertainty requests review once per continuous unknown episode; defaults to `[]` |
| `completion` | Optional rule with the same shape; requests Astra review, never calls `finish` |
| `requires` | Explicit hard visual prerequisites: up to 12 known fact IDs mapped to `yes`/`no`; fresh matching answers are checked **before every motion** in this phase |
| `invalidates` | Facts that each attempted motion can change; defaults to all known facts. A narrower list asserts that other facts are unaffected |

All predicates of a rule must match. `confirmations` defaults to one and counts
consecutive distinct source captures on source-aware backends. Unknown matches neither yes
nor no and clears a matching confirmation streak. An already notified alert or
completion condition stays notified through unknown answers: only new explicit
counterevidence rearms it. Each fact's unknown streak survives new cells; re-reading
or re-evaluating a source adds no confirmation.
The per-cell action and separate verification budgets survive phase changes.
Do not routinely use `requires={"held": "yes"}` for transport. Astra normally
assesses the grasp from all permitted evidence. Declare `requires` only when the
specific fresh visual answer must be a hard condition; a normal review cannot
remove that deliberate requirement. Completion rules with one
confirmation provide a candidate, not evidence of sustained stability.

## Evidence across cells and phase prerequisites

On a backend with source-bound snapshots, `verify(observation_id=None)` checks
the active `watch` facts without moving. Obtain a reference with
`view = robo.snapshot()` and call `verify(view["observation_id"])`
before a dependent motion when a `requires` gate needs evidence. Request
`snapshot(refresh=True)` only when new capture evidence is needed. A normal approach
without an unmet prerequisite needs no extra pre-action visual call; the automatic
endpoint monitor still runs. `snapshot()`
without refresh reads an existing record. Unaligned, stale or unverified
sources are refused; legacy backends retain only their post-action path.
Verification has a separate per-cell counter, currently using the same
configured `max_actions` limit. A verification callback cannot issue motion.

An unknown action outcome is a separate blocker. Query `robo.reconcile(action_id)`
in a new cell before establishing visual prerequisites again. Neither a new
image nor a Valen yes can acknowledge the execution of an unresolved action.

`evidence()` returns a detached snapshot with `facts` and `phase_requirements`.
Each fact includes its question/version, latest value and probabilities, observation
ID, source ID, control step, action revision, freshness, and invalidation reason/action ID.
A new cell or `watch()` preserves this ledger and its source-aware counters. Editing the snapshot
cannot establish a fact. Up to 64 facts and 64 phases with prerequisites are kept
per episode. Reusing an unguarded ID for a new question invalidates its old evidence
and increments its version; IDs used by registered gates must retain their question.

By default, **every attempted motion** invalidates all old facts before execution.
Successful Valen responses refresh only the facts they answer. To preserve a fact
through independent work, explicitly narrow `invalidates` and record why the other
facts cannot change. Action/monitor errors invalidate all facts. These declarations
are the planner's responsibility: the host cannot infer all physical dependencies.

For example, defer a known unresolved alignment fact during independent transport:

```python
# aligned was declared in an earlier watch; deferring never marks it true.
defer_fact("aligned", until="insert", expected="yes",
           reason="Transport is independent of final alignment; insertion is not.")
# Later use phase "align" to establish fresh alignment evidence.
# Before every motion in phase "insert", the host checks aligned == yes.
```

`defer_fact` registers a persistent prerequisite for the exact `until` phase name.
It does not remove another gate or change truth. A required fact with `unknown`, a
contradiction, or stale evidence blocks motion in pause mode, even when the last
monitor event was `clear`. The planner must establish it in an earlier observation
or repair phase. Neither `requires={}` nor a new cell removes registered gates.
Requirements continue to apply on later visits to the same phase. Use stable names;
renaming a phase does not justify bypassing its actual dependencies. The host
enforces declared phase gates, not undeclared geometry or final task acceptance.

## Review and recovery

In explicitly enabled `pause` mode, the current cell stops on:

- A newly triggered alert or completion rule.
- A continuous unknown episode reaching the limit for an explicitly critical fact
  (default two endpoints).
- The per-cell action limit (default eight successful primitives).
- Missing phase contract, action error, unavailable images, malformed model
  response, stale/non-increasing step evidence, or a model timeout.
- An unmet registered prerequisite, before the dependent action executes.

`critical` defaults to empty, so ordinary unknown answers are information for Astra.
For an opted-in critical fact, a continuous unknown episode triggers once. The next
cell can continue after Astra has assessed the feedback and current scene; the
ledger retains the original unknown, probabilities and source. A new known source
ends that episode and allows a later unknown recurrence to trigger again. Alerts
and completion likewise notify once while continuously matching, and need new
explicit counterevidence to rearm. This avoids repeated interruption for the same
unresolved visual issue without pretending that it has been resolved.

Explicit `requires` and `defer_fact` gates remain hard conditions across cells.
An unresolved execution receipt, controller constraint or budget remains governed
by its own contract. Neither Astra review nor a Valen yes clears those blockers.
With the default unknown window, one unknown endpoint on a critical fact permits
the next action unless it leaves a registered prerequisite unresolved.
Use `--monitor-unknown-limit 1` when a critical unknown should pause immediately.
The threshold defaults to 0.8; any diffuse prediction becomes unknown. Valen's
candidate probabilities are not calibrated correctness estimates.

`shadow` mode records these conditions and continues. A missing contract is
recorded, but cannot produce meaningful Valen judgments until Astra supplies
questions. Shadow mode is useful for collecting evidence; it does not prevent
bad actions. It still adds synchronous model latency.

The tool's `supervision` field includes the current contract, recent reports,
compact persistent evidence, unmet prerequisites, review reason, and audit location.
Use `evidence()` for full records if tool text is compacted. On pause it also contains the last attempted
action and current robot-state summary. Exact inspected images accompany a pause
even when `images=none` or `on-demand`, and the report is returned even with
`feedback=pure`. Full records remain on disk.

The interruption derives from `BaseException` so ordinary `except Exception`
blocks cannot swallow it. A host latch blocks subsequent movement in that cell,
even if a bare handler catches the interruption. The next tool call resets the
latch and requires a fresh contract. The scene and Python variables persist;
assignments interrupted by supervision may not have completed. Astra must inspect
the report and replan from the current state instead of replaying the old cell.
The new cell is the handoff to Astra, not a claim that Valen's unknown became yes.
Inspect the evidence and state why the next action is justified; seek a useful
observation when necessary. `requires` is the separate mechanism for a deliberately
persistent visual prerequisite.
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
and response. `evidence.json` holds the latest persistent fact/phase snapshot;
`evidence.jsonl` audits declarations, deferrals, invalidations, and new observations.
Prerequisite review images are current host context, not extra Valen votes.
Timings include image preparation and model communication. Valen
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

## Controller feedback and offline evaluation

The [Valen evaluation CLI](guide/valen-evaluation.md) reads saved request/response
pairs directly, exports independent annotation templates, compares thresholds,
and supports explicit question/crop replay without robot actions.

Controller feedback distinguishes `motion_rejected` (target missed with no native
steps), `motion_incomplete` (partial execution), and `no_progress` (no new endpoint)
from `monitor_error` (image/protocol/service failure). Known zero-step IK refusals
can return to bounded generated Python without a new visual vote. Recoverable
`move_both` partial results are checked by Valen before returning; their report
includes `motion_event="motion_incomplete"` and the actual controller outcome.
Visual review rules still interrupt local correction. Unclassified partial results,
critical execution faults and limits require outer review. Review images alone do
not establish fresh facts. Error messages remain in feedback. Normal reports include
`decisions`: raw winning choice, probability, threshold and abstention reason.

## Tests

```bash
python -m pytest tests/test_supervision.py tests/test_arms.py tests/test_play.py tests/test_cli.py -q
```

Fixtures cover interruption and recovery, phase transitions, uncertainty, evidence
windows, malformed results, action errors, and the HTTP protocol. The HTTP test
opens a temporary loopback port; the remaining tests require neither GPU nor a
model service. They establish control-flow behavior, not visual accuracy.
