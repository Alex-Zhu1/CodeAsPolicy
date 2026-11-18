# Hierarchical Code + PEAR Visual-Delta

CodeAsPolicy is good at global reasoning: finding and remembering objects, staged grasps,
collision-free transport. It is weak in the last 5 cm, where open-loop triangulation leaves
3-8 mm residuals and contact tasks (`insert_key`, `plug_in_charger`, `hang_mugs`,
`press_by_number`, `pour_liquid_into_cup`) fail. The hierarchy splits the work:

```mermaid
flowchart LR
    LLM["Code planner (LLM writes robo.* Python)"] -->|global moves, grasps, transport| Robo["robo API / RoboDojo host"]
    LLM -->|"last mile: robo.visual_align_and_insert(goal)"| Loop["closed-loop sub-skill"]
    Loop -->|"views + state JSON"| VLM["PEAR Visual-Delta policy (fine-tuned Qwen VLM or callable)"]
    VLM -->|"JSON delta"| Nudge["robo.nudge: compact guarded delta executor"]
    Nudge -->|"reached / stopped / out_of_reach"| Loop
    Runs["runs/ episodes, CodeAsPolicy_RSI rollouts"] -->|visual_delta_distill| Data["PEAR SFT / RL JSONL"]
    Data -->|fine-tune| VLM
```

| Layer | Module | Owner |
| --- | --- | --- |
| Protocol parser, prompts, policy client | `src/codeaspolicy/visual_delta.py` | delta-skill |
| `nudge`, `visual_align_and_insert` on RoboDojo | `src/codeaspolicy/hosts/robodojo_visual_delta.py` | delta-skill |
| Distillation + live recording | `src/codeaspolicy/visual_delta_distill.py` | rsi-distill |

## PEAR Visual-Delta protocol v1

Each turn the policy sees a **state JSON** and labelled images, and answers with optional
`<think>...</think>` and one fenced `json` block.

```json
{"arm": {"x": 0.4, "y": -0.2, "z": -1.0, "roll": 0, "pitch": 0, "yaw": 0.05}, "gripper": {"width": 0}}
```

Bimanual mode uses `left_arm`, `right_arm`, `left_gripper` and `right_gripper` as keys. A
top-level `"done": true` ends the sub-skill.

| Field | Unit | Convention |
| --- | --- | --- |
| `x, y, z` | cm | TCP grasp point (link6 + `TCP_OFFSET` = 0.145 m along link6 +x), `next - current`. `frame="world"`: +x robot's right, +y away from the robot, +z up. `frame="tcp"`: the world delta in current link6 axes (+x pointing, +y finger closing, +z normal). |
| `roll, pitch, yaw` | rad | Wrapped difference of `robodojo_kinematics.angles()` (pitch, yaw, roll in degrees). Executor: `target = current + delta` through `kin.orientation(pitch, yaw, roll)`. |
| action `gripper.width` | switch | `+1` open, `-1` close, `0` keep. RoboDojo: `OPEN=-1`, `CLOSE=1`; `ee_joint_state` command `1.0` open, `0.0` closed. |
| state `gripper.width` | cm | Jaw span `round((0.0023 + 0.1091 * opening) * 100, 1)`. |
| state `x, y, z` / angles | cm / rad | Rounded to 1 / 2 decimals. |

> [!IMPORTANT]
> The sign of the gripper switch is opposite to RoboDojo's internal code. Always convert
> through `pear_gripper_to_robodojo()` / `robodojo_gripper_to_pear()` /
> `ee_command_to_pear()` (exported by both modules), never by hand.

## Online layer: `robo.nudge` and `robo.visual_align_and_insert`

- `robo.nudge(arm, x=..., z=..., frame="world"|"tcp", stop_on_stall=True)` moves by a bounded
  relative delta (default at most 6 cm and 20 degrees). Compact mode steps straight along IK
  waypoints with no rest-to-rest spline, so a 1 cm move costs a few native steps. Contact or
  stall ends it cleanly with `status="stopped"`, `stalled=True`,
  `stop_reason="contact_or_stall"`. A table-colliding or unreachable target uses 0 steps.
- `robo.visual_align_and_insert(goal, arm="right", frame="world", max_rounds=8,
  max_steps=40, reserve_steps=20, crop_wrist=True, policy=None)` runs the closed loop: capture
  views and state, ask the policy, run the delta through `nudge`, and report the outcome in the
  next turn. It stops on `done`, convergence, a completed contact, or the budget.
- The policy is an OpenAI-compatible server (`CODEASPOLICY_DELTA_BASE_URL`,
  `CODEASPOLICY_DELTA_MODEL`, `CODEASPOLICY_DELTA_API_KEY_ENV` / `OPENAI_API_KEY`). A Python
  callable can stand in for tests and local policies.

Typical planner use: transport the key to about 2 cm above the lock with ordinary motion,
then `robo.visual_align_and_insert("insert the key into the lock slot", arm="right")`.

## Offline layer: distilling training data

`visual_delta_distill` turns recorded episodes into the exact turn structure the online loop
uses:

- **system**: the JSON delta rules, `## ENVIRONMENT` (axes and cameras for the frame), and
  `## TASK`. Last-mile segments add a sub-goal line.
- **user**: `## LAST ACTION` (`none`, `reached`, or `stopped (reason)`), `## STATE` JSON, and
  `## IMAGES` as `[cam_head] <image>`, `[cam_right_wrist] <image>`, and optionally
  `[cam_right_wrist_crop] <image>`. The record's `images` list holds the paths in token order.
- **assistant**: a templated `<think>` rationale and the reference action JSON. Successful
  segments that reach the end of the episode get a final zero-delta `"done": true` turn.

### Macro-step coalescing

Native steps are far too fine for a VLM turn. `coalesce()` merges consecutive recorded states
into PEAR "move until at rest" deltas. A macro step closes:

- at a contact or stall stop (the next turn then reports `stopped`);
- at a gripper command change, so each turn has at most one switch;
- before a phase change (`call.kwargs.phase`, else the tool name);
- before it would exceed `max_linear_m` (0.06) or `max_angular_deg` (20), the same caps as
  `nudge`, so every label can be executed by one `nudge`.

Steps with no motion (under 1 mm and 0.5 degrees), no switch and no stop are dropped as hold
noise.

### Episode directories

`load_episode()` reads `result.json` (`environment_success`, `env_steps`), `card.json`
(`task_language`), `calls.jsonl` (tool, kwargs, summary, `env_steps`, `action_id`) and
`steps/NNNN/observation.json` with the camera PNGs. Each recorded state is attached to the call
that produced it, by `action_id` or else by cumulative native steps. That call supplies the
phase, the stop (`summary.stalled` / `status == "stopped"` / `stop_reason`) and the gripper
command. Poses come from `<arm>_ee_pose`, or from forward kinematics of the joints when numpy
is available.

```bash
# full episodes, SFT conversations
python -m codeaspolicy.visual_delta_distill extract runs/ -o data/sft.jsonl --success-only

# last-mile contact segments only, wrist ROI crops, one RL sample per turn
python -m codeaspolicy.visual_delta_distill extract runs/ -o data/rl.jsonl \
    --last-mile --crop --format rl --frame tcp

# macro chunks vs native steps, delta distributions, contact-stop ratio
python -m codeaspolicy.visual_delta_distill stats data/sft.jsonl
python -m codeaspolicy.visual_delta_distill stats runs/
```

`--last-mile` keeps runs of transitions in `CONTACT_PHASES` (`align`, `contact`, `nudge`,
`visual_align_and_insert`, `grasp_step`, `probe`, `insert`, `press`, `close`,
`close_gripper`, or a custom `--phases` list). `--radius-m R` keeps transitions that end within
R of the target, which is the episode's final TCP position unless one is given.

### Live recording in CodeAsPolicy_RSI rollouts

```python
from codeaspolicy.visual_delta_distill import ExtractOptions, VisualDeltaEnvRecorder

env = VisualDeltaEnvRecorder(env, "runs/visual_delta", task=instruction,
                             options=ExtractOptions(last_mile=True, crop=True))
with env.phase("insert"):
    ...                       # env.step / env.get_obs as usual
env.mark_stop()               # when an executor detects contact or a stall
env.on_trial_end({"environment_success": solved})   # appends to visual_delta.jsonl
```

The wrapper delegates everything else to the env. It records one frame per observation read
after an action, with gripper levels taken from `<arm>_ee_joint_state` in the actions, and
writes `trial_NNNN/steps/NNNN/observation.json` plus PNGs. That lets `extract` re-process the
trials later with other options.

## Training recipe (suggested)

1. Collect successful `CodeAsPolicy_RSI` rollouts with the recorder, together with the offline
   `runs/`.
2. SFT a small Qwen-VL model on `--last-mile --crop` conversations. The `<image>` tokens and
   the `images` list match LLaMA-Factory and ms-swift multimodal formats.
3. RL with `--format rl` samples. Reward from JSON validity, sign agreement and distance to
   `solution`, then from in-simulator success through `visual_align_and_insert`.
4. Serve it behind an OpenAI-compatible endpoint and point `CODEASPOLICY_DELTA_BASE_URL` at it.

Tests: `tests/test_visual_delta_distill.py`. They are pure Python, and the check against host
kinematics runs when numpy is present.
