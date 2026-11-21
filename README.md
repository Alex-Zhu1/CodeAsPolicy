# CodeAsPolicy: Code Policies for AI Robot Agents

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a> &ensp;
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python"></a> &ensp;
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Version-0.2.0-orange.svg" alt="Version"></a>
</p>

**CodeAsPolicy** brings the **Computer-Use** paradigm to physical and simulated robotics.

Instead of treating robot manipulation as fine-grained, single-step tool calling (which introduces heavy latency, high token costs, and fragile error handling), CodeAsPolicy turns a robot into a programmable, interactive Python environment. An AI agent writes structured Python code cells against a typed `robo` object—composing perception, kinematics checks, multi-step motions, and bounded local corrections within a single decision turn.

---

## Key Features

- **Robot as a Typed Python Object (`robo`)**  
  Prompts are generated directly from the live robot class so documentation cannot drift from runtime primitives—spanning single-arm servos (`robo.segment()`, `robo.move_to()`, `robo.release()`, `robo.pi0_pick()`) and dual-arm Coarse + Delta manipulation (`robo.triangulate()`, `robo.ik_both()`, `robo.execute_motion()`, `robo.delta_grasp()`, `robo.align_feature()`, `robo.nudge()`).

- **Closed-Loop Execution & Provenance**  
  - **Snapshot-Bound Perception**: Camera calibration, 3D rays/triangulation, and visual object tracking (`observe_object` / `observe_hold`) bind to explicit `observation_id` captures; motion invalidates stale coordinates.
  - **Coarse Transport + Guarded Fine Delta**: C2 timed joint splines (`execute_motion`) handle collision-checked transport to pre-action hovers, while `delta_grasp`, `align_feature`, and `nudge` handle close-range contact, stall detection, and post-release withdrawal (`home`).
  - **Structured Failure & Stuck Recovery**: Runtime exceptions and partial motions return current robot state, action receipts, and remaining budgets without discarding prior cell state.

- **Optional Visual Supervision & Public Acceptance Contracts**  
  - Configurable monitor modes (`off` by default; `shadow` and `pause` available for offline/online Valen visual checks; see [`docs/design/valen-role-and-value.md`](docs/design/valen-role-and-value.md)).
  - Public task-acceptance schemas (`task_acceptance.py`) keep benchmark instructions and completion rules explicit without leaking privileged simulator verdicts.

- **Multi-Robot & Runtime Ecosystem**  
  - **RoboDojo** (dual ARX-X5 via XPolicyLab, RGB-only or RGB-D, zero external VLA/SAM3 required under `--primitives no-sam`)
  - **LIBERO-PRO**, **RoboTwin 2.0**, and **RoboCasa365** (via RPent)
  - Agent runtimes: **Codex CLI** (`codex`), **Claude Code** (`claude`), and an in-process **OpenAI-compatible** loop (`openai`).

---

## Quick Start

### Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/Alex-Zhu1/CodeAsPolicy.git
cd CodeAsPolicy
pip install -e .
```

Inspect the live generated robot API reference offline (no simulator needed):

```bash
# Default LIBERO-PRO API
codeaspolicy api

# RoboDojo dual-arm Coarse + Delta API (RGB geometry, no SAM3/VLA services)
codeaspolicy api --robot robodojo_xpl_rgb --primitives no-sam
```

---

### How It Works: Code-as-Action

Rather than issuing one atomic tool call per step, the agent executes Python cells in a persistent workspace namespace where `robo`, `np`, and `math` live:

#### Single-Arm Example (`LiberoRobot`)

```python
# 1. Segment target and destination in the current view
mug = robo.segment("red mug")
plate = robo.segment("plate")

if mug.found and plate.found:
    # 2. Pick with VLA or scripted servo, then verify finger separation locally
    pick = robo.pi0_pick("pick up the red mug")
    if not pick.success or robo.state().gripper_opening < 0.01:
        mx, my, mz = mug.world_xyz
        robo.move_to([mx, my, mz + 0.10], gripper=-1.0)
        robo.move_to([mx, my, mz], gripper=-1.0, step_clip=0.015)
        robo.set_gripper(+1.0)
        robo.move_to([mx, my, mz + 0.12], gripper=+1.0)

    # 3. Carry above destination, lower, and release if still holding
    if robo.state().gripper_opening >= 0.01 and not robo.done:
        px, py, pz = plate.world_xyz
        reached = robo.move_to([px, py, pz + 0.12], gripper=+1.0).reached
        if reached and not robo.done:
            robo.move_to([px, py, pz + 0.04], gripper=+1.0, step_clip=0.012)
            robo.release()
```

#### Dual-Arm Coarse + Delta Example (`RobodojoOfficialColourRobot`)

```python
# 1. Capture a synchronized 3-view snapshot and triangulate 3D target coordinates
view = robo.snapshot()
obs_id = view["observation_id"]
pt = robo.triangulate("cam_head", (284, 410), "cam_right_wrist", (236, 318), observation_id=obs_id)
robo.require_current(pt)

# 2. Build pre-grasp hover pose, preflight IK, and run C2 coarse transport
hover = robo.pose_from_axes(
    (pt.xyz[0], pt.xyz[1], pt.xyz[2] + 0.035),
    point_dir=(0.0, 0.0, -1.0),
    close_dir=(1.0, 0.0, 0.0),
)
if robo.ik("right", hover, jaw_symmetry=True).ok:
    moved = robo.execute_motion("stage-right", right=hover, holding={"right": "empty"})
    if moved["reached"]:
        # 3. Close-range guarded pickup, then clean up and return both arms home
        grasp = robo.delta_grasp(arm="right", target_xyz=pt.xyz, contact_table=True, lift_m=0.06)
        if not grasp.holding:
            robo.show("cam_right_wrist")
```

---

### Running an Episode

Run an interactive `cells` episode on LIBERO-PRO (requires an [RPent](https://github.com/RLinf/RPent) environment; see [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md)):

```bash
export RPENT_ROOT=/path/to/rpent
codeaspolicy play \
    --robot libero \
    --suite libero_spatial_swap \
    --task 7 \
    --seed 2 \
    --arm cells \
    --model gpt-6-astra \
    --max-turns 40 \
    --budget-s 7200 \
    --out runs/demo

# Tabulate and summarize episode metrics
codeaspolicy summarize runs/demo
```

For RoboDojo via XPolicyLab, see [`docs/robots/robodojo.md`](docs/robots/robodojo.md) and [`integrations/xpolicylab/`](integrations/xpolicylab/).

---

## CLI Reference

| Command | Description |
|---|---|
| `codeaspolicy api` | Print the live generated robot API reference (`--robot`, `--primitives`). |
| `codeaspolicy prompt` | Render and inspect the full system + task prompt offline. |
| `codeaspolicy card` | Boot the simulation host and dump `card.json` for a task/seed. |
| `codeaspolicy generate` | Generate a single-shot `policy.py` from a saved `card.json`. |
| `codeaspolicy run` | Execute a standalone `policy.py` file against the RPent host. |
| `codeaspolicy play` | Run an interactive multi-turn episode (`cells` or `program` mode). |
| `codeaspolicy summarize` | Tabulate episode outcomes, token usage, and step counts. |
| `codeaspolicy compare` | Perform strict paired comparisons across evaluation runs. |
| `codeaspolicy valen-eval` | Evaluate saved visual supervision checks offline without a simulator. |

---

## Supported Robots & Benchmarks

| Robot ID | Benchmark / Stack | Description |
|---|---|---|
| `robodojo_xpl_rgb` | [RoboDojo](docs/robots/robodojo.md) (via XPolicyLab) | Dual ARX-X5 arms with 3-view RGB geometry, C2 coarse motion, and guarded delta manipulation. |
| `robodojo_xpl` / `robodojo` | [RoboDojo](docs/robots/robodojo.md) (XPolicyLab / RPent) | Dual ARX-X5 arms with RGB-D observations and full/no-vla/no-sam primitive sets. |
| `libero` *(default)* | [LIBERO-PRO](docs/robots/libero.md) (via RPent) | Single Franka arm with scripted servos, SAM3 perception, and Pi0.5 VLA integration. |
| `robotwin` | [RoboTwin 2.0](docs/robots/robotwin.md) (via RPent) | Dual-arm manipulation with LingBot-VLA. |
| `robocasa` | [RoboCasa365](docs/robots/robocasa.md) (via RPent) | Mobile manipulation with navigation and RLDX-1 VLA. |

---

## Repository Structure

```text
├── src/codeaspolicy/            # Core CodeAsPolicy package
│   ├── cli.py                   # CLI entrypoint (codeaspolicy)
│   ├── _robot.py                # Base robot abstractions, snapshots, and provenance
│   ├── arms.py                  # Persistent code-cell / program toolkits & compact feedback
│   ├── execution.py             # Sandbox execution limits and bounded output
│   ├── prompt.py                # Live docstring-to-prompt renderer & TaskCard
│   ├── libero.py                # LIBERO-PRO robot interface
│   ├── robodojo.py              # RoboDojo dual-arm interface & 3D geometry helpers
│   ├── robodojo_xpl_rgb.py      # RoboDojo official RGB-only adapter
│   ├── robocasa.py              # RoboCasa365 robot interface
│   ├── robotwin.py              # RoboTwin 2.0 robot interface
│   ├── supervision.py           # Optional Valen visual supervision (off / shadow / pause)
│   ├── stuck_recovery.py        # Opt-in stuck detection and recovery hooks
│   ├── task_acceptance.py       # Public task-acceptance criteria validation
│   ├── hosts/                   # Simulator hosts, IK/splines, and guarded delta controllers
│   └── knowledge/               # Robot operating manuals & public acceptance catalog
├── integrations/xpolicylab/     # XPolicyLab RoboDojo deployment adapter
├── docs/                        # Guides, robot pages, and engineering audits (docs/design/)
├── tests/                       # Contract, kinematics, perception, and prompt test suite
└── pyproject.toml               # Package metadata and console scripts
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
