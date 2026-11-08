# RoboUse: Robot-Use for AI Agents

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a> &ensp;
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white" alt="Python"></a> &ensp;
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Version-0.2.0-orange.svg" alt="Version"></a>
</p>

**RoboUse** brings the **Computer-Use** paradigm to physical and simulated robotics.

Instead of treating robot manipulation as fine-grained, single-step tool calling (which introduces heavy latency, high token costs, and fragile error handling), RoboUse turns a robot into a programmable, interactive Python environment. An AI agent writes structured Python code blocks against a unified `robo` object—allowing perception, multi-step motions, closed-loop retries, and conditional recoveries to execute locally in a single decision cycle.

---

## 🌟 Key Features

- **Robot as a Python Object (`robo`)**  
  Expressive, typed, and composable primitives for manipulation, motion planning, and active sensing (e.g., `robo.move_to()`, `robo.staged_grasp()`, `robo.segment()`, `robo.gripper`, `robo.look_at()`).

- **Closed-Loop Execution & Recovery**  
  - **Staged Grasp & Dual-Arm Coordination**: Resumable staged grasps, dual-arm handoff state continuation, and explicit release verification.
  - **Robust Visual Tracking**: SIFT scale and rotation preservation across tracking updates for stable visual servoing.
  - **Safety & Clearance**: Built-in post-contact retreat clearance guards against low lateral sweeps and accidental collisions.
  - **Automated Stuck Recovery**: Declarative stuck detection and recovery routines preserving the live episode state.

- **Visual Supervision & Acceptance Criteria**  
  - Flexible supervision modes (`shadow` for asynchronous review reports, `pause` for interactive review interruptions).
  - Declarative task acceptance criteria and knowledge-base rules to verify multi-phase task completion.

- **Multi-Robot & Platform Ecosystem**  
  - **RoboDojo** (via XPolicyLab)
  - **LIBERO-PRO**, **RoboTwin 2.0**, and **RoboCasa365** (via RPent)
  - Physical single-arm and dual-arm platforms (e.g., Franka, ARX-X5, AF8)

- **Agent Runtime Agnostic**  
  Supports multiple agent runtimes including **Codex CLI**, **Claude Code** (Claude Agent SDK), and custom loops connecting to any **OpenAI-compatible** endpoint.

---

## 🚀 Quick Start

### Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git
cd PyRUA-Lean

# Install with pip or uv
pip install -e .
```

Verify the installation and view the typed robot API:

```bash
robouse api
```

*(Note: `pyrualean` is also available as a backward-compatible command alias).*

---

### How It Works: Code-as-Action

Rather than issuing one atomic tool call per step, the agent writes Python code that executes against the live robot environment:

```python
# 1. Segment the target object and destination
target = robo.segment("red mug")
sink = robo.segment("sink basin")

# 2. Execute staged grasp with local closed-loop verification
if target:
    robo.staged_grasp(target)
    
    # 3. Check physical gripper status before proceeding
    if robo.gripper.has_object():
        # Move above destination and release safely
        robo.move_above(sink, height_m=0.15)
        robo.release()
        robo.retreat()
    else:
        # Trigger local recovery without round-tripping to LLM
        robo.recover_grasp(target)
```

---

### Running an Episode

Run an interactive code-driven episode using the CLI:

```bash
# Run one episode using code cells mode
robouse play \
    --robot libero \
    --suite libero_spatial \
    --task 0 \
    --arm cells \
    --model gpt-6-astra \
    --out runs/demo

# Tabulate and summarize run metrics
robouse summarize runs/demo
```

---

## 🛠️ CLI Reference

The `robouse` CLI provides tools for inspection, prompt rendering, agent execution, and evaluation:

| Command | Description |
|---|---|
| `robouse api` | Print the typed robot API reference and primitive specifications. |
| `robouse play` | Run an episode driven by an AI agent (supports `cells` and `program` modes). |
| `robouse prompt` | Render and inspect the full generation prompt and system instructions. |
| `robouse card` | Boot the simulation environment and dump the task card / initial state. |
| `robouse summarize` | Tabulate episode outcomes, token consumption, and primitive call ledgers. |
| `robouse compare` | Perform strict paired comparisons across evaluation runs. |
| `robouse valen-eval` | Evaluate saved visual supervision checks offline without running a simulator. |

---

## 🦾 Supported Robots & Benchmarks

| Robot ID | Benchmark / Stack | Description |
|---|---|---|
| `robodojo_xpl_rgb` | [RoboDojo](https://dagroup-pku.github.io/PyRUA-Lean/docs/robots/robodojo/) (via XPolicyLab) | Dual ARX-X5 arms with RGB observations, staged grasping, feature tracking, and motion primitives. |
| `libero` *(default)* | [LIBERO-PRO](https://dagroup-pku.github.io/PyRUA-Lean/docs/robots/libero/) (via RPent) | Single Franka arm with scripted servos, SAM3 perception, and Pi0.5 VLA integration. |
| `robotwin` | [RoboTwin 2.0](https://dagroup-pku.github.io/PyRUA-Lean/docs/robots/robotwin/) (via RPent) | Dual-arm manipulation with LingBot-VLA. |
| `robocasa` | [RoboCasa365](https://dagroup-pku.github.io/PyRUA-Lean/docs/robots/robocasa/) (via RPent) | Mobile manipulation with navigation and RLDX-1 VLA. |
| *Real Robot* | Physical AF8 / Franka | Dual-arm or single-arm hardware with force-guarded primitives and wrist cameras. |

---

## 📁 Repository Structure

```text
├── src/robouse/             # Core RoboUse package
│   ├── cli.py               # CLI entrypoint (robouse / pyrualean)
│   ├── _robot.py            # Base robot abstractions and primitive definitions
│   ├── execution.py         # Code execution engine and sandbox limits
│   ├── staged_grasp.py      # Resumable staged grasping logic
│   ├── stuck_recovery.py    # Declarative stuck recovery mechanisms
│   ├── task_acceptance.py   # Multi-phase task acceptance verification
│   ├── hosts/               # Robot hosts & environment adapters (RoboDojo, RPent, etc.)
│   └── knowledge/           # Acceptance criteria & prompt templates
├── integrations/            # Benchmark integrations (e.g. XPolicyLab)
├── tests/                   # Comprehensive test suite
├── docs/                    # Architectural guides, references, and specifications
└── pyproject.toml           # Project configuration and entry points
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
