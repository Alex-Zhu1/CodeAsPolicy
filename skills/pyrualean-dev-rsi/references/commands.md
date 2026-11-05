# 测试与实验操作手册

以下命令从仓库根目录执行。`<...>` 为按本轮目标填写的占位值，不要直接粘贴执行。
具体参数以当前源码/`--help` 为准；历史 runner 仅用于核对环境和配置。

## 找到能工作的解释器与测试范围

本机历史实验使用 `runs/organize-table-astra-20261030/venv/bin/python`。
这是本地已有资源的线索；其他机器使用项目环境，不要依赖此目录必然存在。

```bash
PYRUA_PYTHON="$PWD/runs/organize-table-astra-20261030/venv/bin/python"
test -x "$PYRUA_PYTHON"
"$PYRUA_PYTHON" -c 'import sys, pytest; print(sys.executable); print(pytest.__version__)'
PYTHONPATH="$PWD/src" "$PYRUA_PYTHON" -c 'import pyrualean; print(pyrualean.__file__)'
git status --short
```

按改动选择测试，而不是每轮全部运行：

| 改动 | 优先检查 |
| --- | --- |
| request/action 身份、partial、续接 | `tests/test_execution.py`、`test_execution_resume.py`、`test_interaction_contracts.py` |
| 运动轨迹与双臂 | `tests/test_robodojo_motion.py`、`test_robodojo_trajectory.py`、`test_coordination.py` |
| intent / grasp | `tests/test_robodojo_intent.py`、`test_robodojo_grasp.py`、`test_robodojo_staged_grasp.py` |
| 夹持证据、遮挡重关联 | `tests/test_robodojo_object_evidence.py`、`test_robodojo_hold_reacquire.py`、`test_robodojo_continuous_grasp.py` |
| 提示、卡片、反馈 | `tests/test_prompt.py`、`test_robodojo_card.py`、`test_feedback_compaction.py` |

```bash
"$PYRUA_PYTHON" -m pytest -q tests/test_robodojo_intent.py
"$PYRUA_PYTHON" -m ruff check <本轮修改的Python文件>
git diff --check
```

如果选中的解释器没有 Ruff，使用项目已配置的 Ruff 环境并记录来源。
新文件不会出现在普通 `git diff --check` 中，也需要实际检查。
固定 RSI 程序依赖运行时注入 `robo`；不要把全目录 F821 数量当成运行失败数。

## 准备耗时实验

近期可读参考：
[driver](../../../runs/hold-reacquire-20261104-1422/driver.py)、
[启动环境](../../../runs/hold-reacquire-20261104-1422/run.sh)。
它们含该机器的固定路径、GPU 和配置，不要直接执行旧 driver 或复制其中的全部设置。

1. 创建新目录；将本轮假设、配置及对照写入 manifest。以 Asia/Singapore 标明展示时间；
   日志保留原有 UTC 时间戳，不依据历史目录后缀猜时区。
2. 复制实际运行的 `src/`、`integrations/`、相关 `skills/`，包括 untracked 文件。
   排除缓存/凭据，记录 hash。冻结后运行的 `PYTHONPATH` 指向 `frozen/src`；
   实际包导入位置也要验证，不能只依赖启动脚本的意图。
3. 核对 benchmark/XPolicyLab 客户端版本与资产。已有 `lay_out()` 会复制代码/配置并链接
   Assets；见 [客户端布局实现](../../../src/pyrualean/hosts/robodojo_client_root.py)。
   `robodojo_xpl_rgb` 与 RGB-D 协议应保持各自配置。
4. 检查 GPU 进程/可用显存、SAM3 checkpoint、必要端口和模拟器资源。
   只检查当前任务会用到的依赖；纯单元测试不需要拉起 Isaac、SAM3 或 Valen。
5. 分别设置 native action limit、`--max-turns`、`--budget-s` 和外层进程超时。
   `--max-turns` 不会提高原生步数。预算变更记录为独立实验配置。
6. 保存实际 argv、运行 PID/所属进程组和进度；异常清理只针对本轮进程。
   诊断启动重试应有上限，不因重试覆盖已经完成或部分执行的 attempt。

`python -m pyrualean.play --robot robodojo_xpl_rgb` 的 `--seed` 是 layout 索引，
`--eval-seed` 选择 layout 集。官方 `robodojo.sh eval --seed` 表示 eval seed，
两条入口不能机械复制 seed 含义。
依据：[XPL host](../../../src/pyrualean/hosts/xpolicylab_robodojo.py)。

历史配置例子（用于可比性核对，不是推荐默认）：cells / Astra high / RGB / no-vla /
on-demand / pure / monitor off / stuck recovery off / layout 0 / eval seed 0 /
100 turns / 5400 秒。`insert_key` 的 1000 原生步是单独的预算修改。
模型或运行环境的隔离配置应遵循当前已授权设置，不复制旧脚本里的权限配置。
凭据只通过现有受支持机制使用，不写入经验、manifest、日志或代码快照。

## 只读检查结果

助手接受实验根、任务目录或 episode 目录；实验根扫描仅到一层 `*/episode/`。

```bash
python skills/pyrualean-dev-rsi/scripts/inspect_run.py runs/hold-reacquire-20261104-1422 --verify-source
python skills/pyrualean-dev-rsi/scripts/inspect_run.py runs/hold-reacquire-20261104-1422/insert_key/episode
```

`--verify-source` 要传含 `manifest.json` 和 `frozen/` 的实验根。
只校验清单内 hash，不比较当前工作区或 benchmark，也不证明清单覆盖全部运行依赖。
助手缺失结果时报告 incomplete，不猜测运行中还是已中断；多个原生文件全部列出，
不取第一个当最终结果。退出码 0 只表示成功读取，1 表示不完整/异常，均不代表任务成功。

| 文件 | 用途 |
| --- | --- |
| `episode/result.json` | 环境成功、模型 finish、运行状态、配置与耗时 |
| `episode/bench/eval_result/**/_result.json` | 原生分数和逐 layout 结果 |
| `episode/calls.jsonl`、`tool_turns.json` | 动作参数、回执、观测来源及步数 |
| `episode/execution/*.sqlite3` | request/operation/action 对账；用 `mode=ro` 打开 |
| `episode/interaction_events.jsonl` | 交互计时，注意能力与时钟边界 |
| `episode/prompt.txt`、`card.json`、冻结 skill 文本 | 本轮实际启动上下文 |
| 原生视频、`steps/`、`workspace/cell_*.py` | 对齐现场和实际执行代码 |

复用已有分析入口；输出到新文件，原始记录不回写：

```bash
"$PYRUA_PYTHON" -m pyrualean.evaluation.latency <episode>/interaction_events.jsonl --format interaction --output <新输出文件.json>
ffmpeg -v error -i <原生录像.mp4> -f null -
```

视频能解码不等于任务完成；SQLite integrity 正常不等于物理恢复已验证。
日志定位先使用工具请求、结果与交互时间字段，无需读取模型私有推理或认证文件。

## 保存一次验证的最小记录

```text
目标 / 假设 / 对照目录：
代码与测试版本：HEAD、dirty 文件、冻结 hash 清单
运行配置：入口、解释器/客户端、task/layout/eval seed、模型与预算
验证：命令、退出码、测试范围、测试后的源码改动
结果：运行状态、模型 finish、原生 success 与 score（保留各字段量纲）
成本：原生步、高层调用、轮次、启动/策略/总耗时、可用时的 token
证据：原始结果、动作/观测、回放、必要的账本检查
限制：中断/启动失败、缺失日志、对照差异、未覆盖场景
经验更新：复用的 Lxx、新证据或被修正的假设
```
