# 恢复与控制权交接

[设计索引](../index.md) · [整体架构](../architecture.md) · [补齐顺序](../roadmap.md)

## 状态与范围

本文依据 2026-10-31 UTC 当前工作区静态分析，包含未提交模块及改动。
初版来自代码阅读；2026-11-01 补充[闭环实施与故障测试](../interaction-contracts.md)，未进行真实机器人任务验收。
`HandoffReport`、`DecisionContext` 等是目标契约草案，不能视为已存在的统一 API。
恢复指从实际现场继续任务；它不回滚环境、不自动重放动作、不恢复旧 Python 调用栈。

2026-11-02 增量：统一工具入口现已接入请求去重、失败后的反馈标识、命名操作记录和可持久化的执行账本。
下述机制已有可控后端及进程退出测试；不代表实机停止或仿真器重启后现场还原已经验收。

## 职责边界

- 识别当前问题属于代码修复、程序内纠偏、证据补充、执行对账还是策略重规划。
- 把失败前后发生的动作、已知现场、未知效果和剩余预算交给正确的接收方。
- 决定何时允许程序继续、何时交回 Astra、何时必须等执行系统确认。
- 维持 episode、资源、观测版本和恢复记录的连续性。
- 避免把一次不确定证据直接认定为 stuck，也避免无进展的无界重复。
- 不提供固定抓取、交接或任务阶段模板；具体恢复策略由 Astra 结合现场生成。

程序内可处理的偏差应留在程序内；需要新的任务判断时再进入模型闭环。
效果未知与停止未确认首先是执行状态问题，不能仅靠模型生成新方案解决。

## 输入输出

| 方向 | 内容 | 约定 |
| --- | --- | --- |
| 输入 | 原始错误、暂停或无进展原因 | 保留源模块、代码位置和触发时刻 |
| 输入 | `ActionReceipt` | 列出已完成、部分执行、未知及仍在途动作 |
| 输入 | `ObservationSnapshot`、`EvidenceRecord` | 保留来源、版本、有效性和不确定性 |
| 输入 | `TaskSpec`、`CapabilityManifest` | 恢复动作仍受原任务和当前能力约束 |
| 输入 | `BudgetSnapshot` | 延续累计消耗，不因恢复获得新 episode 额度 |
| 输出 | `HandoffReport` | 说明交接原因、接收方、恢复前置条件和禁止直接重放的动作 |
| 输出 | 更新后的 `DecisionContext` | 为 Astra 写后续代码提供真实现场与明确未知项 |

## 现有实现

- `src/pyrualean/recovery.py` 的 `failure_details` 分类 syntax、primitive、verification、import、runtime 和执行限额。
- `failure_details` 保留生成代码位置，`result_is_error` 统一部分运行时错误标识。
- `src/pyrualean/arms.py` 的 `ArmToolkit._failure_feedback` 返回动作尝试、最后动作、剩余调用数和 retry_allowed。
- 同方法明确 `automatic_replay=False`、`scene_rolled_back=False`，提示命名空间可能已改变。
- 连续同类代码失败通过 fingerprint 计数，达到阈值时建议 change_approach_or_finish。
- `ArmToolkit._feedback` 关联 coordination 和 supervisor 反馈，保留最后动作及当前状态。
- `src/pyrualean/coordination.py` 的 `CoordinationPause` 将关键执行问题交回，recoverable partial 仍可由 Python 处理。
- `src/pyrualean/stuck_recovery.py` 的 `StuckRecoveryConfig` 配置模型确认 stuck 后的恢复指令和轮数。
- `ArmToolkit._finish` 在 cells 模式处理 `finish(status='stuck')`，有额度时返回 `episode_finished=false` 并交付恢复 skill。
- stuck 恢复沿用同一 episode，不重置动作、调用或时间预算；终止现场及耗尽预算不能恢复。

目前已有结构化代码修复、关键执行暂停和确认 stuck 后的策略恢复。
反馈已有 `pyrualean.handoff.v1` 最小字段：来源、最后动作、未决动作、下一责任方和可执行边界；pure 和压缩反馈均保留。
native/XPL 在途动作可对账；unknown 时新 cell 也不能派发，视觉 yes 不会清除执行门槛。完整预算、依赖图和实机停止确认仍待补齐。
`retry_allowed` 当前是基于已知停止原因和调用额度给出的建议；执行前仍需复查预算和现场，不能解释为重试许可或效果确定性保证。

### 请求、逻辑操作与重启恢复

`python`、`run_program` 和 `finish` 的工具参数新增 `request_id` 和 `resume_token`。
MCP 要求提供稳定的 `request_id`；直接 OpenAI runtime 还可使用供应端的调用 ID，且不依赖 tracing 开关。
同一请求 ID、同一参数只返回已存结果与原始图像，不增加调用预算、不再运动；更改参数后复用 ID 会被拒绝。
原请求若在完成记录写入前中断，则只返回待恢复状态，不重跑代码来补造结果。
缓存结果标记 `execution.replayed=true`，其图像及结果是历史数据，当前未决动作仍以新的 handoff 为准。

失败或暂停后，反馈给出 `execution.resume_token`。MCP 的后续新请求必须携带最新标识；
直接模型循环在读取反馈后发起新决策时传递它。宿主同时校验请求入队时的反馈版本，
阻断失败前已排队的调用；已失败 decision 永久失效，修复成功也不重新接收其迟到调用。
恢复标识在修复成功后继续有效，避免迟到的旧 MCP 请求重新获得执行资格。
独立的同步 Python 嵌入调用兼容旧签名，但调用方应提供稳定请求身份来获得跨重试去重。

对不能重复的逻辑操作，代码使用 `once(step_id, robo.method, *args, **kwargs)`：

```python
result = once("lift-red-1", robo.move_to, target, arm="left")
if not result["value"]["reached"]:
    print(result)  # 从实际终点规划剩余运动，不能把调用返回视作任务完成
```

同一 step ID 必须对应同一个方法与参数。正常返回的操作再次出现时只返回保存的 JSON 值；
`status="returned"` 表示方法返回，不表示到达目标或任务成功。`once(step_id)` 可读取操作及关联回执。
若操作被中断，再次执行同一 ID 会暂停；先对账，再为确有必要的剩余工作分配新的 continuation ID。
有意重复的事件分别编号，例如 `press-red-1`、`press-red-2`。
框架不按代码文本或坐标去重，也不推断任意新代码的业务意图；绕开 `once` 的新动作仍是新请求。

生产 `play_episode` 在 `execution/<robot-session-id>.sqlite3` 写入执行记录。
请求开始、调用预算、动作派发前的意图、执行回执、命名操作结果和工具结果分别持久提交；
单写者文件锁阻止两个宿主同时使用一份账本。SQLite 使用 FULL 同步，不用 pickle 恢复对象。
嵌入调用不提供 `execution_state_path` 时使用内存账本，仅支持本进程去重。

连接到**仍存在的同一个执行端会话**后，宿主可创建：

```python
restored = ArmToolkit(
    robot_for_same_backend_session, arm="cells", workspace=original_workspace,
    execution_state_path=original_database, resume=True,
    # 使用与原会话一致的 task_card_path 和预算配置
)
context = restored.resume_state()
```

恢复校验会话、机器人、任务卡与预算配置，保留已用调用数、原墙钟截止时间、请求及动作身份。
未确认动作按 unknown 处理；没有准确回执时仍不放行。Python 变量、调用栈和视觉证据不恢复，
需要无副作用地重建 helper，并重新观察现场。恢复出的新代码也须携带新的恢复标识。
`close()` 在宿主停止处理调用后释放账本锁；执行线程仍占用时返回 false，保留锁而不阻塞关闭流程。
普通 `play_episode` 仍启动新 episode；这个恢复入口不负责重新启动仿真器、重建物理现场或恢复另一执行端的会话。
后端没有稳定 session ID 或原动作查询能力时，不能用旧记录猜测执行结果。

## 待补齐契约

`HandoffReport` 应至少包含以下内容；具体字段名与统一协议一并确定。

| 内容 | 作用 |
| --- | --- |
| 原因与接收方 | 明确交给程序、执行系统、Astra 还是任务结束处理 |
| 失败位置 | 保留代码、能力调用和原始错误，避免后续清理异常覆盖主因 |
| 动作边界 | 指出最后确认动作、部分执行动作、未知动作和禁止重放项 |
| 现场与证据 | 引用 `ObservationSnapshot`、`EvidenceRecord`，注明缺失和过期 |
| 代码残留 | 指出赋值、helper、文件或状态可能已改变，旧变量可能仍存在 |
| 资源 | 附 `BudgetSnapshot`，区分局部额度和 episode 剩余资源 |
| 前置条件 | 明确重新观察、回执对账、停止确认、补证据等未满足条件 |
| 后续许可 | 明确仅允许观察、允许有限动作或 episode 已终止 |

恢复资格不能只由“还有调用次数”决定；必须同时考虑效果确定性、停止状态和证据有效性。
恢复上下文应保留已确认进展，不能把当前现场假设成任务初始状态。
动作后超时只说明回包没有按时到达，不能由 Astra 猜测“应该没有执行”。
对账完成前，即使模型已经生成新代码，依赖未知动作的执行仍应受阻。
对账及取消/停止责任见[执行控制](execution-control.md)。

程序暂停后的后续代码具有新的代码调用身份，关联原 `HandoffReport`。
可以复用 cells 中的计算 helper，但需要重新确认依赖旧观测的目标和变量。
恢复 skill 提供诊断思路和可选程序模式，不授予能力、不重置预算、不构成现场证据。
finish 的模型声明、episode 结束和环境成功判断应分别记录。
遇到不可恢复的终止条件时，应保留未完成项和证据，不将“停止尝试”写成“任务成功”。

## 失败交接

| 触发情况 | 首要接收方 | 继续条件 |
| --- | --- | --- |
| 无动作的语法/接口错误 | Astra 修正代码 | 代码可运行且能力约定匹配 |
| 已知效果的可恢复数值偏差 | 当前程序 | 有显式纠偏规则及局部预算 |
| 视觉 unknown 或证据冲突 | 观测/验证系统，必要时 Astra | 获得有效证据或改变策略 |
| 未处理执行错误、关键约束失败 | Astra | 现场与动作边界已确认 |
| 动作效果 unknown | 执行系统 | 回执与现场对账，不能直接重放 |
| 已请求取消、停止未确认 | 执行系统 | 确认停止或按能力声明处理 |
| 模型确认 stuck | Astra 与按需恢复 skill | episode 未结束、仍有恢复和执行额度 |
| 本次代码或动作局部额度耗尽 | Astra / 恢复模块 | 核对现场与全局剩余额度，决定剩余工作 |
| episode 总预算耗尽或终止 | 任务记录与评估 | 不允许通过新代码重置总预算 |

补充证据应遵循[观测来源](observation-perception.md)与[验证有效性](verification-evidence.md)规则。
再次读取同一张图片不自动提供第二次独立确认。
代码恢复的命名空间与执行边界见[代码运行时](code-runtime.md)。

## 验收场景

- 第二个动作失败后，报告清楚保留第一个动作，后续代码只处理实际剩余工作。
- 普通数值偏差在程序内有限纠正，关键约束失败立即交回且不执行依赖动作。
- 同一失败反复出现时，反馈保留累计次数并促使换策略，而非无界重试。
- A 的执行效果未知时，允许查询与观测，阻止直接重放 A 或发出依赖 A 的动作。
- 取消返回后未确认停止时，恢复报告仍包含该前置条件。
- `finish(status='stuck')` 获得恢复 skill 后沿用现场及累计预算；终止 episode 不会复活。
- 跨 cell helper 可复用，旧目标需要重新确认，旧 Python 调用栈不会被恢复。
- 任务未完成但资源耗尽时，结果保留失败原因和未完成项，不被代码正常退出覆盖。
