# 恢复与控制权交接

[设计索引](../index.md) · [整体架构](../architecture.md) · [补齐顺序](../roadmap.md)

## 状态与范围

本文依据 2026-10-31 UTC 当前工作区静态分析，包含未提交模块及改动。
初版来自代码阅读；2026-11-01 补充[闭环实施与故障测试](../interaction-contracts.md)，未进行真实机器人任务验收。
`HandoffReport`、`DecisionContext` 等是目标契约草案，不能视为已存在的统一 API。
恢复指从实际现场继续任务；它不回滚环境、不自动重放动作、不恢复旧 Python 调用栈。

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
