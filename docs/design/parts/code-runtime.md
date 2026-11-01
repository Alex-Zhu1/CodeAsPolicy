# 代码运行时与会话

[设计索引](../index.md) · [整体架构](../architecture.md) · [补齐顺序](../roadmap.md)

## 状态与范围

本文依据 2026-10-31 UTC 当前工作区静态分析，包含未提交的模块和改动。
“现有实现”不等同于运行验证；本文不报告本次测试或机器人执行结果。
`DecisionContext`、`BudgetSnapshot`、`HandoffReport` 等目标契约仍是草案。
本模块讨论已有 Python 执行方式及其完整语义，不在此增加 JavaScript 运行时。

## 职责边界

- 为生成代码提供稳定的领域绑定、受约束的执行环境和可追溯的代码身份。
- 管理 cell/program 的开始、返回、异常、暂停、预算耗尽和 episode 结束。
- 汇总代码输出、动作回执、观测和验证结果，供程序或 Astra 后续判断。
- 明确赋值、文件、已执行动作的保留行为，不提供代码块自动回滚。
- 将执行控制交给[执行器](execution-control.md)，将新策略判断交回 Astra。
- 维持普通 Python 分支、函数和有限循环，不要求固定任务阶段或 DSL。

框架区分代码执行结束、动作完成、任务完成三个事实。
恢复是在真实现场生成和运行后续代码；它不恢复已经展开的 Python 调用栈。

## 输入输出

| 方向 | 内容 | 约定 |
| --- | --- | --- |
| 输入 | 代码及其身份 | 标明 cell/program、来源、版本和入口 |
| 输入 | `DecisionContext` | 包含任务、能力、观测、证据、约束和预算的引用 |
| 输入 | `CapabilityManifest` | 绑定当前会话可调用的 `robo` 能力 |
| 输入 | `BudgetSnapshot` | 区分本次代码额度与 episode 总额度 |
| 输出 | 代码执行状态、stdout/stderr、显示内容 | 输出截断必须可见；状态不代表任务判定 |
| 输出 | `ActionReceipt`、`ObservationSnapshot`、`EvidenceRecord` 的引用 | 不能用代码返回值替代动作与现场证据 |
| 输出 | `HandoffReport` | 说明为何交回、现场变化、待确认项和后续许可 |

## 现有实现

- `src/pyrualean/arms.py` 的 `ArmToolkit` 提供 cells、program 两种代码交互方式。
- `ArmToolkit._python` 在持久命名空间内执行 cell，允许跨调用复用已定义的 helper。
- `ArmToolkit._run_program` 执行生成的程序文件，并汇总程序结果。
- `src/pyrualean/runner.py` 的 `load_policy` 使用唯一模块名加载策略入口。
- `run_policy` 生成 `PolicyOutcome`，区分 completed、paused、finished、error、timeout；该类型注明与任务成功独立。
- `src/pyrualean/execution.py` 的 `ExecutionBudget` 用 trace 检查工作区 Python 帧和动作前预算。
- `ExecutionLimit` 派生自 `BaseException`，避免被普通 `except Exception` 误当作可重试动作错误。
- `BoundedOutput` 在写入时限制输出内存，`ExecutionLimits` 还约束显示图片数。
- `src/pyrualean/sandbox.py` 的 `make_builtins`、`blocked_names`、`audit_source` 提供导入、文件访问和源码审计护栏。
- `ArmToolkit._failure_feedback` 汇总错误位置、动作尝试、剩余调用次数和继续建议。

上述代码已具备保留现场、回报错误和重新提交代码的基础。
Sandbox 明确不是对恶意进程内 Python 的完整隔离边界。
Python tracing 不能硬停阻塞的原生调用；其检查通常发生在原生调用返回之后。
`runner.run_host` 的进程组超时是进一步兜底，不能等同于远端设备停止确认。

## 待补齐契约

每次代码执行应有明确会话归属，并关联它生成时使用的 `DecisionContext`。
输入观测、动作和验证可有多个版本；运行时不能自动把旧变量解释为最新现场。
跨 cell 保留的对象应能区分普通计算值与带观测来源的目标或证据。
某次赋值被中断时，旧变量可能仍在命名空间中，反馈应明确这一点。

| 维度 | 目标约定 |
| --- | --- |
| 代码状态 | 正常返回、被暂停、失败、限额和 episode 终止分别表达 |
| 动作状态 | 通过 `ActionReceipt` 独立表达，不由代码状态推导 |
| 效果确定性 | 明确哪些动作效果已知，哪些仍待执行系统核实 |
| 会话保留 | 明确命名空间、文件、helper、观测引用和累计额度是否保留 |
| 后续执行 | 新代码在满足现场确认和预算条件后执行，不回放失败代码块 |
| 反馈交付 | 错误、暂停、输出截断和证据缺失保持结构化且彼此可区分 |

`BudgetSnapshot` 应分别记录以下资源，而非都命名为 actions：

- 代码调用次数与本次 wall time。
- primitive 调用尝试次数，包括可能零物理步的失败尝试。
- 双臂执行 checkpoint 的提交次数。
- 已知或保守预留的 native ticks，注明二者差异。
- 新观测采集、视觉核验和显示输出的额度。
- episode 累计资源，以及为验证、收尾和恢复预留的资源。

现有 `_stateful` 是动作额度的计数点；`move_both` 的多个 checkpoint 会多次经过它。
目标契约必须先明确计量单位，再决定如何汇总，不能把 checkpoint 次数误报为高层动作次数。
读取历史图像不应隐式重置观测有效性，也不应算成独立现场确认。
相关规则由[观测系统](observation-perception.md)和[验证系统](verification-evidence.md)共同提供。

### 完整代码就绪与接入协议

运行时和模型接入共同记录 request、完整工具参数、校验通过与执行开始的时间，
关联 decision/attempt/tool 身份；详见 [交互延迟设计](../interaction-latency.md)。
直接 OpenAI 的当前 transport 等完整 JSON 再派发；Codex、Claude 的流式能力须按各自客户端判断。
未来流式派发只接收 provider 确认完成的工具项，并保留整个 cell 编译和准入检查，不能执行 Python token 前缀。

工具项去重、断流后对账、迟到请求拒绝和前项暂停后的依赖阻断属于同一执行契约。
生成取消与机器人停止分别记录；代码已派发时不能通过重试请求再次执行副作用。
完整代码校验通过后若仍在排队，实际派发前须重查请求有效性、deadline、episode/预算和观测依赖，
不把 `code_ready` 当作永久许可。已开始执行的代码仍遵守自己的执行限额，生成期限不表示动作停止。
官方 Astra 直连与 Chat Completions 兼容网关须声明各自协议和转换责任，不能只增加一个 stream 参数就声称完成适配。

## 失败交接

| 情形 | 运行时处理 | 后续接收方 |
| --- | --- | --- |
| 语法或导入错误，尚未派发动作 | 输出代码位置与已知副作用情况 | Astra 修正代码 |
| 已执行部分动作后代码异常 | 保留动作记录、现场和命名空间变更提示 | Astra 只生成剩余工作 |
| 程序规则可以处理的结果 | 返回具体结果字段，允许预算内分支 | 当前 Python 程序 |
| 视觉不确定或关键约束失败 | 暂停后续依赖动作，保留执行回执 | Astra 重新判断 |
| 原生调用未返回或动作效果未知 | 请求执行系统核实，限制后续动作许可 | [恢复交接](recovery-handoff.md) |
| episode 已结束或总预算耗尽 | 拒绝进一步动作，输出终止原因 | 任务记录与评估 |

代码限额通常应交回模型；episode 总额或明确终止则不能通过另一个 cell 重置。
取消请求及停止确认由执行系统报告，不能由 Python 异常处理器推断。

## 验收场景

- 无限 Python 循环会交回控制，已完成的赋值和动作仍保留。
- 阻塞原生调用超时被标为执行状态未确认，不声称机器人已经停止。
- 第一个动作成功、第二个动作失败时，后续修正代码不会自动重放第一个动作。
- 两个 cell 使用同一个 helper 时，预算检查仍覆盖该 helper 的执行。
- 一个 `move_both` 经多个 checkpoint 后，反馈能分别显示高层意图、执行段和 native ticks。
- 输出被截断时仍能读取原始失败原因、交接状态和关键证据引用。
- 普通 cell 返回不触发任务成功判定，动作到位也不自动结束任务。
- episode 结束后，继续提交代码不能重新获得运动许可。
