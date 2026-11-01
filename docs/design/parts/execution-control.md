# 动作执行、控制与回执

[设计索引](../index.md) · [整体架构](../architecture.md) · [补齐顺序](../roadmap.md)

## 状态与范围

初版依据 2026-10-31 UTC 静态分析；2026-11-01 更新 native/XPL 精确回执及对账，见[闭环实施记录](../interaction-contracts.md)。
“现有实现”描述代码语义；没有据此宣称已验证真实硬件停止或实时控制保证。
`ActionReceipt`、`CapabilityManifest`、`BudgetSnapshot` 等是待实现的目标契约草案。
范围包括领域调用、backend、原生动作、双臂控制器与远端 Bridge 的交接。

## 职责边界

- 把程序给出的目标落实为可执行动作，并落实已声明的输入和约束检查。
- 在执行器内处理既定的跟踪误差检查、步进、同步和必要的停止条件。
- 报告动作实际进展、到位误差、检查覆盖及结果确定性。
- 将执行生命周期与观测建立可核验的因果关联。
- 接受取消请求，并单独报告动作停止是否得到确认。
- 不替 Astra 选择任务策略，不把目标到位认定为抓取、支撑或任务成功。

保留直接的 `robo` API；后端可以由驱动、规划器或学习策略实现。
低层控制与停止机制不应依赖 Astra 的下一次响应；具体能力由[说明书](capabilities-manual.md)声明。
当前模拟器步进控制不能直接等同于独立运行的实机伺服与硬件停止系统。

## 输入输出

| 方向 | 内容 | 约定 |
| --- | --- | --- |
| 输入 | 领域方法、参数、目标和约束 | 坐标、单位、控制点和夹爪语义明确 |
| 输入 | `ObservationSnapshot` 引用 | 明确动作基于哪个现场，执行前判断是否仍适用 |
| 输入 | `CapabilityManifest`、`BudgetSnapshot` | 明确实际检查范围、执行额度与停止能力 |
| 输出 | `ActionReceipt` | 关联调用、接受、执行进展、效果确定性与停止情况 |
| 输出 | 当前 `ObservationSnapshot` 引用及到位误差 | 回包对应本动作；无法对应时明确标为待核实 |
| 输出 | `HandoffReport` | 承接 partial、未知效果、约束失败或停止未确认 |

## 现有实现

- `src/pyrualean/_robot.py` 的 `RobotBase._stateful` 实现动作前检查、动作版本递增、回包吸收及账本记录。
- `src/pyrualean/_backend.py` 的 `Backend` 提供 `execute_tool`、`env_steps`、`cancel` 等接口。
- `unwrap` 解析 payload/envelope；`CallRecord` 记录本地调用 index、参数、耗时、ok、错误和摘要。
- `src/pyrualean/hosts/robodojo_native.py` 的 `NativeToolkit.execute_tool` 用锁串行派发，并为 stateful 调用记录观测。
- `src/pyrualean/coordination.py` 的 `move_both` 使用 observe、plan、execute/checkpoint、stop/discard 协议。
- `DualMove` 已表达 completed、rejected、partial，以及 recoverable、双臂误差、局部剩余步数和检查覆盖。
- `src/pyrualean/hosts/robodojo_coordination.py` 的 `Controller` 检查 stale plan、路径、跟踪误差、相对位姿和局部预算。
- `src/pyrualean/hosts/robodojo_tools.py` 的 `_act` 把两臂 joint/gripper 指令放入同一次 `env.step`。
- `src/pyrualean/hosts/xpolicylab_bridge.py` 的 `Bridge` 连接客户端的 update_obs/get_action 与工具的 step。

部分执行和有限纠偏已有实现，不能把它们列为完全空白。
同步保证是共享原生动作时钟与受限跟踪，不是刚性耦合或力控。
控制器声明的几何覆盖包含 sampled arm/table，未声明臂间或物体碰撞保证。
动作前预留计数避免异常后退还额度，但该计数不能证明异常情况下物理动作已经发生。

## 待补齐契约

`ActionReceipt` 应关联 session/episode、代码调用、动作和尝试身份，并引用前后观测。
至少分开以下维度，不用单一 `ok`、`status` 或异常同时承担全部含义：

| 维度 | 需要说明的事实 |
| --- | --- |
| 生命周期 | 请求是否接受、派发、正在执行或已经终结 |
| 效果确定性 | 已知效果还是 unknown，依据什么回执或现场证据 |
| 实际效果 | 确认未执行、部分执行或完成了哪些具体动作 |
| 到位结果 | 实际位姿、误差与目标是否满足，独立于物体关系和任务完成 |
| 取消与停止 | 取消是否请求/接受，停止是否确认，以及确认来源和时间 |
| 资源 | 预留额度、确认消耗及剩余额度分别是多少 |

传输超时意味着等待失败，不自动意味着动作失败、未执行或已停止。
终结状态也可能仍有 unknown 效果；必须进入对账流程，不能仅换一个 action_id 后重试。
重试语义、去重能力、同一动作状态查询以及旧 session 回包处理应由适配器明确声明。
没有相应协议能力的后端应明确其限制，并在效果不确定时阻止依赖动作继续。

Bridge 已按 session、parent action 和 native action ID 精确结算，超时后保留 pending。
迟到 A 回包不能完成 B，旧会话和不完整来源协议被隔离。运行时 `action_receipt / reconcile`
查询原请求而不重放；执行完成与效果已知同时满足时才移除门槛，回执关联到代码调用和观测。
上述内容有协议故障测试；通用预算预留/结算与其他远端后端仍需补齐。

`NativeToolkit.cancel_active_and_wait` 已返回派发范围的取消确认；取消唤醒本地等待，但在途动作保持 unknown，直到准确回执到达。
`Controller.stop` 当前停止计划后续派发，不能证明已派发远端动作或硬件运动已经停止。
目标契约应将执行栅栏、取消确认、停止确认与新鲜现场读取组织成恢复前置条件。
代码 tracing、进程结束和停止确认的边界见[代码运行时](code-runtime.md)。

## 失败交接

| 情形 | 执行器处理 | 下一步 |
| --- | --- | --- |
| 参数或路径检查拒绝，确认零执行 | 返回拒绝原因和当前现场 | 程序有限调整或 Astra 重规划 |
| 可恢复跟踪偏差且效果已知 | 返回 partial、误差及剩余预算 | 程序按显式规则纠偏 |
| 关键约束失败 | 停止后续派发并保留实际终点 | 暂停代码，交回 Astra |
| 传输失败、效果 unknown | 保留动作身份与额度，冻结依赖动作 | 查询回执并重新观察 |
| 取消已请求但停止未确认 | 继续标明在途状态，不放行冲突动作 | 等待确认或按后端能力升级 |
| 局部预算耗尽 | 停止本次继续派发，返回实际进展 | [恢复交接](recovery-handoff.md)在全局额度内判断后续策略 |
| episode 总预算耗尽 | 关闭新动作准入，保留停止状态与未决效果 | 会话管理处理终止与审计，不再开放动作 |

视觉核验失败应附在动作结果之后，不能把已经完成的运动改写为未执行。
动作是否满足物体关系由[验证与证据](verification-evidence.md)判断。

## 验收场景

- 一臂目标不可达时，在另一臂开始动作前拒绝整个同步计划。
- 发生可恢复偏差时，返回两臂实际终点；关键约束失败时暂停后续代码。
- dispatch 前后分别抛异常时，额度均可审计，实际效果确定性不被计数替代。
- A 超时后迟到的观测不能完成 B；旧 episode 的回包不能更新新会话现场。
- 已接受取消但运动未停时，回执不标为停止确认，后续冲突动作被拦截。
- checkpoint、primitive attempts 和 native ticks 分别计量，总预算跨调用保留。
- 结束代码、到达运动目标和完成任务三个结果可独立表达。
