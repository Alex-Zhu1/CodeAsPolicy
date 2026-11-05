# 任务定义与会话

[设计索引](../index.md) · [整体架构](../architecture.md) · [推进路线](../roadmap.md)

## 状态与范围

初版基于 2026-10-31 UTC 静态梳理；2026-11-02（Asia/Singapore）补充未提交的来源任务卡实现，见[仓库审查](../repository-review-20261102.md)。
“现有实现”描述已读到的代码；`TaskSpec`、`BudgetSnapshot` 等名称表示目标协议草案，尚未统一实现。
设计对象是 Robot Use 完整框架；Computer Use 仅提供交互与执行语义的设计参照。
本次文档整理不改变运行逻辑，也不声称已经完成实机或仿真验证。

## 职责与边界

任务定义负责明确要完成什么、必须遵守什么，以及谁有权给出最终任务判定。
会话负责把任务绑定到具体环境、执行模式和累计预算，并管理开始、暂停、继续与终止。
Astra 可以提出阶段和假设，但不能把这些派生内容自动升级为原始任务要求。
会话终止与机器人停止确认需要分别表达；结束模型对话不证明设备已停止。

框架应保留三种执行配置的区别：

| 配置 | 现有语义 | 恢复边界 |
| --- | --- | --- |
| 在线 `cells` | Python namespace 跨调用持久，模型读取反馈后继续生成代码 | 可在同一现场修复剩余工作 |
| `program` | 每个程序使用新 namespace，workspace 与现场保留 | 允许次数决定是否有下一次程序；`max_programs=1` 无第二次机会 |
| 离线生成 / RSI | 固定程序执行；RSI 复用 cells 通道传递离线程序 | 没有在线模型重规划，只能运行程序已有分支 |

`arm` 在运行参数中表示 cells/program 模式，不表示左臂或右臂。
暂停继续是从真实现场生成后续代码，不是恢复到被中断的 Python 栈帧。
宿主执行记录可通过 `ArmToolkit(..., execution_state_path=..., resume=True)` 恢复到同一后端会话，
保留调用预算、原墙钟截止时间和未决动作；不恢复 Python namespace 或重建仿真现场。
请求与命名操作的防重放边界见[恢复与交接](recovery-handoff.md#请求逻辑操作与重启恢复)。

## 输入输出

| 方向 | 内容 | 当前载体 / 目标草案 |
| --- | --- | --- |
| 输入 | 权威任务、benchmark 标识、初始公开状态与图像 | `TaskCard` |
| 输入 | host 身份、实际能力、执行模式 | `RobotAdapter` / `CapabilityManifest` 草案 |
| 输入 | 时间、环境步数、代码调用及程序次数限制 | 分散配置 / `BudgetSnapshot` 草案 |
| 输出 | 不可混淆的任务要求、派生条件和假设 | `TaskSpec` 草案 |
| 输出 | 会话状态、终止原因、剩余预算 | 当前 result 字段 / 统一会话状态草案 |
| 输出 | 模型声明与任务权威判定 | `finish` 与 `environment_success`，继续分开保存 |

## 现有实现

- `src/pyrualean/prompt.py`：`TaskCard` 保存任务指令、对象名、初始图像、机器人事实和步数预算；`render()` 区分实际初始图与 transfer 示例图。
- `TaskCard.sources` 保存组装、原始指令、初始观测、预算及能力的来源说明；旧 JSON 缺少此字段时按空字典读取。
- `TaskCard.acceptance`（2026-11-04 新增）保存有来源的公开验收要求，schema 为 `pyrualean.task_acceptance.v1`。每项包含 `id`、`kind`、`condition`、`source`；kind 区分结果 `outcome`、顺序 `order`、持续约束 `invariant` 和收尾状态 `final_state`。这是原始指令的补充，不替换 `task_language`。
- 验收要求的覆盖范围 `coverage` 默认 `unknown`；任务提供方声明 `partial` 或 `complete` 时必须同时提供 `coverage_source`。空列表表示未提供补充要求，不能声明完整覆盖。提供了完整要求也不表示现场已经满足要求。
- `acceptance.verdict_authority` 默认 `environment`；任务卡不装载验收结果。RoboDojo 现有公开观测没有提供详细验收要求，因此各 host 默认保留空列表和未知覆盖范围，不读取隐藏判分、对象真值，也不从任务名推导补充要求。
- 全量与精简任务卡均展示验收要求及来源。结束前的提示要求核对完整原始指令、结果、顺序、持续约束和收尾状态，并区分观测证据、模型假设与权威验收。`plan(..., status="done")` 仍只记录子目标声明。这是上下文与数据契约的改动，没有新增自动验收器、强制完成门禁或未知覆盖时停止执行的规则。
- 审查期间新增 `initial_state` / `provenance`，保存未舍入测量、观测标识、时钟、产物摘要和创建时预算；完整审计数据在 `card.json`，模型入口用 `render(compact=True)` 的启动摘要。该并行增量的验证状态见仓库审查记录。
- `hosts/robodojo_card.py::build_card` 供 RPent 与 XPL RGB-D/RGB 共用：原始任务指令不改写，缺失或无效初始位姿/开度标为 unknown，预算回退标明配置来源，不从任务名或隐藏状态推导任务事实。
- RoboDojo 任务卡保存启动时的执行能力快照；运行中能力应重新调用 `motion_capabilities()` 查询。来源字段和能力快照尚未构成版本化的统一任务协议。
- `src/pyrualean/prompt.py`：`SCORING_NOTES` 区分实时成功信号与最终场景评分，`CONTRACT` 明确离线程序执行期间没有模型。
- `src/pyrualean/robots.py`：`RobotAdapter` 描述机器人类、host 和知识来源；`live_success` 指示是否提供在线任务成功信号。
- `src/pyrualean/play.py`：`play_episode()` 启动 host、生成任务卡、建立 toolkit、运行模型循环并写入最终结果。
- `src/pyrualean/arms.py`：`ArmToolkit.execute_tool()` 拒绝最终 finish 之后的代码；`_finish()` 保持最终决定幂等。
- `integrations/xpolicylab/PyRUA_RSI/model.py`：`Model._play()` 调用 cells 通道执行固定程序；不能据此认为 RSI 有在线规划器。

当前 `TaskCard.facts` 仍是通用字典；公开任务验收要求可放入独立的 `acceptance`，模型假设保留在规划便笺中。
`finish(status="success")` 是模型声明；`play_episode()` 另外读取 backend 的任务结果。

## 待补齐的契约

下列字段是设计草案，不是可以直接调用的现有 API。

| 草案对象 | 建议内容 | 必须保持的约定 |
| --- | --- | --- |
| `TaskSpec` | task_id、instruction、constraints、acceptance、verdict_source、assumptions | 派生验收条件引用原始要求；假设保留来源与未验证状态 |
| `BudgetSnapshot` | session_id、revision、各预算已用/剩余/计量单位 | 不知道的额度标为 unknown；暂停和恢复不退款 |
| 会话状态 | active、paused、terminal，以及 reason、pending_action_ids | paused 不开放绕过动作前置条件的通道；terminal 不再接收动作 |

任务资源应绑定到明确的环境实例与 `CapabilityManifest` 版本，避免重连后沿用另一现场的事实。
剩余预算应按权威计数器计算；阶段修复额度与全局额度分开，前者目前主要由程序自己维护。
统一 `episode_finished` 的类型和原因字段；当前不同路径存在布尔值与终止原因字符串两种用法。
保留执行状态、动作结果、模型判断和任务评分四种信息，避免单个 `success` 覆盖全部层次。
P0 先明确任务和会话协议；P1、P2 接入动作对账及恢复；P3 再完善上下文消费。

每次需要 Astra 新判断时，另记录 decision 身份、交互目标、deadline 及其与 episode 总额度的关系。
等待超时使该请求失效，迟到代码不得重新获得执行许可；供应端取消确认和设备停止确认分别记录。
仍在执行的动作由执行层持有责任，不能因模型请求重试而重复派发。
这些是待补契约，时点与失败交接见 [交互延迟设计](../interaction-latency.md)。

## 失败交接

| 情况 | 接收方 | 必须携带的信息 |
| --- | --- | --- |
| 任务有矛盾或必需能力缺失 | Astra / 任务提供方 | 原始要求、冲突项、缺失能力；不擅自放宽目标 |
| 初始化失败 | 会话管理与记录模块 | 失败阶段、环境是否建立、是否有动作发生 |
| 暂停或普通代码失败 | Astra | `HandoffReport` 草案、最新观测、已有进展、剩余预算 |
| 达到预算或权威终止 | 会话管理与执行器 | 终止原因、在途动作、停止是否确认；随后拒绝新动作 |
| 最终评分不可获得 | 记录与评估 | 保留 unknown 及证据缺口，由评分策略决定如何计分 |

## 验收场景

1. 相同任务以 cells 和单次 program 运行，前者可在线修复，后者失败后不获得额外程序次数。
2. RSI 通过 cells 运行固定程序，报告仍明确说明没有在线模型重规划。
3. 模型宣称成功但 benchmark 未确认，两个结果均保留且能看到各自来源。
4. stuck 恢复保留环境、namespace 和累计预算；预算耗尽时不能重新打开会话。
5. 初始示例图来自 transfer seed，后续目标必须依据实际现场重新定位。
6. 结束会话时存在未确认动作，报告保留在途状态而非声称机器人已停止。

相关设计：[Astra 上下文](planner-context.md)、[知识与 skill](knowledge-skills.md)、[记录与评估](records-evaluation.md)。
