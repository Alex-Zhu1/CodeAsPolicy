# 记录与分层评估

[设计索引](../index.md) · [整体架构](../architecture.md) · [推进路线](../roadmap.md)

## 状态与范围

本文基于 **2026-10-31 UTC 当前工作区的静态梳理**，包括尚未提交的模块与修改。
当前已有调用、观测、恢复、监督和任务汇总记录；贯穿全部模块的统一事件协议尚未实现。
`ActionReceipt`、`ObservationSnapshot`、`EvidenceRecord` 等名称均表示目标草案。
对象是 Robot Use 框架的可审查性与评估，Computer Use 仅作为动作归属和结果不确定性的设计参照。

## 职责与边界

记录模块保存“依据什么、执行了什么、发生了什么、为何继续或停下”的关联，供复现和诊断使用。
评估模块在保留原始事实的基础上应用评分策略，不反向改写任务、证据或运行时阈值。
正常 Python 执行、动作到位、视觉事实成立、模型宣称完成和 benchmark 成功必须分别记录。
未知结果不能静默变成否定；benchmark 可以把未知计失败，但必须保留其底层未知原因。

| 层次 | 评价对象 | 不能替代的结论 |
| --- | --- | --- |
| 执行 | 拒绝、部分执行、超时、暂停、停止确认 | 任务成功 |
| 感知与验证 | 正确率、误报漏报、unknown、时延、证据有效性 | 全程动作可靠性 |
| 规划与恢复 | 有效进展、重试代价、恢复效果、错误归因 | 单一模型 token 数的优劣判断 |
| 任务 | 权威完成判定与成本 | 每个中间视觉判断均正确 |

## 输入输出

| 方向 | 内容 | 当前载体 / 目标草案 |
| --- | --- | --- |
| 输入 | 任务卡、完整 prompt、代码、配置、skill 文本 | 运行目录快照 |
| 输入 | 工具调用与 primitive 参数、结果、耗时、步数 | `tool_turns.json`、`calls.jsonl` |
| 输入 | 图像来源、事实问题版本、视觉结果、暂停原因 | observation、supervision audit / 统一证据协议草案 |
| 输入 | 模型声明、native verdict、预算与结束原因 | `result.json` / `BudgetSnapshot`、`HandoffReport` 草案 |
| 输出 | 可追溯事件、单次结果、聚合统计 | 现有 JSON/JSONL 与 `EpisodeMetrics` |
| 输出 | 离线视觉评估与人工标注模板 | Valen evaluation 输出目录 |

## 现有实现

- `src/pyrualean/_backend.py`：`CallRecord`、`Ledger` 保存 primitive 参数、结果、耗时和环境步数。
- `src/pyrualean/arms.py`：`ArmToolkit.execute_tool()` 记录每次代码工具的执行情况；`_observe_result()` 关联观测编号、动作版本和 ledger 索引。
- `src/pyrualean/arms.py`：`_failure_feedback()` 保存失败位置、部分执行、最后动作、状态、剩余调用数和不回滚语义。
- `src/pyrualean/play.py`：`play_episode()` 写入 prompt、工具记录、调用日志和最终结果，并分别保存模型 finish 与 backend 成功判定。
- `src/pyrualean/evaluation/metrics.py`：`EpisodeMetrics`、`load_episode_result()`、`summarize()` 汇总成功率、调用、步数、时间与 token。
- `src/pyrualean/evaluation/valen.py`：`load_samples()` 读取已保存检查，`threshold_metrics()` 使用独立标注，`evaluate()` 默认执行缓存评估。
- `src/pyrualean/stuck_recovery.py`：`StuckRecoveryConfig.record()` 记录恢复文本 hash；恢复事件另存 `stuck_recovery.jsonl`。
- 首批 P0/L0 新增 `src/pyrualean/interaction_trace.py` 和 `evaluation/latency.py`：记录 host 调用阶段、可得的模型请求时点，离线汇总现有日志；`play.py` 将日志与配置引用写入结果。

当前 acknowledged / completed_motion_calls 统计正常返回的 primitive，不能推断 `reached` 或 `holding` 达成。
code arm 的汇总约定把缺失成功结果视为失败；RPent 缺原生结果可为 unknown，比较时必须保留这一差异。
Valen 准确率与误判评估要求 sample、fact、question 匹配的独立人工标注；无标签时仍可统计分布并导出标注模板。
unknown 下降本身不证明准确率提升。
显式 replay 只重新调用 Valen，不执行机器人；问题含义改变后不能沿用旧问题标签。

## 待补齐的契约

统一事件模型的字段是设计草案，建议以现有日志增量衔接，而非丢弃已有记录。

| 字段组 | 建议内容 | 约定 |
| --- | --- | --- |
| 标识与版本 | schema_version、session_id、event_id、decision_id、code_id | 编号稳定且可关联，不仅依赖文件写入顺序 |
| 动作链 | action_id、request/receipt/stop 标识、状态与因果引用 | 保留在途、部分完成和未知；迟到回包归属可审查 |
| 观测链 | observation_id、采集时间、来源、设备版本 | 取回时间与采集时间分开，重复读取不形成独立样本 |
| 证据链 | fact_id、question_version、观测引用、有效性与失效原因 | 模型预测、人工标签、native truth 分开 |
| 决策链 | 依据的 context revision、失败尝试、预算、下一步 | 区分代码结束、动作结束、任务结束 |

保存多层 verdict 及其 authority、evidence refs、unknown reason；由独立 scoring policy 生成 benchmark 分数。
扩展动作结果未知率、取消确认延迟、恢复成功率、无进展代价、观测成本和证据失效统计。
规划、感知、执行与基础设施错误允许多因并存；没有证据时保留未归因，不根据最终失败猜测根因。
审查路径应能从一次模型决策定位源代码、相关动作、观测和判断；日志压缩不能切断这些关联。
P0 确定事件标识与语义，P1 记录动作和观测链，P2 关联恢复，P3 完善分层统计与对比实验。

### 模型出码延迟

从 P0 记录 `feedback_ready → request_started → tool_arguments_complete → code_ready → execution_started`，
首响应事件、首 token 和工具返回另记。主要指标是反馈就绪到完整代码通过校验的等待，
请求到完整参数用于定位；不把 `policy_wall_s`、首 token 或工具耗时互相替代。
时间点、时钟域和缺失值规则见 [交互延迟设计](../interaction-latency.md)。

按首次/后续判断、runtime/服务路径、实际 reasoning、任务和缓存状态分组，
报告 p50/p95、样本数、超时及无代码返回比例，同时保留成功率、返工与总任务时间。
重试、断流和 compaction 不从数据中静默排除；内部推理耗时无服务端证据时保持未归因。
已有 Codex rollout 可通过新增离线入口重建工具间等待代理指标；host trace 另记录
`tool_received`、`code_validated`、执行及反馈边界。它们不代表完整 `code_ready` 准入契约已实现，
也不补造外部客户端的供应端请求或首 token 时点。产物与使用方式见 [输出文件](../../guide/outputs.md#interaction-timing)。

## 失败交接

| 情况 | 接收方 | 处理约定 |
| --- | --- | --- |
| 动作结果或停止确认缺失 | 恢复模块 / 评估 | 保留 unknown，先对账；不能按成功动作继续统计 |
| 图像或日志丢失 | 记录模块 / Astra | 标明证据缺口及影响，当前判断不足时重新观察 |
| 模型声明与 native verdict 不一致 | 评估 | 并列保存，不用一个覆盖另一个 |
| 视觉样本没有独立标注 | 离线评估 | 可报分布与 unknown，准确率保留未定义 |
| 日志格式或版本不兼容 | 汇总入口 | 报告无法解析的记录，不静默丢弃样本 |

## 验收场景

1. 从一次部分执行失败追溯到模型代码、动作尝试、对应观测及下一次修复决策。
2. primitive 正常返回但 `reached=False`，执行 acknowledgement 与动作达成指标保持不同。
3. 模型说成功、视觉 unknown、native 失败时，三个结果都可见且有各自依据。
4. 同图重复读取不增加独立视觉样本数；变更问题后旧标注不被自动复用。
5. 未取得最终评分时，原始记录保留 unknown，评分策略可以另行将其计入失败分母。
6. 比较 cells、单次 program 与 RSI 时标明执行协议、预算和是否存在在线模型修复。

相关设计：[任务与会话](task-session.md)、[Astra 上下文](planner-context.md)、[知识与 skill](knowledge-skills.md)。
