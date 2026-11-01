# 知识、说明书与 skill

[设计索引](../index.md) · [整体架构](../architecture.md) · [推进路线](../roadmap.md)

## 状态与范围

本文基于 **2026-10-31 UTC 当前工作区的静态梳理**，包括尚未提交的模块与修改。
现有说明书、监控与 stuck skill 已有实际加载路径；通用 skill manifest 和统一协议仍是设计草案。
对象是 Robot Use 的知识供给与决策辅助，Computer Use 仅提供按需加载和反馈交接的设计参照。
说明书和 skill 只提供上下文；它们不改变任务授权、执行权限或验收标准。

## 职责与边界

说明书定义稳定接口事实；skill 提供经验、验证方法和恢复思路；实时反馈说明这一刻实际发生什么。
Astra 决定具体策略，skill 不自动执行预设任务流程，也不赋予缺失能力。
运行时落实已声明的检查和预算；文本中的建议不能代替运行时保证。
模型可组合单臂动作、同步双臂动作与受支持的 VLA，选择依据仍来自任务和现场。

| 层次 | 内容 | 不应承担什么 |
| --- | --- | --- |
| 框架核心说明 | 如何执行代码、读取反馈、暂停、继续与结束 | 场景特定的动作方案 |
| 环境能力说明 | API、单位、坐标、姿态、夹爪与保证范围 | 未声明的实际能力或当前物体状态 |
| 按需 skill | 验证模式、经验、故障诊断、重新规划方法 | 任务真值、固定的抓取交接状态机 |
| 实时反馈 | 动作回执、观测、事实版本、当前预算 | 长期经验或模型推测的伪装 |

## 输入输出

| 方向 | 内容 | 当前载体 / 目标草案 |
| --- | --- | --- |
| 输入 | robot、host、primitive set 与真实方法 | `RobotAdapter`、robot class / `CapabilityManifest` 草案 |
| 输入 | 任务阶段、观测、未决条件与触发事件 | prompt、反馈 / `DecisionContext`、`HandoffReport` 草案 |
| 输出 | API reference 和操作知识 | `api_reference()`、`knowledge_text()` |
| 输出 | 当前适用的指导与 helper 说明 | packaged core、可选 guides、`strategy` 提示 |
| 输出 | 可复现的加载记录 | 已冻结 skill 文本与 hash / 通用 skill manifest 草案 |

## 现有实现

- `src/pyrualean/prompt.py`：`api_reference()` 生成实际类的接口说明；`knowledge_text()` 按机器人与 primitive set 读取打包 Markdown。
- `src/pyrualean/play.py`：`build_prompt()` 加入环境知识；监控启用时加入 planning core，stuck 只先加入触发协议。
- `src/pyrualean/supervision.py`：`planning_instructions()` 读取 `knowledge/astra_valen_supervision.md`；`config_record()` 记录配置及内容信息。
- `src/pyrualean/stuck_recovery.py`：`recovery_instructions()` 读取 `knowledge/astra_stuck_recovery.md`；`StuckRecoveryConfig` 保存文本、轮次和 hash。
- `src/pyrualean/arms.py`：`ArmToolkit._finish()` 在确认 stuck 且允许继续时下发恢复正文、状态和可用图像。
- `src/pyrualean/play.py`：`play_episode()` 保存 `supervision_skill.md`、`stuck_recovery_skill.md`；可选 guides 被复制到工作区。
- `skills/astra-valen-supervision/SKILL.md`、`skills/astra-stuck-recovery/SKILL.md`：提供较完整的指导及参考示例，供开发和显式加载。

仓库完整 SKILL 与 `src/pyrualean/knowledge/` 中的 runtime core 是不同文件。
修改完整 SKILL 不会自动改变下一次运行实际注入的 packaged core；需要核对真实 `prompt.txt` 和冻结文本。
当前不会自动发现任意 skill 目录；不要把“文件存在”写成“模型已经读取”。
CLI cells 默认配置 stuck recovery；一般库调用是否配置取决于调用方，不能笼统认为所有入口都启用。
监控启用时 planning core 初始注入；普通暂停或 unknown 不触发 stuck core，显式 `finish(stuck)` 才触发。

## 待补齐的契约

以下 skill manifest 字段是文档草案，不是现有可调用接口。

| 字段组 | 建议内容 | 约定 |
| --- | --- | --- |
| 身份 | id、version、content_hash、source | 实际加载内容可追溯，不能只记录目录名 |
| 适用性 | 设备能力、执行模式、观测要求、接口版本 | 与 `CapabilityManifest` 匹配，不适用时明确拒绝 |
| 触发 | initial、explicit_request、confirmed_stuck 等 | 与普通暂停、错误和终止事件区分 |
| 输入 | 所需事实、观测引用、预算、当前阻塞 | 缺失项显式报告，不由 skill 文本补造 |
| 输出 | 建议、可选程序模式、验证方法、未决条件 | 不改变事实真值、不改任务验收、不退款 |

确定完整 SKILL 与 packaged core 的维护关系：使用单一来源生成，或明确版本对应并校验一致性。
将稳定接口事实、经验建议与场景示例分别标注，保留经验的适用条件和限制。
动态加载应把触发依据、选中版本和实际投递位置写入记录，便于比较运行结果。
skill 可以帮助 Astra 识别不必要假设，但不能移除真实任务约束或绕过声明的有效依赖。
P0 明确知识与事实的边界，P2 固化恢复触发协议，P3 再扩展按需选择与版本管理。

## 失败交接

| 情况 | 接收方 | 处理约定 |
| --- | --- | --- |
| 能力或 API 不兼容 | 上下文模块 / Astra | 不注入误导性方案，报告缺失条件并选择其它方法 |
| 文档、core 与实际 API 有冲突 | 知识维护与运行入口 | 以实际能力和执行结果为准，记录版本冲突 |
| skill 所需观测缺失 | Astra / 观测模块 | 获取有效证据或换策略，不把经验当作当前事实 |
| 已确认 stuck 但仍有预算 | 恢复模块 / Astra | 在同一现场下发 guidance，保留动作、namespace 和预算 |
| 会话已终止或恢复轮次耗尽 | 会话管理 | 返回最终状态，加载 skill 不能重新开启任务 |

## 验收场景

1. 只修改仓库 SKILL 时，运行记录能显示实际使用的 packaged core 版本，避免误判变更已生效。
2. 普通 Valen 暂停不加载 stuck core；确认 stuck 后才下发，并明确 `episode_finished=false`。
3. 关闭监控但保留 stuck 配置，恢复文本不会要求调用当前未绑定的监督 helper。
4. 同一 skill 面向不同机器人时先匹配坐标、接口和观测能力，不能盲目复制动作参数。
5. skill 建议的检查与任务无因果关系时，Astra 可以解释后调整；真实约束继续保留。
6. 离线 RSI 不因通用 skill 机制获得在线模型修复，实验协议保持可比较。

相关设计：[任务与会话](task-session.md)、[Astra 上下文](planner-context.md)、[记录与评估](records-evaluation.md)。
