# 观测、动作与恢复闭环：实施记录

[整体架构](architecture.md) · [推进路线](roadmap.md)

2026-11-01。本批实现一条贯通的协议链路，范围是 native host / XPolicyLab Bridge、
code cells 与已有视觉监督。测试使用仿真夹具和故障注入，不代表实机停止或任务成功验收。
后续零动作回归已修复并完成真实仿真 smoke 与模型复跑：93 个高层动作、790 原生步，
包括两次已知 partial 交接；任务仍失败。见[正常任务推进回归](normal-progress-regression.md)。
本轮没有触发 unknown 对账，不能把正常执行恢复提升为全部恢复协议或任务质量已通过。

## 当前调用方式

保留 `robo.observe()` 的双臂位姿对象，另加 `robo.snapshot(refresh=False)` 返回脱离内部存储的观测字典。
`refresh=True` 请求物理只读采集；不发 hold、夹爪或运动命令。不同采集 ID 不保证统计独立。

```python
# 示例有明确释放依赖；不是每个 cell / 动作都必须执行的前缀。
watch("release", {"supported": "Is the object supported by the table?"},
      requires={"supported": "yes"})
source = robo.snapshot()  # 需要新现场证据时显式 refresh=True
verify(source["observation_id"])
robo.set_gripper("right", robo.OPEN)
```

`verify` 在启用 supervisor 的 cells 中提供，与 `watch`、`evidence` 一样由宿主绑定。
它检查来源、有效性和只读边界；`requires` 决定动作门槛，`clear` 只表示未触发监督规则。
核验过程不能发出物理动作。每 cell 的核验次数独立计数，当前上限复用 `MonitorConfig.max_actions`。
默认不启用视觉监督；显式启用后，没有额外前置依赖或未决动作时，正常动作可在 `watch` 下继续，动作后监督仍执行。
普通 critical unknown 只触发一次复查，下一 cell 可按 Astra 判断继续；`requires` 与执行 unknown 的阻断仍保留。
详见[职责修复](valen-role-and-value.md)。
显式 verify 用于需要补证据的依赖；已有当前来源不因进入新 cell 就必须重新采集或重复视觉推理。

如果动作返回 unknown，原 cell 停止，新动作保持阻塞：

```python
# 新 cell：只查询这次动作，不重发动作
receipt = robo.reconcile()  # 也可传反馈中明确的 action_id
print(receipt["status"], receipt["effect_known"], receipt["execution_complete"])
```

只有精确回执确认效果和执行边界后，运行时才解除 unknown 门槛。随后在新 cell 重新声明阶段、
读取/刷新现场并核验必要条件，只提交剩余动作。重新看图、Valen 的 yes、新动作 ID、
修改 Python 变量、捕获异常或更换 cell 均不能自行消除待对账动作。
没有对账能力的后端保持受限，不能猜测动作是否发生。

已知效果的 recoverable partial 仍允许已有数值规则做有限纠偏；没有处理规则时交回 Astra。
迟到原生完成回执只将中断的高层动作结算为 partial，不证明原高层目标到位。

## 身份与持久化

| 层 | 已实现字段与接口 | 保证 |
| --- | --- | --- |
| 源观测 | `session_id / observation_id / source_id / captured_at / source_freshness / aligned` | 采集时间与落盘时间分离，来源缺失保留 unverified |
| 录制快照 | `step / native_steps / action_id / images`；每步 `observation.json` | 状态、image_state、图片、标定按录制步绑定；重读不是新来源 |
| 代码调用 | host session、turn、可用的 `local_call_id / provider_call_id` | 单次代码内每个动作保留调用上下文 |
| 高层动作 | `action_id / status / effect_known / execution_complete / observation_id` | `robo.action_receipt()` 返回完整脱离内部存储的回执 |
| 原生派发 | native action ID、parent action ID、session | 只接受精确回包；迟到、重复、错配与旧会话不完成别的动作 |
| 证据 | source、问题版本、失效动作 ID、跨 cell unknown 计数 | 同来源重问不新增确认，不改变已记录的那次判断 |
| 交接 | `handoff / action_receipts`，保存在工具反馈和 turns | pure 模式和压缩反馈保留待对账 ID 与后续限制 |

反馈顶层 `observation.observation_id` 仍是**这次反馈**的编号，源编号放在
`source_observation_id`；两者不能混用。`state(step=旧值)` 不再覆盖当前状态缓存。
实时来源查询与录制 metadata 分离，同 native step 的外部新 capture 也会使旧核验过期。
派发前再次比较 source/native step；不匹配时返回零动作 rejected 回执。

## partial、unknown 与取消

| 状态 | 运行时行为 | 下一责任方 |
| --- | --- | --- |
| rejected，确认零执行 | 保留拒绝原因和当前现场 | 程序调整或 Astra |
| completed，执行/效果已知 | 检查到位与物体关系；不等于任务完成 | 程序与验证器 |
| partial，执行/效果已知 | 从实际终点有限纠偏或交回 | 程序或 Astra |
| unknown / running / 不完整回执 | 保留 ID，阻止新动作，允许观察和对账 | 执行系统 |
| 对账完成但证据已失效 | 仍需满足阶段 `requires` | 验证器与程序 |
| episode 预算或停止原因已生效 | 对账不重置 episode，不重开动作 | 会话终止与记录 |

取消回执区分 `requested / confirmed / scope`；当前确认范围是原生派发，
`physical_stop_confirmed=None`。清空尚未领取的请求可证明零派发；已经领取的请求必须等精确回执。
代码结束、动作完成、任务完成始终分开。pending action 存在时拒绝 `finish(status="success")`。

## 覆盖和后续契约

- 新版两份 XPL deploy 提供来源协议、只读 observe 控制消息以及终局末帧/回执。后续修复增加逐 render product 完成事件、annotator 图同步与采集前后计数证据；smoke 与整轮任务的 96 个对齐快照已验证当前真实路径。确实滞后或无法证明对齐的图片仍不能用于严格前置核验。
- 无来源协议的 native 环境保留 unverified 记录及旧动作后核验兼容路径；独立 `verify` 拒绝这类输入。它们不享有新版来源保证。
- 旧 backend 正常同步返回仅标 acknowledged，不伪装成精确远端回执；异常后的未决动作仍被拦截。program 也返回 unknown 暂停，但独立视觉 helper 仍限 cells。
- 后续增量已将 native/XPL `locate/back_project/segment/camera_meta` 绑定到 snapshot，
  增加使用时过期检查；覆盖与限制见[感知快照实施](perception-snapshots.md)。旧后端不自动获得这项保证。
- 采集/推理独立预算、跨所有 runtime 的完整 Handoff/BudgetSnapshot、实机停止确认及多任务质量回归仍待补齐。Python 只读约束是公开能力边界，不是恶意代码安全沙箱。

采集能力不足或持续来源错误由 host 修复；有依据的只读刷新仍不能恢复时，给出明确能力失败。
不能把无法满足的采集条件不断交给 Astra 重试，也不能用关闭来源检查恢复执行。

完整复跑后另修复了 partial rich 摘要丢失关键字段的问题：离线回放保留完整动作/来源 ID、
误差和实际终点，补丁后的代码全量回归 703 项通过。该补丁未包含在本次冻结源码运行中，尚未完成新整轮验收；
其测试证据与[真实运行报告](../../runs/organize-table-normal-progress-20261101-072852/REPORT.md)分别记录。

## 回归入口

`tests/test_observation_transport.py` 覆盖真实 Bridge 与两份 deploy 的只读刷新、回包归属、
取消、迟到对账、终局图、旧图和不完整协议；`tests/test_interaction_contracts.py` 覆盖
首次核验、unknown 暂停、错配回执、新 cell 对账/补证据和剩余动作、历史缓存及压缩反馈。
`tests/test_supervision.py`、`test_evidence.py`、`test_coordination.py` 保持监督、证据和双臂执行回归。

本批全量回归结果为 **612 passed**（允许测试用本地 MCP/HTTP 监听，模型与机器人均为测试替身）。
随后补充的真实 NativeBackend/Bridge/监督器/cells 串联用例通过，链路文件共 **7 passed**；
最终运行时调整另由 **146 passed、1 deselected** 的相关测试覆盖，跳过项是已在全量运行中通过的本地 HTTP 测试。
