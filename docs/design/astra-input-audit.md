# Astra 输入审计：原版与当前的提示、接口和反馈

[设计索引](index.md) · [推进路线](roadmap.md) · [本轮任务报告](../../runs/organize-table-handoff-rerun-20261101-093121/REPORT.md)

2026-11-01 UTC。以下保留修复前的只读审计；审计本身未改产品或启动模型。
后续已按用户要求实施[输入修复](astra-input-fix.md)，验证与同配置复跑单独记录，不改写本次历史证据。

## 判断

**输入层存在需要先清理的问题，不能仅将差异归为“Astra 这次写了不同代码”。** 默认关闭监督后仍注入视觉许可说明；正常回包大部分是重复协议字段；异常反馈发生实际交付截断；新增 assert 指引与按需图像组合，改变了失败后能看到现场的时机。

这些问题有实际文本和执行证据，但不能据单次轨迹认定某一条导致得分 75→50。需要区分“说明确实矛盾”“输入变化与代码/反馈形态变化均已观察到”和“最终任务质量的因果归属”。本轮等待比原版短，也不能根据文字变多反推等待必然变慢。

## 对照对象与口径

- 原版：`runs/organize-table-astra-20261030`，score=75。
- 当前：`runs/organize-table-handoff-rerun-20261101-093121`，score=50。
- 两轮模型均为 Astra/high，on-demand 图像、pure 反馈，任务/种子及主要预算对齐，Valen 与专用 stuck 恢复未启用。
- prompt 使用当时归档的 `episode/prompt.txt`；反馈使用模型实际收到的 `function_call_output`，再与工具原始回包比较。宿主 ledger 和 tool_turns 不能直接代替模型可见输入。
- 只读取公开代码、工具输入输出与来源数据，不分析模型私有 reasoning。原版源码没有完整冻结，不能把当前 Git 版本冒充原版实现。

| 输入规模 | 原版 | 当前 | 差异 |
| --- | ---: | ---: | ---: |
| 控制、评分与执行说明 | 2,559 字符 | 7,551 字符 | +4,992 |
| API reference | 22,577 | 32,275 | +9,698 |
| Operating knowledge | 19,270 | 22,402 | +3,132 |
| Task card | 1,061 | 1,061 | 完全相同 |
| 完整 prompt | 45,467 | 63,289 | **+39.20%** |
| 实际交付反馈累计 | 46,232 | 195,102 | **4.22 倍** |
| 每 Python 回包平均文字 | 1,390 | 4,399 | **3.16 倍** |
| stdout 字段累计 | 41,153 | 46,469 | +12.9% |

字符不等于 token。stdout 字段按 JSON 编码值统计；反馈累计包含最后 finish 与客户端时间包装。单 cell 指标排除 finish。完整分解见[输入统计](../../runs/astra-input-audit-20261101/input-comparison.json)和[反馈统计](../../runs/astra-input-audit-20261101/feedback-summary.json)。

## 1. 默认 off 输入仍残留视觉许可要求

当前 prompt 的 `Reusable local loops` 无条件写入：

- 抓取验证使用 `strategy.verify_grasp`，需要新鲜 object-held 证据和正向比较。
- `Only verified permits dependent work`。
- `All callbacks still need watch contracts`，示例引用 `evidence()`。

但是这轮监督关闭，没有相应 watch/evidence 接口说明。helper 的局部返回约束，被放在全局抓取指导的位置，适用范围没有限定清楚。之前的 Valen 职责修复没有清理完共享 helper 提示。

**可能影响：**让模型把可选抓取核验理解成普遍许可，或为不存在的监督接口组织程序。

**本轮边界：**`strategy / watch / evidence` 实际调用均为 0；鼠标直接检查夹爪后抬升并搬运。没有证据这段文字在本轮形成硬门控，也不能用它解释全部失分。

证据：[当前 prompt 第78行](../../runs/organize-table-handoff-rerun-20261101-093121/episode/prompt.txt#L78)；源码位置 `src/pyrualean/strategies.py:255–273`。更详细的逐行对照见[提示审计](../../runs/astra-input-audit-20261101/prompt-findings.md)。

## 2. pure 反馈的重心偏向协议，且已有真实截断

当前 `handoff / action_receipts / observation` 三个自动字段合计 127,270 字符，约占全部反馈的 **65.2%**，尚未算 recovery。相同动作在 last_action、receipt 列表和 observation.action_ids 中重复描述；每张图再携带完整来源字段。每 cell 的 stdout 均值反而从 1,247 降到 1,056 字符。

此外，43/44 个 Python 回包的 `allowed_next=check_prerequisites`、`new_motion_allowed=null`，42 个提示旧目标可能过期；本轮所有 unresolved_actions 都为空。这些是正常路径默认措辞，不能当成 43 次异常或硬阻塞。它们可能让模型把检查协议、重新定位视为默认工作，而非围绕当前目标误差组织下一步。

有两处已实际发生的输出问题：

| 场景 | 事实 | 影响范围 |
| --- | --- | --- |
| 首个 cell 打印定位结果 | provenance 扩大 repr，触发 stdout 上限，省略 478 字符；下一 cell Astra 自写 `brief()` 过滤字段 | 截在最后一个 stand 的 provenance 尾部，几何数值仍在；新增输出负担确实需要模型处理 |
| cell 30 的 partial + assertion | 工具原始 JSON 14,106 字符，实际交付 JSON 11,988 字符；出现字面截断标记和损坏字段名，未带 output_truncated | recovery.last_motion 重新带入完整 native receipts；关键位置误差仍在，不能声称本次因此失分 |

第二处说明“宿主已摘要”不等于“模型收到完整可用摘要”。现有 20,000 字符宿主阈值没有覆盖实际交付链路，本例被下游非结构化裁剪；档案不足以确定具体中间层的固定上限。

这应优先作为现有反馈契约问题处理，保留审计记录不等于每轮都把完整审计放进模型上下文。详见[反馈审计](../../runs/astra-input-audit-20261101/feedback-findings.md)与[公开工具投影](../../runs/astra-input-audit-20261101/feedback-public.json)。

## 3. assert 用法与按需图像改变了交接时机

原版“同 cell 多步执行并做程序内检查”的原句仍在；当前额外在 API 前强调 short phases 和 `assert`。生成代码中断言从 **0 增至16**，Python cells 从33增至44。原版常通过 `move_checked()` 返回 bool 再做 if 分支，当前更多在不满足 reached 时抛出断言。

不能将所有新增 cell 归因于此提示，也不应取消必要的依赖检查。不过存在可证的交接差异：

| 路径 | 未到位之后的行为 | Astra 获得的图像 |
| --- | --- | ---: |
| 原版 cell 16 | 跳过抓取分支，继续执行末尾 show | 2 张 |
| 当前 cell 18 | assert 提前结束，末尾 show 被跳过 | 0 张 |
| 当前 cell 30 | assert 提前结束，末尾 show 被跳过 | 0 张 |

当前 cell 19 随即发纠偏动作；cell 31 虽在 move 前写了 show，但 show 只登记图像请求，cell 结束才向 Astra 返回，不能在同一 cell 中形成“模型先看图，再决定动作”。因此两次都是模型读到失败数值后，没有新图就生成了纠偏动作。

数值状态有时足够支持纠偏，两次后续动作也继续了工作；这不证明每次失败都必须补图。问题在于：**提示鼓励的退出方式破坏了程序自己安排的观察机会。** 需要先澄清普通可处理结果、必要的模型交接与图像返回时机，不应直接新增强制核验层。

代码证据：[原版 cell16](../../runs/organize-table-astra-20261030/episode/workspace/cell_016.py)、[当前 cell18](../../runs/organize-table-handoff-rerun-20261101-093121/episode/workspace/cell_018.py)、[当前 cell30](../../runs/organize-table-handoff-rerun-20261101-093121/episode/workspace/cell_030.py)、[当前 cell31](../../runs/organize-table-handoff-rerun-20261101-093121/episode/workspace/cell_031.py)。

## 4. 来源检查与任务有效性之间的重点失衡

当前新增“动作尝试或新 capture 使旧结果失去当前性”的说明，同时保留“机械臂 home 时把对象测一遍”的原建议。初始测量可作为何种规划记忆、何时必须重新测量，没有讲清楚。实际 locate 从22增至44，但其中含真实返工，不能都归因于 freshness 提示。

更具体的代码问题出现在 cell28：locate 返回 `found=False / center_xyz=None`，`require_current(clock)` 正常通过，随后索引空坐标报错；下一 cell 才补 found 检查。来源检查正确地证明了“这是当前帧的未找到结果”，并没有证明“找到了目标”。

这个例子说明：代码加入了新的来源检查，却仍漏掉最基本的任务条件。API 已说明仍需检查 found/note/geometry，不能诊断成 require_current 实现错误；也没有证据本轮被 stale 门控卡住——7次显式检查均未因过期拒绝。

正常反馈同时存在 completed 与 reached、记录 step 与原生 step 的不同语义，也增加了解读负担。但当前程序仍使用 reached，不能认定它把 completed 当成到位。应先统一已有说明和摘要，让几何与任务结果突出，而非再增加新的许可字段。

证据：[当前 cell28](../../runs/organize-table-handoff-rerun-20261101-093121/episode/workspace/cell_028.py)、[cell29](../../runs/organize-table-handoff-rerun-20261101-093121/episode/workspace/cell_029.py)、[API 审计](../../runs/astra-input-audit-20261101/api-findings.md)。

## 5. 感知确实改了，但目前没有证据它造成了退步

原版承诺同一对象文本的双视图高度可以用于后续较差视图；当前只允许同一 capture 内复用。它是实际能力语义变化，不能只当作新增 provenance 文案，可能影响遮挡后的稳定性。

但本次对照有重要反证：

- 初始 keyboard 几何完全一致；figurine / clock 的中心差异约0.1毫米，top_z一致。
- 原版公开回包没有一次“使用此前测量高度”的记录；当前有几何来源的35次 locate 全部 `height_prior_used=false`。
- 后续场景已因不同动作而分叉，不能把不同视角和遮挡下的测量直接作因果对照。

因此，跨 capture 记忆可作为后续离线诊断项，目前不应回退来源绑定或静默恢复旧高度缓存。旧图与新标定混用的正确性问题依然需要避免。

## 已排除与未证明的解释

- **任务目标没变：** Task card 和评分段完全一致。
- **基础接口没变：** move_to / set_gripper 的签名、默认值、说明，以及 Move / Gripper / Home 字段块逐字一致；不能据此保证全部驱动内部实现相同。
- **机械知识没被删：** 主要坐标、可达性、姿态、抓持、抬升、释放支撑说明保留，变量持久化和连续动作建议也保留。
- **本轮未使用新增大块能力：** strategy、move_both、observe 调用均为0；它们增加输入篇幅，但未形成实际执行门控。
- **未发现本轮交付旧图冒充新图：** 当前58/58张工具图片与其来源指向的归档帧哈希一致。
- **没有普遍丢失关键运动数值：** reached、距离误差、最终位置、夹爪宽度仍可读，包括被截断的cell30。

原版对雕像采用两视图细部三角化，当前未采用同样策略，但相机标定接口仍可用。输入组织变化与模型单次采样差异都是候选因素，尚不能认定哪一项独自造成25分损失。

## 后续最小处理顺序

本次只提出问题和删减方向，不实施修改：

1. **清掉不适用的现有说明。** 默认off输入不应继续出现无条件watch或普遍视觉许可措辞；可选helper的条件应限于它自身。
2. **修正现有反馈的交付与重复。** 优先处理错误路径完整审计重新进入回包、下游截断、默认repr膨胀；保留关键数值、必要来源和未决状态，完整明细仍在日志。
3. **把正常结果分支放回主线。** 说明中明确可用if/有限纠偏处理普通未到位；需要模型判断时，确保计划中的证据能随交接返回。不是简单去掉断言，也不是每次失败强制增加图片。
4. **明确来源检查的边界。** 当前性、找到对象、几何可信和任务完成各自代表什么；不让新增检查替代原有任务判断。感知高度变化单独诊断。

先清理这些已有输入，再用固定配置判断动作质量是否恢复。此阶段没有理由继续叠加验证器、审批入口或新的恢复状态机。

复算脚本：[audit_inputs.py](../../runs/astra-input-audit-20261101/audit_inputs.py)、[audit_feedback.py](../../runs/astra-input-audit-20261101/audit_feedback.py)。完整prompt差异见[prompt.diff](../../runs/astra-input-audit-20261101/prompt.diff)。
