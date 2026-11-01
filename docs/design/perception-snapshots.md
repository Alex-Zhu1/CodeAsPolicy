# 感知绑定观测快照

[整体架构](architecture.md) · [观测与感知](parts/observation-perception.md) · [推进路线](roadmap.md)

2026-11-01。本批在已有观测、动作与恢复闭环上，把 native/XPL 的
`locate / back_project / segment / camera_meta` 绑定到明确的录制快照。
历史图像可以继续分析，但历史分析与当前动作依据是两种不同用途。
随后真实任务暴露采集来源无法建立和反馈过大的集成问题；来源绑定本身不证明正常动作链可用。
对应修复已通过真实采集 smoke，并完成 Astra 整理桌面复跑；96 个快照均对齐，任务仍失败。
来源保证与目标/抓持识别质量分别评价，见[正常任务推进回归](normal-progress-regression.md)。

## 调用与职责

```python
view = robo.snapshot()  # 使用已有有效记录；需要新现场时才 refresh=True
rgb = robo.image("cam_head", step=view["step"])
calibration = robo.camera_meta("cam_head", observation_id=view["observation_id"])

mask = robo.segment("the red block", observation_id=view["observation_id"])
if mask.found and mask.centroid is not None:
    point = robo.back_project(*mask.centroid, camera=mask.camera,
                             observation_id=mask.provenance["observation_id"])
    print(point.xyz, point.provenance)

target = robo.locate("the red block", observation_id=view["observation_id"])
# 在依赖坐标的代码处重新检查；随后仍需检查 found、note 和数值/任务条件。
robo.require_current(target)
```

省略 `observation_id` 时，支持该协议的后端固定读取当前**录制快照**，不会隐式重新采集。
需要新现场时显式 `snapshot(refresh=True)`。历史引用未知、属于其他会话、记录不匹配或
传感器文件缺失时直接报错；不静默使用当前图像、深度或标定替代。

| 部件 | 输入/输出 | 负责的保证 |
| --- | --- | --- |
| 公共 RoboDojo API | 观测 ID、相机、文本/像素 → typed result + `provenance` | 保留旧调用参数；结果携带原来源，返回副本不能改写宿主账本 |
| `PerceptionMixin` | 已注册快照 → session/ID/record step | 校验回包归属、记录 perception ID 和调用时 action revision，提供使用时当前性检查 |
| NativeToolkit / Steps | 精确记录 → 冻结传感器包 | 图像、深度、标定、image_state 在一次查询中始终来自同一记录 |
| 数值感知工具 | 冻结包 → mask/坐标/几何 | 声明参与相机、标定依据、测量方法、假设和限制；不再读取现场补齐历史数据 |
| 程序与 Astra | 结果、当前性、证据 → 后续策略 | 判断识别是否正确、几何是否适用；过期时重新采集/定位，只执行剩余工作 |

`require_current` 只是检查，不固定物理世界，也不替代 `verify / requires`、执行准入或动作回执。
将 `target.center_xyz` 拆成普通 tuple 后不会自动保留依赖关系；应保留原结果，在使用前检查。
未调用检查的任意 Python 坐标计算，运行时不能反推其来自哪张图片。

## 数据与来源

每个录制目录保留 `observation.json` 和显示用 PNG，并新增：

- `sensors.json`：完整传感器包的结构、状态、图像时刻的状态与标定。
- `sensors.npz`：无损数组，保留原深度 dtype 和单位解释；加载禁用 pickle。

感知从磁盘重建独立的只读数组，不在内存累计历史图像。落盘前复制一次源包，避免环境复用
数组缓冲区改写已记录现场。这个增量增加传感器存储与压缩开销，尚未测量对 L0 等待/执行分布的影响。

完整来源记录供审计和按需查询，不要求 Astra 每轮读取全部路径、标定和传感器明细。
后续反馈修复在 rich 摘要中压缩重复 provenance，优先保留程序输出、几何限制、来源引用及异常；
结果对象和宿主审计仍保留完整来源。完整复跑没有再出现来源阻断，38 次 locate/back_project
返回时均为 current；但这不等于识别和几何准确，也没有形成速度或重复定位次数的受控对照。
复跑后 partial 摘要另有关键字段保护补丁，通过离线回放及当前代码 703 项全量回归；
尚未重跑整轮任务，不能计入本次冻结版本表现。

结果中的 `provenance` 使用 `pyrualean.perception.v1`：

| 字段 | 含义 |
| --- | --- |
| `session_id / observation_id / source_id / step / native_steps` | 会话、观测引用、源采集身份、录制索引和原生步数分别表达 |
| `perception_id / operation / action_revision` | 这次感知结果及它观察的动作版本，由宿主管理 |
| `captured_at / source_freshness / aligned / frozen` | 原来源时间、可信采集/未知来源、对齐与固定输入的状态 |
| `images[camera]` | 图像来源及 artifact，深度、标定和 image_state 的来源与文件引用 |
| `method / geometry` | 深度投影或 RGB 推断；参与相机、坐标系、单位、桌面/形状假设、不确定性 |
| `is_current_record` | 查询时是否为最新录制记录；不是现场有效性的证明 |
| `current_at_return` | 返回时检查的适用性；以后不会自动更新这个历史字段 |

`robo.perception_is_current(result)` 重新查询现场，核对宿主保存的来源、动作版本和未决动作；
不依赖调用者手填 `current_at_return`。来源变化、同原生步数的新采集、动作尝试、未知执行结果、
实时来源查询失败、外部会话或被改写的来源字段均不能通过。宿主最多保留 256 个结果引用；
过期/被淘汰的引用需重新定位。`require_current` 检查失败时抛出 `ToolError`，后续语句不会执行。

## 图像、标定与缓存

- 深度路径始终使用该记录的深度与 K/T。历史腕部图像不会套用机械臂当前位姿。
- RGB 路径优先使用快照内标定；没有矩阵时使用固定相机模型或图像时刻的腕部状态推导。
  模型依据与采集依据分别标注。已经提供但残缺、非有限或焦距非法的标定不会被模型默默替代。
- 多相机推理的参与相机都记录来源；相机参与计算不等于它与主相机的识别结果已经一致。
- 区域缓存按图像内容、来源、标定和 image_state 区分；机械臂关节相同不代表现场没有变化。
- 快照路径只在同一来源内复用高度先验。历史查询不会把曾经测得的高度悄悄变成当前测量。
  单视图、桌面交点和形状拟合仍是估计，保持 `views / note / surface` 与方法假设。

## 失败交接与覆盖

| 情况 | 行为 | 下一步 |
| --- | --- | --- |
| 旧快照仍可读取 | 使用旧包完成分析，结果保留旧来源 | 仅作历史分析；需要动作时重新观察 |
| 推理期间现场改变 | 完成原快照计算，返回 `current_at_return=false` | 程序/模型判断是否重采 |
| 会话/ID/记录错配、历史数据缺失 | 明确报错，禁止回退当前数据 | 修正引用或重新采集 |
| 标定非法或几何不足 | 报错或保留 `found` 与几何限制，不编造坐标 | 补来源、改变视角或策略 |
| 结果已过期 | 当前性检查拒绝；新 cell 不恢复旧结果效力 | 新快照、重新定位和条件核验 |
| 回调尝试运动 | 公开状态改变接口在只读作用域中拒绝 | 程序显式请求动作，再重新观察 |

严格绑定适用于 `NativeBackend`，覆盖 native/XPL 的深度和 RGB 路径。旧 RPent 等未声明
`supports_snapshot_perception` 的后端保留原参数调用，结果标 `binding=unsupported / unverified`；
显式要求观测 ID 时拒绝，不能伪装成绑定成功。LIBERO/Robocasa 尚未迁移到这套感知协议。

这批保证来源和当前性，不证明目标身份、碰撞安全、物体支撑或任务成功。独立采集/推理预算、
来源过期的时间策略、跨捕获对象关联与实机验收仍是后续契约。

## 验证入口

- `test_perception_snapshots.py`：冻结历史深度/标定、原始数组、重复来源的精确记录、错误身份和缺失文件。
- `test_perception_sources.py`：RGB 缓存、同关节新图、标定变化、多视图/单视图假设与只读查询。
- `test_perception_api.py`：typed result 兼容、来源副本、历史 camera_meta、非法标定和旧后端限制。
- `test_perception_lifecycle.py`：推理迟到、同一步换来源、动作后失效、跨会话、只读回调和新 cell 恢复。

全部使用可控传感器/控制器替身；没有调用真实模型或驱动实机。
这些是感知绑定增量的替身测试。后续[完整复跑](../../runs/organize-table-normal-progress-20261101-072852/REPORT.md)
补上真实采集与动作证据，但抓取/接近误差、held 可观测性和任务结果仍需改进；未在本轮触发的过期目标继续依靠故障测试覆盖。

本次验证：全量回归 660 项通过；随后补充精确记录和标定边界用例，以上四组专项测试
最终 53 项通过。变更相关 Python 文件通过 Ruff 检查，`git diff --check` 无异常。
