# 仓库审查与近期实施记录（2026-11-02）

[设计索引](index.md) · [推进路线](roadmap.md) · [历史任务对照](original-vs-current.md)

## 当前结论

最近的工作已从监督与恢复扩展到观测来源、动作回执、反馈压缩、机器人约定，以及时间参数化轨迹。
**最新已归档的 `organize_table` 约定修复复跑为 score=75、1528.16 s，环境成功仍为 false。**
这恢复了原始单轮的得分水平，但不代表完整任务成功、稳定提速或双臂协作通过验收。
工作区新增的时间参数化轨迹和任务卡来源说明尚未被这份实测覆盖。

当前优先级是验收新执行路径，再复盘抓持、放置和抽屉失败；P0–P3、L0–L3 仍按
[路线中的阶段条件](roadmap.md)分别判断，不能因接口存在或离线测试通过就整体标记完成。

## 审查范围与版本边界

- 审查日期：2026-11-02，Asia/Singapore；收尾增量观察截至约12:13（04:13 UTC）。实验目录中的时间后缀保留原始命名。
- Git 基点：`513755a6724945178072e616309f35cb4c00707e`；重点覆盖 `86be251`、`d5a61c9`、`513755a` 和审查开始时的未提交修改。
- 全仓按模块检查源码、CLI/runtime 入口、host、部署、测试、文档、评估和 CI；近期 diff 及其调用链重点阅读。
  RSI 固定程序按接入方式、预算和评估口径复核，没有重新执行全部 54 个任务，也不是对每份历史程序的逐行行为证明。
- 实测结论来自本地归档的 `result.json`、`comparison.json`、验证清单与报告；本次没有启动模型、Isaac 或机器人实验。
- 审查只更新项目管理与设计文档；保留已有代码、测试和未提交改动。
- `.ua/` 目前有中间产物，但缺少 `knowledge-graph.json`，因此未用图谱推导影响范围或生成 diff overlay。

### 全仓模块复核

| 范围 | 当前实现与本次关注点 | 覆盖边界 |
| --- | --- | --- |
| `cli.py`、`play.py`、`generate.py`、`runner.py` | 生成、在线 cells/program、预算、任务卡与结果归档 | Codex / OpenAI / Claude 的采集边界不同；供应端 TTFT 不能由客户端等待反推 |
| `arms.py`、`execution.py`、`sandbox.py` | 完整 cell 编译、持久 namespace、超时/输出限制、失败回包 | Python 执行限制不等于进程隔离或硬件停止 |
| `_robot.py`、`interaction.py`、`perception.py` | 动作身份、快照、感知来源、pending 对账与旧证据检查 | native/XPL 新协议不能推广到所有旧 host |
| `supervision.py`、`evidence.py`、`recovery.py`、`stuck_recovery.py`、`strategies.py` | 可选 Valen、显式依赖、普通复查、新 cell 接续与程序局部纠偏 | 默认 off；显式 pause 的新接续尚缺匹配配置的任务收益证据 |
| 六个 robot adapter、`hosts/` | LIBERO、RoboCasa、RoboTwin 与 RoboDojo 的状态/动作/感知适配 | 本轮执行增量集中于 RoboDojo，其他机器人不获得时间参数化能力 |
| `coordination.py`、`motion.py`、RoboDojo 执行模块 | 同步规划、实际终点、路径约束、C2 关节轨迹与回执 | 无实测速度保证、臂间/物体碰撞保证或力控 |
| `integrations/xpolicylab/` | 在线 Lean 与固定程序 RSI，共享 Bridge/robot API；两份部署适配器都接入子步协议 | 修改默认运动路径同时影响在线代码和已有 RSI 程序 |
| `evaluation/`、`examples/`、`tests/`、`docs/`、`.github/` | 分层指标、监督批跑、回归覆盖、实验解释与文档导航 | 历史排行榜、单 seed 批跑和当前单任务诊断必须分开 |

## 最近修改台账

| 批次 | 已实现内容 | 当前状态与证据 |
| --- | --- | --- |
| `86be251`：闭环恢复与视觉监督 | watch/视觉审查、失败交接、stuck 恢复、监督批跑与回放 | 已提交；后续语义在 `d5a61c9` 中继续修正；旧批跑不代表新版本已验收 |
| `d5a61c9`：代码交互稳定化 | 快照绑定、来源检查、action ID/partial/unknown、独立 verify、双臂同步、运行限制、交互计时与离线汇总 | 已提交；协议与故障测试覆盖，P0/P1/P2 均仍有未完成契约 |
| `d5a61c9`：Astra 输入与 Valen 职责 | Valen 默认可选，普通复查交回 Astra，保留显式依赖；压缩反馈、清理矛盾提示、失败后交付已有图像 | 输入清理轮 score=50；反馈减少 45.24% 且无截断，但该轮任务性能未改善 |
| 短接触动作说明，后纳入 `513755a` | 接触推动分段、保持朝向退出、减少不必要 home | 独立知识修改实验 score 50→75；不涉及连续速度/加速度控制 |
| `513755a`：约定与单臂姿态 | 统一 TCP/腕部、世界坐标、四元数/角度、夹爪命令/测量、首步生效、home 与 blocked；纯位置目标不选半转候选 | 已提交；归档 108 项相关测试；同配置复跑 score=75、1528.16 s，未完成任务；该轮没有 `motion_plan` 调用 |
| 未提交：时间参数化执行 | `MotionProfile`；C2 五次关节样条与解析导数界；`move_both(via=...)`；XPL 子步位置/速度传输；`Move` 新增状态、计划路径和停止原因 | 理想关节与真实 adapter 展开测试覆盖；尚无当前执行路径的完整物理仿真任务复跑 |
| 未提交：任务卡与说明书 | `TaskCard.sources`；共享 `build_card` 保留原始指令，缺失测量标 unknown、预算回退标来源；启动能力快照和四份 RoboDojo 知识同步；审查期间进一步新增 `initial_state` / `provenance` 与 compact 渲染 | 首轮覆盖 RPent、XPL RGB-D/RGB 任务卡及旧 JSON 兼容；期间新增部分单独复核，不把先前777项测试套用于后续源码；不是统一 `TaskSpec` / `CapabilityManifest` 已完成 |
| 收尾时并行开发：执行计划记录 | 新增 `execution_plan.py`，Astra `plan()` 保存当前目标、下一动作、完成条件、证据文字和状态，独立保存最新记录与修订历史；`arms.py` / `play.py` 正在接入 | 仅完成静态增量登记，出现在下述定向测试之后；不属于777项或84项通过的覆盖版本，不标记已验收 |

未提交增量涉及 `motion.py`、`hosts/robodojo_trajectory.py`、`hosts/robodojo_card.py` 和两份新增测试文件；
这些文件在审查开始时均未跟踪。提交该功能时应与引用它们的已有文件一起纳入，避免只提交 tracked diff。

审查期间任务卡相关源码仍在并行编辑：`initial_state` 保存未舍入的公开机器人测量，
`provenance` 保存观测标识、时钟、图像/状态/指令摘要和创建时预算；`card.json` 保留完整记录，
`play.build_prompt` / `render_prompt` 改用 `render(compact=True)` 提供精简启动视图。
这些是文档整理期间观察到的额外增量，测试版本边界见本页末尾。
此后还观察到 `ExecutionPlan` 接入：计划明确标为 `planner_note`，不替代原始任务、核验证据、
动作权限或 finish。该功能仍在并行修改，应完成接口与恢复/反馈回归后单独更新验收记录。

### 时间参数化能力的实际范围

客户端只有声明并满足 `robodojo_joint_substeps_v1`、仿真时间步、子步数和位置/速度目标支持时，
才开放时间参数化执行；缺少协议的 host 保留原几何执行，显式 `motion` 或非空 `via` 请求拒绝。
**支持协议时，普通 `move_to` 和 `move_both` 也默认进入新路径，不需要调用者显式传 `motion`。**

导数限制约束的是计划关节指令，单位为 rad/s、rad/s²、rad/s³；不是末端 TCP 速度、真实跟踪速度或接触力保证。
连续性限定于单次调用的仿真时间，调用首尾静止；单独调用、暂停和模型/RPC 等待之间没有连续性承诺。
`home`、`set_gripper` 的原生路径未统一改为此样条，不能把整轮录像描述为已经实现连续平滑运动。
RPent 没有因 Python 层新增接口就自动获得 XPL 客户端的子步能力。

## 实验结果：补齐成功与失败记录

下表均为 `organize_table` / layout 0 / eval seed 0 / `robodojo_xpl_rgb` 的单轮记录。
所有轮次 `environment_success=false`；score 是原生聚合评分，不是成功率。
“高层调用”取 `stateful_calls`，其中同步 checkpoint 也计为调用，不能直接当完整动作数。

| 本地运行目录（`runs/` 下） | 版本/目的 | Score | 总耗时 s | 原生步 | 高层调用 |
| --- | --- | ---: | ---: | ---: | ---: |
| `organize-table-astra-20261030` | 原始参考 | 75 | 1636.28 | 965 | 117 |
| `organize-table-closed-loop-20261101-051031` | 采集/前置核验回归 | — | 316.95 | 0 | 0 |
| `organize-table-normal-progress-20261101-072852` | 恢复正常动作后的闭环版本 | 50 | 2429.17 | 790 | 93 |
| `organize-table-handoff-rerun-20261101-093121` | 职责修复、配置对齐 | 50 | 1232.82 | 1000 | 99 |
| `organize-table-input-cleanup-20261101-115354` | 输入与反馈清理 | 50 | 1646.02 | 920 | 97 |
| `organize-table-motion-tune-20261101-154117` | 短接触动作与回撤说明 | 75 | 1590.30 | 941 | 124 |
| `organize-table-dual-prompt-20261102-004730` | 双臂提示实验，后续回退 | 50 | 1411.87 | 963 | 117 |
| `organize-table-conventions-20261102-023049` | 约定与单臂姿态修复 | 75 | 1528.16 | 870 | 114 |

- **短接触动作实验**：Python 累计等待 999.67→931.31 s、工具往返 518.41→546.37 s；
  任务得分提高，但不能把全部耗时变化归因于动作说明。外层 driver 收尾退出 143，episode 已完成并归档；
  后处理补齐统计，异常应随样本保留。
- **双臂提示实验**：实际进入 `motion_plan` 3 次，1 次执行 4 个原生步后因 measured table clearance 停止，
  2 次规划拒绝，完整同步动作 0 次。更短耗时伴随更低得分，不能算协作收益；该提示实验未作为最终接受的基线。
- **约定修复实验**：相比 1590.30 s 基线减少 62.14 s，但只是一组单次比较；home 调用仍为 9，
  home 原生步反而 122→144。6 次未到位反馈、结束还剩 130 步，Astra 以 stuck 结束。
  两轮都未观察到纯位置末端半转，因此单测证明了修复分支，不证明它解释了该轮得分或耗时。
- 后三轮显式配置均沿用 high / cells / no-vla / on-demand / pure / monitor off / stuck recovery off，
  调用预算 100、policy 5400 s、原生 1000 步。不同源码仍不能视为固定版本重复采样。
- 上述三轮新记录均未覆盖当前未提交的样条、子步协议或来源任务卡。

证据文件在各目录的 `episode/result.json`、`comparison.json` 与 `REPORT.md` / `REPORT.txt`；
约定修复另有 `orientation-audit.json`、`partial-moves.json`、`source-validation.json` 和 `validation.json`。
`runs/*` 由 Git 忽略，本页保留关键数值供仓库读者查阅；原始日志、图片和视频仍需本地归档才能完整复算。

### 其他评估不得混用

- `runs/robodojo-54-astra-valen4b-20261031/REPORT.md`：54 个任务各一轮、28 成功、26 失败；
  17 个启动失败样本补齐资产后重试，已完成 episode 没有择优重跑。属于旧监督配置，不是当前样条的回归结果。
- `integrations/xpolicylab/evaluation/robodojo_summary_rsi.md`：固定 RSI 程序、eval seed 0 的 2100 episodes，
  Avg SR=82.82%，Avg Score=87.16；开发也使用 seed 0。文件另记录未用于开发的 seed 2 结果。
  这是历史固定程序评估，既不能与 28/54 直接比较，也不能证明修改默认执行器后仍维持相同成绩。
- L0 `insert_key` 仍只有两份完整样本和两份中断记录；没有达到三份完整样本的原定验收条件。

## Review 发现与后续任务

以下优先级表示项目推进顺序；“验收缺口”与已复现代码错误分开记录。

| ID / 优先级 | 发现与依据 | 状态 | 下一责任模块与完成条件 |
| --- | --- | --- | --- |
| R1 / P1 | 新客户端自动切换单/双臂默认轨迹；历史 75 分和 RSI 结果均来自旧执行路径 | 验收缺口 | 执行控制 + 集成：冻结源码与客户端版本，先核对真实子步、跟踪误差、原生步预算、取消/partial，再做同配置任务与代表 RSI 程序回归；记录实测运动和任务质量 |
| R2 / P1 | 完整任务仍失败；最新约定轮无同步规划，双臂提示轮无完成的同步动作 | 已确认未达目标 | Astra/能力说明 + 评估：针对抓持、放置、抽屉和双臂几何失败提出单一假设；固定版本重复采样，并列 score/success、等待、工具成本和无进展重试 |
| R3 / P2 | `.github/workflows/docs.yml` 是唯一 workflow，没有 pytest/lint 作业；`src tests` lint 有 5 项现存错误 | 工程缺口 | 测试/CI：建立可复现 dev 环境，修复核心 lint 并接入测试；为注入 `robo` 的历史 RSI 脚本明确独立静态检查口径 |
| R4 / P2 | 任务卡有来源、执行器有实时能力字典，但尚无统一版本化能力/预算/会话契约；其他后端覆盖不同 | 部分实现 | 任务会话 + 能力：区分启动快照与实时能力，明确未知预算/停止确认和适配器覆盖矩阵；不将 `sources` 当作 schema 验收完成 |
| R5 / P2 | 项目管理页尚停留在 score=50，缺少后续 75→50→75 及默认执行变化；原始证据被 Git 忽略 | 文档状态已修正；证据归档待规范 | 记录评估：本页、路线和模块页同步；后续每个增量保存可访问的精简指标、版本指纹和原始产物索引 |
| R6 / P1 | 审查期间 prompt 切到 compact 任务卡后，`tests/test_robodojo_card.py::test_hosts_preserve_native_instruction_and_full_startup_capabilities` 的3个 host 用例仍要求包含完整 `card.render()`，定向检查实际失败；随后执行计划仍在接入 | 新增渲染契约与旧断言不一致，后续增量未冻结 | 任务卡/计划/测试：核对精简视图保留原始指令、预算回退、能力限制与观测来源；完整来源留在 JSON；同步测试并验证 plan 持久化、失败接续和权限边界，不能沿用先前通过结论 |

真实 unknown 对账、实机停止确认、独立采集/核验预算、跨 runtime 上下文与多任务质量回归仍属于
原路线的待补内容。本次没有把它们改写为新缺陷，也没有用本地测试替代物理验收。

## 本次验证

| 检查 | 结果 | 解释 |
| --- | --- | --- |
| 首轮全量 `pytest -q` | 773 passed，4 failed；4项本机端口测试在允许 socket 的环境复跑全部通过 | 合计777项通过，适用于首轮读取的版本；失败初因是沙箱 socket 权限 |
| 后续任务卡增量定向测试 | 84 passed，3 failed | 五个测试文件共87项；3个失败均为 compact 视图与旧全文包含断言冲突，见R6；不是 socket 失败 |
| 修改的 Python 文件 lint | 初次15文件通过；后续16文件检查有2项问题 | 并行增量中 `play.py` 的I001和 `prompt.py` 的E501；其后新加入的计划模块不在该次范围内 |
| 首轮 `ruff check src tests` | 5项现存问题 | `interaction.py` 3项E501，`runner.py` 与 `test_interaction_contracts.py` 各1项I001；三文件内容与HEAD一致，独立于后续新增lint问题 |
| 首轮全仓 Python AST 静态解析 | 172个文件通过 | 含当时 tracked Python 与新增 Python 文件，不含后来出现的执行计划模块；不执行固定任务程序 |
| 更新的设计 Markdown 相对文件链接 | 8个文件，无缺失目标 | 检查本地文件存在性；不等于网站已构建或远端链接可用 |
| `git diff --check` | 通过 | 无新增空白错误 |
| MkDocs 网站构建 | 未执行 | 当前验证环境未安装 mkdocs、material、pymdownx；没有声明文档站构建通过 |

测试解释器为本地 `runs/organize-table-astra-20261030/venv/bin/python`（Python 3.11），
lint 使用已有 `/tmp/pyrualean-supervision-lint/bin/ruff`（0.17.3）。复核命令：

```bash
python -m pytest -q
python -m pytest -q \
  tests/test_claude_runtime.py::test_play_episode_wires_the_claude_runtime \
  tests/test_robodojo_colour.py::test_an_episode_under_the_adapter_runs_the_real_loop \
  tests/test_robodojo_colour.py::test_a_trial_the_client_ends_is_wrapped_up_without_another_request \
  tests/test_supervision.py::test_http_protocol_against_local_valen_compatible_server
python -m pytest -q tests/test_robodojo_card.py tests/test_prompt.py \
  tests/test_play.py tests/test_robodojo.py tests/test_robodojo_trajectory.py --tb=short
ruff check src tests --output-format concise
git diff --check
```

扩大到 `src tests examples integrations docs/hooks` 的 lint 检查另报10717项，其中10704项位于历史
RSI `programs/`，包括运行时注入的 `robo` 所产生的未定义名称与风格问题；不能将它们等同于10704个运行故障。
应为固定程序单独约定检查口径；核心范围的5项结果不受此噪声影响。

日志和源码摘要保存在本地 `runs/repository-review-20261102/`。该目录同样被 Git 忽略，
本页保留结果；首轮测试与后续并行增量分别归档。执行计划接入发生于定向检查之后，
因此本文不宣称收尾时的持续变化工作区已经全量通过。本次审查未修改产品代码或测试。
