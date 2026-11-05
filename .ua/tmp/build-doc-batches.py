import json, math, posixpath, re
from pathlib import Path

root = Path('/mnt/data_hdd/mhzhu/GPTCode/PyRUA-Lean')
ua = root / '.ua'
scan = json.loads((ua/'intermediate/scan-result.json').read_text())
batches = json.loads((ua/'intermediate/batches.json').read_text())['batches']
inventory = {f['path']: f for f in scan['files']}
def kind(p):
    return {'docs':'document','config':'config','infra':'pipeline'}.get(inventory[p]['fileCategory'],'file')
def ident(p): return kind(p)+':'+p

summaries = {
'.github/workflows/docs.yml':'构建 MkDocs 文档站，使用 Python 3.11 安装文档依赖并执行严格构建。主分支推送且仓库身份匹配时，将站点发布到指定页面分支。',
'README.md':'介绍通过 Python 对象控制机器人的代码生成框架，区分在线模型执行与离线 RSI 固定程序。提供安装、机器人环境、运行入口和基准报告链接。',
'mkdocs.yml':'配置 Material 文档站的导航、主题、Markdown 扩展和 API 生成钩子，将入门、机器人、运行指南及架构设计组织为可浏览站点。',
'pyproject.toml':'定义 pyrualean Python 包、setuptools 构建和命令行入口，以及开发与可选工具依赖、测试和代码检查配置；包本身不声明必需运行依赖。',
'docs/design/architecture.md':'梳理现有在线执行链、Python 单元与完整程序边界，以及控制、代码和模型三类反馈循环。提出统一上下文、动作凭据和证据契约；这些设计项不等同于已实现的公共类型。',
'docs/design/index.md':'作为架构设计文档入口，连接十个职责切面、总体架构和演进路线。明确文档依据当前源码静态梳理，并区分现有能力与建议中的公共契约。',
'docs/design/roadmap.md':'提出分阶段演进顺序：先统一契约，再完善观测与执行，随后补齐证据、恢复、上下文和评估。记录验收目标与迁移原则，属于规划而非完成状态。',
'docs/design/parts/capabilities-manual.md':'梳理机器人手册、有效动作原语和协调能力的现有入口，提出统一能力清单与动作契约。强调模型只能使用环境实际暴露的能力。',
'docs/design/parts/code-runtime.md':'分析持久 Python 单元与完整程序的执行、反馈和预算边界，以及沙箱状态在故障后的保留方式。提出运行时契约整理方向，说明动作效果不会自动回滚。',
'docs/design/parts/execution-control.md':'分析动作从机器人接口经后端到原生环境的执行与取消路径，以及协调控制检查点。提出动作凭据等契约，并指出取消请求与完成等待之间的语义差异。',
'docs/design/parts/knowledge-skills.md':'区分随运行注入的机器人知识、监督与恢复核心，以及仓库中的开发技能文档。提出知识清单和版本追踪方案，说明卡住恢复知识只在确认交接后注入。',
'docs/design/parts/observation-perception.md':'梳理状态、图像、相机信息和感知工具的来源，区分历史观测与重新获取的证据。提出携带坐标系、标定和新鲜度信息的统一观测表示。',
'docs/design/parts/planner-context.md':'分析提示构建、工具反馈及模型决策上下文，提出整理决策输入和恢复交接报告的方向。将模型规划与执行约束分离，避免假设已有固定任务状态机。',
'docs/design/parts/records-evaluation.md':'分析运行记录、监督审计和评估输出的职责，强调环境真值与启发式判断分开存储。提出统一事件协议与可复查的评估链。',
'docs/design/parts/recovery-handoff.md':'区分有界局部修正与上层模型重新规划，梳理故障观测、持久命名空间和剩余预算的交接。提出统一交接契约，强调保留真实场景而不回滚或恢复已中断调用栈。',
'docs/design/parts/task-session.md':'梳理任务配置、环境会话、执行预算、模型结束判断和环境成功真值。提出统一任务与预算快照的设计，避免将建议中的数据结构当作现有实现。',
'docs/design/parts/verification-evidence.md':'分析 Valen 原子视觉事实、前置条件和宿主执行门控的协作方式，强调证据新鲜度与独立验证。提出证据协议统一方案，区分已实现逻辑与未来契约。',
'docs/guide/agents.md':'比较 Codex、Claude 和 OpenAI 运行时的配置与工具交互，解释各自如何共享机器人提示、执行预算和运行产物。',
'docs/guide/cli.md':'说明 api、prompt、play、card、generate、run、summarize、compare 和 valen-eval 命令，以及运行参数、默认值与离线评估入口。',
'docs/guide/coordination.md':'介绍双臂协调控制、Pose 目标、夹爪动作和检查点，并说明原生 move_both 的支持范围。强调能力检测和实际并行控制的环境约束。',
'docs/guide/outputs.md':'列出每个运行目录中的结果、提示、工具回合、调用记录、执行单元、视频和监督审计等产物，帮助定位与复查一次机器人执行。',
'docs/guide/recovery.md':'说明 Python 执行失败后的结构化反馈、恢复观测与修复单元，以及已发生动作和持久命名空间的保留。介绍局部纠错和卡住交接的边界，避免盲目重放。',
'docs/guide/running.md':'说明如何运行机器人任务及选择模型运行时、单元或完整程序模式、反馈方式和预算。介绍默认监督暂停与卡住恢复行为。',
'docs/guide/strategies.md':'介绍 relocalize、correct 和 grasp_verified 等可组合策略，通过用户回调与有界重试处理局部不确定性；超出局部能力时将判断交回模型。',
'docs/guide/valen-evaluation.md':'说明如何从保存的视觉监督审计创建标注模板、复查裁剪和问题并比较阈值。评估基于历史产物，不启动机器人任务。',
'docs/robots/adding-a-robot.md':'说明新增 XPolicyLab 机器人适配的责任划分，包括模拟器真值、观测更新、动作接口、原生原语和机器人知识手册。',
'docs/robots/index.md':'概览支持的机器人与仿真环境，对比运行栈、VLA、分割感知和相机配置，并链接各环境指南。',
'docs/robots/libero.md':'介绍基于 RPent 和 MuJoCo 的 LIBERO Franka 环境、任务变体与启动配置，以及 Pi0.5 和 SAM3 等可选能力。',
'docs/robots/robocasa.md':'介绍 RoboCasa PandaOmron 移动操作环境、厨房任务和导航使用方式，说明 RPent 栈及相关感知与策略配置。',
'docs/robots/robodojo.md':'介绍 RoboDojo ARX-X5 双臂环境、XPolicyLab 和 IsaacSim 接入、原生动作与 SAM3 感知，说明默认 RGB 观测和评估协议边界。',
'docs/robots/robotwin.md':'介绍基于 RPent 和 SAPIEN 的 RoboTwin 双臂环境、任务配置、相机与 LingBot VLA 等能力。',
'integrations/xpolicylab/evaluation/robodojo_summary_official.md':'记录 RoboDojo 官方团队任务集的历史评估协议、分维度表现和任务级结果。该报告中的结果属于所列配置和种子，不代表此次静态分析重新运行了实验。',
'integrations/xpolicylab/evaluation/robodojo_summary_rgb.md':'记录默认 RGB 观测协议下的任务结果和当时已完成的种子覆盖情况，提供整体与任务级指标。与 RGB-D 和其他任务数量的报告分别解释。',
'integrations/xpolicylab/evaluation/robodojo_summary_rgbd.md':'记录使用深度与相机矩阵的 RGB-D 历史实验，覆盖 54 个配置和多个种子。明确这与发布策略默认的 RGB 输入协议不同。',
'integrations/xpolicylab/evaluation/robodojo_summary_rsi.md':'记录离线固定程序 RSI 的任务级和总体评估，区分开发种子与留出种子结果。描述固定程序协议，避免与在线模型策略直接混同。',
'src/pyrualean/knowledge/astra_stuck_recovery.md':'作为确认卡住后注入的恢复知识，要求沿用当前场景、Python 命名空间与剩余预算重新规划。强调先检查新证据，再开展依赖这些事实的动作。',
'src/pyrualean/knowledge/astra_valen_supervision.md':'作为运行时监督规划知识，说明模型如何编写 Python 回调、核验原子视觉事实并声明执行前置条件。把观察判断、宿主门控和局部修正结合起来，不提供固定任务答案。',
}

summaries.update({'mkdocs.yml': '配置 Material 文档站的导航、主题、Markdown 扩展和 API 生成钩子，新增交互延迟专题并连接十个架构职责切面及演进路线。', 'docs/design/architecture.md': '梳理现有在线执行链、cells/program/RSI 模式边界和三类闭环，并提出任务、观测、动作、证据及恢复的公共契约。新增完整代码就绪计时、决策期限与流式完整工具项的跨模块要求，均明确为待补设计。', 'docs/design/index.md': '作为框架设计入口，连接总体架构、交互延迟专题、十个职责切面与推进路线。明确现有实现、部分实现、待补契约和未来验收的区别，并注明当前内容包含未提交工作区实现。', 'docs/design/interaction-latency.md': '分析已有 Astra/Codex 日志中反馈到下一段代码的等待，区分历史代理指标、总策略时间与尚未实现的统一分项计时。提出决策与请求身份、派发前期限及证据复查、流式完整工具项边界，以及同时衡量响应时间和任务质量的对照实验。', 'docs/design/roadmap.md': '提出从公共契约到观测、动作回执、独立验证、恢复和上下文的分阶段推进路线。将交互延迟基线与接口兼容性核查前移到 P0，并明确流式派发和候选预计算所依赖的基础契约；阶段与验收均为规划。', 'docs/design/parts/code-runtime.md': '分析持久 Python 单元与完整程序的执行、反馈、沙箱和预算边界，以及故障后真实副作用与命名空间的保留。提出完整工具项的计时、编译、去重和派发准入契约，要求排队后重新核查期限、会话与证据依赖，不执行流式代码前缀。', 'docs/design/parts/planner-context.md': '分析提示构建、在线模型循环与持久辅助函数，提出版本化决策上下文和失败交接内容。新增出码延迟与判断粒度设计：按实际运行路径记录推理配置和缓存，比较本地规则与模型往返，拒绝直接派发依赖过期证据的候选。', 'docs/design/parts/task-session.md': '梳理权威任务、环境生命周期、分散预算及模型与原生判定，提出统一任务和预算快照。新增每次决策的身份、交互目标与期限，区分取消生成、动作在途与设备停止确认，并明确迟到代码不得恢复原派发许可。', 'docs/design/parts/records-evaluation.md': '分析调用、观测、监督、恢复和评估记录的现有基础，提出跨层因果事件与分层判定协议。新增反馈就绪到完整代码就绪的计时和分组统计设计，要求保留失败、重试、断流、压缩与缺失时点，避免把总策略时间解释为模型推理时间。'})

robot_notes={
'libero':'说明 Franka 动作的米与弧度单位、夹爪正负约定和 SAM3 感知及几何定位',
'robocasa':'说明移动底盘导航、移动后的重新观测、操作坐标系和 xyzw 四元数约定',
'robodojo':'说明 ARX 双臂的 TCP 与 EEF 区别、角度单位、世界坐标、协调动作、夹爪和恢复边界',
'robodojo_xpl_rgb':'说明 RGB 头部与腕部相机、图像掩码、双臂几何控制与恢复；默认观测不提供深度',
'robotwin':'说明世界坐标、wxyz 四元数、EEF 与 TCP 偏移以及双臂夹爪和感知约定',
}
for robot,note in robot_notes.items():
    for suffix in ['', '-novla']:
        p=f'src/pyrualean/knowledge/{robot}{suffix}.md'
        summaries[p]=note+'。'+('该版本仅使用原生控制与感知能力，不包含 VLA 操作指导。' if suffix else '手册还包含 VLA 使用指导，实际可用原语取决于运行配置。')

for policy in ['PyRUA_Lean','PyRUA_RSI']:
    base=f'integrations/xpolicylab/{policy}/'
    rsi=policy.endswith('RSI')
    texts={
      'README.md': ('说明 XPolicyLab 离线固定程序策略的安装、启动与评估：按任务加载程序，首次返回完整 Python，随后结束，不在线请求语言模型。' if rsi else '说明 XPolicyLab 在线机器人代码生成策略的安装、模型配置和评估入口，连接策略服务、环境客户端与本地 PyRUA 执行栈。'),
      'deploy.yml': ('配置 RoboDojo RGB 离线程序策略、机器人接口、执行预算和输出目录，部分模型配置仅为接口兼容而保留。' if rsi else '配置 RoboDojo RGB 在线 OpenAI 策略、无 VLA 原语、按需图像、执行预算和结果目录。'),
      'download_assets.sh':'从 Hugging Face 或 ModelScope 下载 SAM3 检查点，并验证目标文件非空，供策略服务加载感知模型。',
      'eval.sh':'选择空闲端口启动策略服务，等待服务就绪后运行环境客户端，并通过退出清理停止后台服务。',
      'install.sh':'在已有 CUDA PyTorch 环境中安装 PyRUA、XPolicyLab 和感知依赖，用约束保留现有 torch 与 torchvision 版本，并执行导入检查。'+('额外安装离线程序所需的 SciPy。' if rsi else ''),
      'setup_eval_env_client.sh':'整理任务、种子、GPU、Conda 环境和策略配置参数，调用 XPolicyLab 公共环境客户端启动流程。',
      'setup_eval_policy_server.sh':'激活策略环境，配置 GPU、Python 路径和 SAM3 检查点，将部署配置及覆盖参数传给 XPolicyLab 策略服务器。',
    }
    summaries.update({base+k:v for k,v in texts.items()})

explicit={
'README.md':['src/pyrualean/cli.py','src/pyrualean/play.py','integrations/xpolicylab/PyRUA_Lean/model.py','integrations/xpolicylab/PyRUA_RSI/model.py'],
'docs/guide/agents.md':['src/pyrualean/openai_runtime.py','src/pyrualean/claude_runtime.py','src/pyrualean/generate.py','src/pyrualean/prompt.py'],
'docs/guide/cli.md':['src/pyrualean/cli.py'],
'docs/guide/coordination.md':['src/pyrualean/coordination.py','src/pyrualean/hosts/robodojo_coordination.py','src/pyrualean/robodojo.py'],
'docs/guide/outputs.md':['src/pyrualean/play.py','src/pyrualean/_backend.py','src/pyrualean/supervision.py'],
'docs/guide/recovery.md':['src/pyrualean/recovery.py','src/pyrualean/stuck_recovery.py','src/pyrualean/arms.py'],
'docs/guide/running.md':['src/pyrualean/play.py','src/pyrualean/cli.py','src/pyrualean/arms.py'],
'docs/guide/strategies.md':['src/pyrualean/strategies.py'],
'docs/guide/valen-evaluation.md':['src/pyrualean/evaluation/valen.py','src/pyrualean/cli.py'],
'docs/robots/adding-a-robot.md':['src/pyrualean/robots.py','src/pyrualean/_robot.py','src/pyrualean/hosts/robodojo_xpolicylab.py'],
'docs/robots/index.md':['src/pyrualean/robots.py'],
'src/pyrualean/knowledge/astra_stuck_recovery.md':['src/pyrualean/stuck_recovery.py'],
'src/pyrualean/knowledge/astra_valen_supervision.md':['src/pyrualean/supervision.py','src/pyrualean/prompt.py'],
}
explicit['docs/design/interaction-latency.md'] = ['src/pyrualean/play.py', 'src/pyrualean/cli.py', 'src/pyrualean/generate.py', 'src/pyrualean/openai_runtime.py', 'src/pyrualean/claude_runtime.py', 'src/pyrualean/arms.py', 'integrations/xpolicylab/PyRUA_Lean/deploy.yml']
for robot in ['libero','robocasa','robodojo','robotwin']:
    explicit[f'docs/robots/{robot}.md']=[f'src/pyrualean/{robot}.py']
for robot in robot_notes:
    for suffix in ['','-novla']:
        explicit[f'src/pyrualean/knowledge/{robot}{suffix}.md']=[f'src/pyrualean/{robot}.py','src/pyrualean/prompt.py']

for batch in batches:
    idx=batch['batchIndex']
    if idx not in range(6,16): continue
    data=json.loads((ua/f'tmp/docs-tests-extract-{idx}.json').read_text())
    assert data['scriptCompleted'] and not data.get('filesUnreadable')
    extracted={r['path']:r for r in data['results']}
    nodes=[]; edges={}
    def edge(p,q,t='documents'):
        if q not in inventory or p==q: return
        e={'source':ident(p),'target':ident(q),'type':t,'direction':'forward','weight':{'documents':.5,'related':.5,'configures':.6,'depends_on':.6,'triggers':.6,'deploys':.7,'imports':.7}[t]}
        edges[(e['source'],e['target'],t)]=e
    for f in batch['files']:
        p=f['path']; t=kind(p); r=extracted[p]
        assert not any(r.get(k) for k in ['definitions','services','endpoints','steps','resources','classes']),p
        exported={x['name'] for x in r.get('exports',[])}
        assert not any(x['endLine']-x['startLine']+1>=10 or x['name'] in exported for x in r.get('functions',[])),p
        n=r.get('nonEmptyLines',f['sizeLines'])
        tags= ['文档','架构设计','演进规划'] if p.startswith('docs/design/') else ['机器人知识','动作约定','运行提示'] if '/knowledge/' in p else ['文档','评估结果','实验协议'] if '/evaluation/' in p else ['文档','使用指南','机器人'] if t=='document' else ['配置','部署','构建'] if t=='config' else ['持续集成','文档构建','部署'] if t=='pipeline' else ['部署脚本','环境配置','评估']
        nodes.append({'id':ident(p),'type':t,'name':Path(p).name,'filePath':p,'summary':summaries[p],'tags':tags,'complexity':'simple' if n<50 else 'moderate' if n<=200 else 'complex'})
        text=(root/p).read_text()
        # Only connect source references to paths present in the official inventory.
        for q in explicit.get(p,[]): edge(p,q)
        if t=='document':
            for q in inventory:
                if q in text: edge(p,q,'related' if kind(q)=='document' else 'documents')
            for href in re.findall(r'\]\(([^)\s]+)',text):
                href=href.split('#')[0]
                if not href or '://' in href: continue
                q=posixpath.normpath(posixpath.join(posixpath.dirname(p),href))
                edge(p,q,'related' if q in inventory and kind(q)=='document' else 'documents')
        for q in batch['batchImportData'].get(p,[]): edge(p,q,'imports')
        if p=='mkdocs.yml':
            for q in inventory:
                if q.startswith('docs/') and (q in text or q[5:] in text): edge(p,q,'configures')
        if p=='pyproject.toml':
            for q in ['src/pyrualean/cli.py','src/pyrualean/__init__.py']: edge(p,q,'configures')
        if p=='.github/workflows/docs.yml':
            edge(p,'mkdocs.yml','depends_on'); edge(p,'docs/hooks/api_reference.py','triggers')
        if '/PyRUA_' in p:
            base=posixpath.dirname(p)+'/'
            if p.endswith('README.md'):
                for q in inventory:
                    if q.startswith(base) and '/' not in q[len(base):]: edge(p,q,'related' if kind(q)=='document' else 'documents')
            if p.endswith('deploy.yml'):
                for q in ['model.py','deploy.py']: edge(p,base+q,'configures')
            if p.endswith('/eval.sh'):
                for q in ['setup_eval_policy_server.sh','setup_eval_env_client.sh']: edge(p,base+q,'triggers')
            if p.endswith('setup_eval_policy_server.sh') or p.endswith('setup_eval_env_client.sh'): edge(p,base+'deploy.yml','depends_on')
            if p.endswith('install.sh'): edge(p,'pyproject.toml','depends_on')
    edges=list(edges.values())
    parts=math.ceil(max(len(nodes)/60,len(edges)/120))
    paths=sorted(n['filePath'] for n in nodes); chunk=math.ceil(len(paths)/parts)
    outputs=0
    for offset in range(0,len(paths),chunk):
        selected=set(paths[offset:offset+chunk]); ns=[n for n in nodes if n['filePath'] in selected]; ids={n['id'] for n in ns}
        es=[e for e in edges if e['source'] in ids]
        outputs+=1
        name=f'batch-{idx}.json' if parts==1 else f'batch-{idx}-part-{outputs}.json'
        (ua/'intermediate'/name).write_text(json.dumps({'nodes':ns,'edges':es},ensure_ascii=False,indent=2)+'\n')
    print(f'batch {idx}: {len(nodes)} nodes, {len(edges)} edges, {outputs} parts')
