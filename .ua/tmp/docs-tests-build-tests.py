import importlib.util,ast,json,pathlib,re
spec=importlib.util.spec_from_file_location('base','.ua/tmp/docs-tests-build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('texts','.ua/tmp/docs-tests-test-summaries.py');texts=importlib.util.module_from_spec(spec);spec.loader.exec_module(texts)
TOPICS={'execution':'代码执行边界、输出限额与证据版本','libero':'LIBERO 类型化接口、抓放流程和调用账本','openai_runtime':'OpenAI 兼容规划循环、消息顺序、重试与预算','play':'交互运行器环境、模型目录回退及使用量审计','primitives':'完整与禁用 VLA 原语集合的接口、知识和运行契约','prompt':'API 参考、规划提示和任务卡序列化','recovery':'代码预检、部分执行、失败定位和恢复反馈','robocasa':'RoboCasa 导航、移动底座、夹爪及 VLA 接口','robodojo':'RoboDojo 双臂接口、隐藏成功判定与宿主环境桥接','robodojo_colour':'纯彩色几何测量、深度分支、异步客户端桥接与图像时序','robodojo_motion':'双臂运动学、工具执行、深度几何与自身遮挡过滤','robodojo_sight':'彩色相机几何、多视图目标测量和掩码筛选','robots':'机器人基类、注册表和通用任务契约','robotwin':'RoboTwin 双臂动作、VLA、世界坐标与宿主适配','runner':'独立策略加载、运行结果、超时和终止传播','sandbox':'受限导入、文件访问边界与内省逃逸阻断','strategies':'有界抓持验证、重定位和纠偏策略','stuck_recovery':'明确卡住后的恢复派发、轮次限制和证据保留','supervised_batch':'批量任务结果分类、续跑与进程所有权','supervision':'视觉监督契约、证据新鲜度、依赖门槛与暂停','valen_eval':'Valen 缓存评估、独立标注、阈值与回放变体'}
HELP={
'_reply':'构造带工具调用或文本内容的模型回复数据',
'card_images':'创建测试任务卡所需的临时图像文件',
'_toolkit':'创建使用临时代码工作区的 ArmToolkit',
'_run':'以脚本化传输执行 OpenAI 运行器并返回审计结果',
'_events':'构造包含会话与完成项目的运行事件记录',
'_card':'创建用于提示和原语配置测试的任务卡',
'_public_methods':'枚举机器人公开方法用于 API 参考覆盖检查',
'_dump':'序列化任务卡并返回用于往返校验的数据',
'rc_backend':'创建带抽屉与罐子的 RoboCasa 模拟后端',
'rc':'把模拟后端包装为 RoboCasa 机器人接口',
'_axis_quat':'由指定轴和角度生成测试四元数',
'_mul':'计算两个四元数的乘积',
'_quat':'按库约定组合俯仰、偏航和滚转为四元数',
'_pointing':'由四元数计算末端指向向量',
'rd_backend':'创建带双臂、瓶子和垃圾桶的 RoboDojo 模拟后端',
'rd':'把模拟后端包装为 RoboDojo 机器人接口',
'_fake_bridge':'构造与真实运行桥接模块形状一致的测试替身',
'_sibling':'独立加载同目录的其他测试模块以复用图形辅助工具',
'_ee_pose':'用正运动学构造末端位置与四元数观测',
'world':'创建测试场景、分割器与原生工具包',
'_off':'计算测量中心与已知几何中心的平面误差',
'_deploy':'从实际集成目录加载部署适配模块供协议测试',
'_episode':'构造桥接服务、策略客户端和测试回合环境',
'_adapter_model':'基于 XPolicyLab 模板加载实际策略模型适配器',
'_swing':'生成使右臂及腕相机转动的原生动作',
'_depth':'以射线几何渲染桌面物体的相机轴向深度',
'_checkout':'在临时目录构造评估宿主需要的 RoboDojo 项目布局',
'_pose':'由位置与姿态角构造目标位姿矩阵',
'sim':'创建六关节双臂模拟器和原语测试上下文',
'_now':'计算模拟器当前指定机械臂的末端位姿',
'_overhead':'构造俯视相机下盒体的标定、掩码和深度',
'_render':'用射线投射生成盒体、圆柱和圆锥的头部相机视图',
'_head_view':'生成头部视图及被物体占据的像素掩码',
'_located':'把渲染图像和分割掩码送入实际定位工具并返回结果',
'_arcs':'生成不同高度圆形物体近侧半圆表面点',
'render':'渲染指定相机每个像素所见的形状索引',
'views':'构造头部和双腕部相机视图集合',
'measured':'对指定形状执行多相机测量并返回几何与诊断结果',
'off':'计算测量结果相对形状真值的平面距离误差',
'rt_backend':'创建带锤子和积木的 RoboTwin 模拟后端',
'rt':'把模拟后端包装为 RoboTwin 机器人接口',
'_write':'写入临时策略文件用于模块加载与执行测试',
'cells':'创建受沙箱约束的代码单元工具包',
'grasp_obs':'构造带证据版本和新鲜度的抓持观测',
'toolkit':'创建启用卡住恢复的测试工具包',
'stuck':'提交明确卡住的结束报告以触发恢复判断',
'answer':'构造带概率与谓词结果的 Valen 响应',
'setup':'组装脚本化 Valen 客户端、视觉监督器与工具包',
'audit':'写入监督审计和图像记录供离线评估测试',
}
CLASSES={
'ScriptedTransport':'按预设序列返回 HTTP 状态或回复，并保留每次请求快照供消息协议断言。',
'_ToolResult':'模拟 RPent 工具返回对象，仅保存工具名与结果负载。',
'FakeRobocasaBackend':'模拟厨房抽屉、罐子、底座和夹爪；按 RoboCasa 响应封装返回运动、导航及 VLA 结果。',
'_FakeState':'提供模拟状态的图像或数组产物路径，支持宿主任务卡导出测试。',
'_FakeToolkit':'把测试后端和状态产物封装成宿主需要的工具包接口。',
'FakeRobodojoBackend':'模拟双臂、瓶子和垃圾桶，复现状态改变与只读工具响应，成功判定不进入机器人公开状态。',
'_FakeEnv':'向宿主返回带原生步数和回合标志的模拟环境状态。',
'_JointEnv':'模拟每步按固定比例逼近目标关节的双臂环境，用于归位与预算测试。',
'_JointPrimitives':'包装关节环境并统计取消检查次数。',
'Scene':'模拟桌面几何、双臂状态和三路彩色相机，供无需真实模拟器的视觉与桥接测试。',
'Segmenter':'以射线投射场景模拟 SAM3，支持文字提示和像素提示的目标掩码查询。',
'TaskEnv':'模拟部署适配器使用的评估环境、动作计数和客户端终态判定。',
'Client':'在同一进程模拟 XPolicyLab 模型客户端协议并记录调用。',
'LateScene':'模拟图像落后状态一帧的观测，并可显式提供图像对应状态。',
'Told':'按预设掩码和置信度覆盖特定分割响应，用于多视图质量验证。',
'HalfSeen':'模拟腕部相机只看到物体一侧的遮挡分割结果。',
'DepthScene':'在彩色测试场景上增加深度图与标定以验证深度分支。',
'_SimArms':'模拟六关节双臂的逐步运动和夹爪行程，可设置阻挡与夹持开度。',
'_Primitives':'组合模拟环境、分割客户端和取消检查，提供原生工具运行上下文。',
'_Sam3':'保存固定掩码并记录分割请求，替代真实分割服务。',
'ToyState':'保存玩具机器人的步数、任务、末端位置与终止标志。',
'ToyRobot':'以单个状态改变工具验证机器人基类的解析、摘要与运动分发。',
'_NpyArtifacts':'包装模拟后端，提供裸 NumPy 世界坐标图产物。',
'NpyRobot':'声明使用裸 NumPy 数组世界坐标图的测试机器人。',
'BudgetState':'在通用状态记录上增加回合截断字段。',
'BudgetRobot':'验证仅由解析后的状态报告步数截断的机器人适配路径。',
'FakeRobotwinBackend':'模拟双臂、锤子和积木，复现 RoboTwin 工具响应、VLA 分段成功和动作预算。',
'ScriptedValen':'按预设序列返回视觉谓词结果或异常，同时记录监督输入包。',
}
METHOD={
'__init__':'初始化该测试替身的输入、状态与调用记录', '__call__':'保存请求快照并返回下一条脚本化传输结果','execute_tool':'记录请求并将工具名分发到模拟处理器','solved':'返回测试后端保存的最终任务成功判定','artifact':'按名称和步数提供模拟图像、数组或文本产物','env_steps':'返回已消耗的原生环境动作数','cancel':'设置取消标志供后续执行检查','_map':'生成并缓存模拟世界坐标图','_yaw':'由底座四元数求偏航角','_set_yaw':'把底座偏航角写回四元数状态','_eef_quat':'根据测试俯仰状态生成末端四元数','_success':'按抽屉关节开度判断模拟任务成功','_step':'推进模拟原生控制步数','_apply_gripper':'应用夹爪命令并维护抓持状态','_do_move_to':'模拟绝对位姿运动并返回到达、步数与受阻信息','_do_move_delta':'由当前位姿计算相对目标并复用绝对运动','_do_rotate_pitch':'模拟相对俯仰调整并同步姿态状态','_do_set_gripper':'模拟夹爪开合和对应动作消耗','_do_release':'复用张开夹爪动作模拟释放物体','_do_scripted_grasp':'模拟接近、闭合和抬升组成的脚本抓取','_do_navigate_to':'模拟移动底座导航、校准和目标到达','_do_move_base':'模拟底座平移并返回实测位移','_do_rldx_skill':'模拟 RoboCasa VLA 分块执行与任务进度','_do_rldx_arm':'以受限底座参数执行机械臂 VLA','_readonly':'返回状态、反投影或世界图查询的只读结果','_record':'保存当前模拟状态及最近工具执行结果','_view':'把指定历史记录转换为宿主状态响应','artifact_path':'将后端产物写入临时路径供宿主读取','_spent':'检查模拟回合是否耗尽动作上限','_tick':'受剩余预算约束推进原生动作计数','_summary':'整理双臂位姿和夹爪状态为响应数据','_angles':'返回指定机械臂的俯仰、偏航和滚转','_gripper':'在运动时应用夹爪保持或释放规则','_release':'更新物体释放位置并判定是否进入垃圾桶','_do_view_env_state':'读取保存的模拟环境状态','_do_segment':'返回指定提示词的模拟分割结果','_do_back_project':'校验像素范围并返回模拟世界坐标','_do_pi0_pick':'模拟冻结抓取策略的分块动作和双臂反馈','_do_locate':'返回目标瓶子的模拟尺寸与抓取几何','_do_move_pose':'模拟抓取点位姿运动及角度保持','_do_grip':'模拟完整夹爪行程并报告实际抓持','_do_home':'逐步逼近起始关节并遵守动作预算','get_status':'返回原生步数、任务终态和测试环境状态','get_obs':'组合当前机器人状态与相机观测','step':'应用一次模拟关节或夹爪动作并返回观测','_check_cancelled':'记录工具执行时的取消检查','state':'生成当前双臂位姿和夹爪状态字典','view':'根据机器人状态构造头部或腕部相机位姿','segment':'按测试场景或预置掩码返回模拟 SAM3 分割结果','take_action':'推进评估环境动作数并更新回合终态','get_running_env_idx_list':'返回仍在运行的环境索引','is_episode_end':'结合成功条件和步数上限锁定回合结束','call':'分发 XPolicyLab 客户端协议到策略模型方法','_state_from_envelope':'从后端响应解析测试机器人状态','_summary_dict':'生成可序列化的简明状态摘要','go':'调用基类状态改变接口执行运动','__getattr__':'把未覆盖属性转发给内部模拟后端','_status':'组织 RoboTwin 原生成功与预算状态','_budget_exhausted':'判断动作计数是否达到上限','_run':'按成功与预算限制执行指定数量模拟动作','_completion':'将模拟终态转换成完成原因和截断标志','_tcp':'按固定工具偏移计算末端抓取点','_require_active':'在成功或预算耗尽后拒绝继续执行','_do_render':'返回只刷新视图的成功结果','_do_rotate_wrist':'保持位置并更新指定机械臂腕部偏航','_do_lingbot_act':'模拟 RoboTwin VLA 抓锤及敲击阶段','_world_at':'由像素和已知物体区域生成世界坐标','_world_error':'构造带错误代码与细节的世界查询失败负载','_robot_state':'汇总双臂关节、末端和夹爪状态','load':'从模拟后端产物字节加载 NumPy 数组','predict':'保存监督请求并返回下一条预设响应或异常'}

def modpath(mod):
 p='src/'+mod.replace('.','/')
 for cand in (p+'.py',p+'/__init__.py'):
  if cand in base.ALLFILES:return cand
 return None
rootexports={}
for n in ast.parse(pathlib.Path('src/pyrualean/__init__.py').read_text()).body:
 if isinstance(n,ast.ImportFrom) and n.level==1 and n.module:
  for a in n.names:rootexports[a.asname or a.name]=modpath('pyrualean.'+n.module)

ns=[];es=[];missing=[]
for r in json.load(open('.ua/tmp/docs-tests-extract-20.json'))['results']:
 p=r['path'];key=pathlib.Path(p).stem.removeprefix('test_');src=pathlib.Path(p).read_text();lines=src.splitlines();tree=ast.parse(src)
 fs=r.get('functions',[]);testfs=[f for f in fs if f['name'].startswith('test_')];sums=texts.TEST_TEXT[key].splitlines()
 assert len(testfs)==len(sums),(key,len(testfs),len(sums))
 translated={f['name']:'验证'+s+'。' for f,s in zip(testfs,sums)}
 ns.append(base.node(p,'file','',f'验证{TOPICS[key]}。通过固定输入、模拟后端或临时资源检查正常行为及失败边界。',complexity='complex' if r.get('nonEmptyLines',0)>200 else 'moderate',tags=['测试','回归验证','行为契约']))
 local={};classids={}
 for f in fs:
  name=f['name'];summary=translated.get(name) or (HELP.get(name,'')+'。')
  if summary=='。':missing.append((p,name));summary='为本文件的回归验证构造测试输入与执行上下文。'
  n=base.node(p,'function',name,summary,f['startLine'],f['endLine'],tags=['测试','断言验证','边界检查'] if name.startswith('test_') else ['测试','测试夹具','模拟数据'])
  ns.append(n);local[name]=n['id'];es.extend([base.edge(base.fid(p),n['id'],'contains'),base.edge(base.fid(p),n['id'],'exports')])
 for c in r.get('classes',[]):
  cn=base.node(p,'class',c['name'],CLASSES[c['name']],c['startLine'],c['endLine'],tags=['测试','测试替身','状态模拟']);ns.append(cn);classids[c['name']]=cn['id'];es.extend([base.edge(base.fid(p),cn['id'],'contains'),base.edge(base.fid(p),cn['id'],'exports')])
  ac=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==c['name'])
  for m in ac.body:
   if not isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
   sm=METHOD.get(m.name)
   if not sm:missing.append((p,c['name']+'.'+m.name));sm='执行测试替身的局部状态处理'
   mn=base.node(p,'function',c['name']+'.'+m.name,sm+'，服务于'+TOPICS[key]+'的回归验证。',m.lineno,m.end_lineno,tags=['测试','模拟接口','状态处理']);ns.append(mn);es.extend([base.edge(base.fid(p),mn['id'],'contains'),base.edge(cn['id'],mn['id'],'contains')])
  for inherit in ac.bases:
   if isinstance(inherit,ast.Name) and inherit.id in classids:es.append(base.edge(cn['id'],classids[inherit.id],'inherits'))
 for cg in r.get('callGraph',[]):
  if cg.get('caller') in local and cg.get('callee') in local:es.append(base.edge(local[cg['caller']],local[cg['callee']],'calls'))
 for dest in base.BATCHES[20]['batchImportData'][p]:es.append(base.edge(base.fid(p),'file:'+dest,'imports'))
 # 导入 map 原样保留为空；下面只记录源码确认的测试覆盖关系。
 covered=set()
 for im in ast.walk(tree):
  if isinstance(im,ast.ImportFrom) and im.module and im.module.startswith('pyrualean'):
   if im.module=='pyrualean':
    for a in im.names:
     q=rootexports.get(a.name) or modpath(im.module+'.'+a.name)
     if q:covered.add(q)
   else:
    q=modpath(im.module)
    if q:covered.add(q)
    for a in im.names:
     q=modpath(im.module+'.'+a.name)
     if q:covered.add(q)
  elif isinstance(im,ast.Import):
   for a in im.names:
    if a.name.startswith('pyrualean'):
     q=modpath(a.name)
     if q:covered.add(q)
 for q in base.ALLFILES:
  if not q.startswith(('src/','examples/','integrations/')) or not q.endswith('.py'):continue
  if q in src:covered.add(q)
 for q in covered:es.append(base.edge(base.fid(q),base.fid(p),'tested_by'))
print('MISSING',missing)
base.output(20,ns,es)
