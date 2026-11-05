import ast,json,math,re,sys
from pathlib import Path
root=Path('/mnt/data_hdd/mhzhu/GPTCode/PyRUA-Lean'); ua=root/'.ua'
indices=[int(x) for x in sys.argv[1:]]
batches={b['batchIndex']:b for b in json.loads((ua/'intermediate/batches.json').read_text())['batches']}
scan=json.loads((ua/'intermediate/scan-result.json').read_text()); paths={f['path'] for f in scan['files']}
# Supplementation only: official file/function/class discovery remains authoritative.
# Class methods are those listed by official extraction, with AST source ranges
# completing the upstream omission of method ranges (approved by root agent).
all_ev={}
for ix in [1,2,3,4,5,19]:
 for f in json.loads((ua/f'tmp/evidence-{ix}.json').read_text()):all_ev[f['path']]=f
symbol_ids={}
for path,e in all_ev.items():
 for s in e['symbols']:symbol_ids[(path,s['name'])]=f"{s['kind']}:{path}:{s['name']}"
weights={'contains':1.0,'imports':.7,'calls':.8,'exports':.8,'inherits':.9,'tested_by':.5,'depends_on':.6,'documents':.5,'configures':.6,'related':.5}
def edge(a,b,t):return {'source':a,'target':b,'type':t,'direction':'forward','weight':weights[t]}
for i in indices:
 b=batches[i]; evidence=json.loads((ua/f'tmp/evidence-{i}.json').read_text())
 official={r['path']:r for r in json.loads((ua/f'tmp/ua-file-extract-results-{i}.json').read_text())['results']}
 semantics={}; current=None
 for line in (ua/f'tmp/semantics-{i}.txt').read_text().splitlines():
  if not line or line.startswith('#'):continue
  if line.startswith('@'):
   p,s,t=line[1:].split('|',2); current=p; semantics[p]={'summary':s,'tags':t.split(','),'symbols':{}}
  else:
   n,s=line.split('|',1); semantics[current]['symbols'][n]=s
 missing=[]; nodes=[]; edges=[]
 for f in evidence:
  path=f['path']; r=official[path]; sem=semantics.get(path)
  if not sem:missing.append(path);continue
  typ={'code':'file','script':'file','markup':'file','docs':'document','config':'config'}.get(f['category'],'file')
  fid=f'{typ}:{path}'
  complexity='simple' if r['nonEmptyLines']<50 else 'moderate' if r['nonEmptyLines']<=200 else 'complex'
  nodes.append({'id':fid,'type':typ,'name':Path(path).name,'filePath':path,'summary':sem['summary'],'tags':sem['tags'],'complexity':complexity})
  for s in f['symbols']:
   summary=sem['symbols'].get(s['name'])
   if not summary:missing.append(f"{path}:{s['name']}");continue
   nid=symbol_ids[(path,s['name'])]; length=s['range'][1]-s['range'][0]+1
   node={'id':nid,'type':s['kind'],'name':s['name'],'filePath':path,'lineRange':s['range'],'summary':summary,'tags':sem['tags'][:2]+(['类方法'] if s['owner'] else ['数据模型'] if s['kind']=='class' else ['函数']),'complexity':'complex' if length>100 else 'moderate' if length>35 else 'simple'}
   if s['owner']:node['languageNotes']='方法由官方结构列出；精确行号通过只读源码 AST 补足官方结果未提供的方法范围。'
   nodes.append(node);edges.append(edge(fid,nid,'contains'))
   if s['owner']:edges.append(edge(f"class:{path}:{s['owner']}",nid,'contains'))
   elif any(x['name']==s['name'] for x in r.get('exports',[])):edges.append(edge(fid,nid,'exports'))
  for target in b['batchImportData'].get(path,[]):edges.append(edge(fid,f'file:{target}','imports'))
  if r['language']!='python':continue
  tree=ast.parse((root/path).read_text()); alias={}
  for n in ast.walk(tree):
   if not isinstance(n,ast.ImportFrom) or not n.module:continue
   suffix=n.module.replace('.','/')
   candidates=[p for p in b['batchImportData'].get(path,[]) if p.endswith(suffix+'.py') or p.endswith(suffix+'/__init__.py')]
   if len(candidates)==1:
    for a in n.names:alias[a.asname or a.name]=(candidates[0],a.name)
   # Tests have src-layout imports missing in authoritative importMap.
   # Record only tested_by evidence, without changing imports.
   if path.startswith('tests/') and n.module.startswith('pyrualean'):
    target='src/'+suffix+'.py'
    if target in paths:edges.append(edge('file:'+target,fid,'tested_by'))
  for c in r.get('callGraph',[]):
   ln=c['lineNumber']; containers=[s for s in f['symbols'] if s['kind']=='function' and s['range'][0]<=ln<=s['range'][1]]
   if not containers:continue
   caller=min(containers,key=lambda s:s['range'][1]-s['range'][0]); source=symbol_ids[(path,caller['name'])]
   callee=c['callee']; target=None
   if '.' not in callee:
    target=symbol_ids.get((path,callee))
    if not target and callee in alias:target=symbol_ids.get(alias[callee])
   elif callee.startswith('self.') and caller['owner']:
    target=symbol_ids.get((path,caller['owner']+'.'+callee[5:]))
   if target and target!=source:edges.append(edge(source,target,'calls'))
  for n in tree.body:
   if isinstance(n,ast.ClassDef):
    source=symbol_ids.get((path,n.name))
    for base in n.bases:
     if not isinstance(base,ast.Name):continue
     target=symbol_ids.get((path,base.id)) or (symbol_ids.get(alias[base.id]) if base.id in alias else None)
     if source and target and target.startswith('class:') and source!=target:edges.append(edge(source,target,'inherits'))
 extra_path=ua/f'tmp/extra-edges-{i}.json'
 if extra_path.exists():edges.extend(json.loads(extra_path.read_text()))
 if missing:
  print('MISSING',i,json.dumps(missing,ensure_ascii=False));continue
 assert len({n['id'] for n in nodes})==len(nodes),(i,'duplicate node')
 assert len({n['filePath'] for n in nodes})==len(b['files'])
 assert all(re.search('[\u4e00-\u9fff]',n['summary']) for n in nodes)
 seen=set();edges=[e for e in edges if e['source']!=e['target'] and not ((e['source'],e['target'],e['type']) in seen or seen.add((e['source'],e['target'],e['type'])))]
 assert sum(e['type']=='imports' for e in edges)==sum(len(x) for x in b['batchImportData'].values())
 parts=max(1,math.ceil(max(len(nodes)/60,len(edges)/120))); filelist=sorted(f['path'] for f in b['files']); width=math.ceil(len(filelist)/parts); groups=[set(filelist[j:j+width]) for j in range(0,len(filelist),width)]
 assigned_edges=0;outputs=[]
 for part,group in enumerate(groups,1):
  ns=[n for n in nodes if n['filePath'] in group]; ids={n['id'] for n in ns}
  allids={n['id'] for n in nodes}
  es=[e for e in edges if e['source'] in ids or (e['source'] not in allids and e['target'] in ids)]
  assigned_edges+=len(es)
  name=f'batch-{i}.json' if len(groups)==1 else f'batch-{i}-part-{part}.json'
  (ua/'intermediate'/name).write_text(json.dumps({'nodes':ns,'edges':es},ensure_ascii=False,indent=2)+'\n');outputs.append(name)
 assert assigned_edges==len(edges)
 print(json.dumps({'batch':i,'files':len(filelist),'nodes':len(nodes),'edges':len(edges),'parts':outputs,'warnings':[]},ensure_ascii=False))
