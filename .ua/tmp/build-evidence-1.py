import ast,json
from pathlib import Path
root=Path('/mnt/data_hdd/mhzhu/GPTCode/PyRUA-Lean')
for i in [1,2,3,4,5,19]:
 result=json.loads((root/f'.ua/tmp/ua-file-extract-results-{i}.json').read_text())
 evidence=[]
 for r in result['results']:
  path=r['path']; source=(root/path).read_text(); lines=source.splitlines()
  rec={'path':path,'category':r['fileCategory'],'moduleDoc':'','symbols':[]}
  if r['language']=='python':
   tree=ast.parse(source); rec['moduleDoc']=ast.get_docstring(tree) or ''
   by_start={n.lineno:n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))}
   def add(symbol,kind,owner=None):
    node=by_start.get(symbol['startLine'])
    if node is None:
     node=next((n for n in by_start.values() if n.name==symbol['name'] and n.lineno in range(symbol['startLine'],symbol['endLine']+1)),None)
    assert node is not None,(path,symbol)
    body=node.body
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str): body=body[1:]
    code='\n'.join(lines[n.lineno-1:n.end_lineno] for n in [])
    snippets=[]
    for n in body:
     # Show statement opening lines; retain full short statements.
     snippet='\n'.join(lines[n.lineno-1:min(n.end_lineno,n.lineno+2)])
     if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
     snippets.append(snippet.strip())
    doc=ast.get_docstring(node) or ''
    name=(owner+'.' if owner else '')+symbol['name']
    rec['symbols'].append({'name':name,'kind':kind,'range':[symbol['startLine'],symbol['endLine']],'owner':owner,'doc':doc,'body':'\n'.join(snippets)[:1400]})
    return node
   for f in r.get('functions',[]):add(f,'function')
   for c in r.get('classes',[]):
    n=add(c,'class')
    for m in n.body:
     if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)) and m.name in c['methods']:
      if m.end_lineno-m.lineno+1>=10 or (not m.name.startswith('_')) or m.name=='__init__':
       add({'name':m.name,'startLine':m.lineno,'endLine':m.end_lineno},'function',c['name'])
   rec['imports']=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
  else:rec['text']=source[:18000]
  evidence.append(rec)
 (root/f'.ua/tmp/evidence-{i}.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
 print(i,len(evidence),sum(len(e['symbols']) for e in evidence))
