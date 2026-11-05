import collections
import hashlib
import json
from pathlib import Path
import re

root = Path('/mnt/data_hdd/mhzhu/GPTCode/PyRUA-Lean')
ua = root / '.ua'
scan = json.loads((ua / 'intermediate/scan-result.json').read_text())
graph = json.loads((ua / 'intermediate/assembled-graph.json').read_text())
inventory = {f['path']: f for f in scan['files']}
nodes = graph['nodes']
edges = graph['edges']
node_ids = {n['id'] for n in nodes}
file_types = {'file', 'config', 'document', 'service', 'pipeline', 'table', 'schema', 'resource', 'endpoint'}
root_nodes = {n['filePath']: n for n in nodes if n['type'] in file_types and n.get('id') == n['type'] + ':' + n.get('filePath', '')}
issues = []
warnings = []
if len(node_ids) != len(nodes):
    issues.append('Duplicate node IDs')
for path in sorted(set(inventory) - set(root_nodes)):
    issues.append('Missing file-level node: ' + path)
for n in nodes:
    if not n.get('summary') or not n.get('tags'):
        issues.append('Empty summary/tags: ' + n['id'])
    if not re.search('[\u4e00-\u9fff]', n.get('summary', '')):
        issues.append('Missing Chinese summary: ' + n['id'])
    path = n.get('filePath')
    if path and path not in inventory:
        issues.append('Path outside inventory: ' + n['id'])
    if path and n.get('lineRange'):
        start, end = n['lineRange']
        actual_lines = len((root / path).read_text().splitlines())
        if not (1 <= start <= end <= actual_lines + 1):
            issues.append(f'Invalid line range: {n["id"]}: {start}-{end}/{actual_lines}')
for e in edges:
    if e['source'] not in node_ids or e['target'] not in node_ids:
        issues.append('Dangling edge: ' + repr(e))
    if e['source'] == e['target']:
        issues.append('Self edge: ' + repr(e))
edge_keys = {(e['source'], e['target'], e['type']) for e in edges}
if len(edge_keys) != len(edges):
    issues.append('Duplicate edges')
missing_imports = []
for source, targets in scan['importMap'].items():
    for target in targets:
        if ('file:' + source, 'file:' + target, 'imports') not in edge_keys:
            missing_imports.append([source, target])
if missing_imports:
    issues.append(f'Missing imports: {missing_imports}')

# Compare declarations against the official structure output. This checks
# semantic output coverage without substituting a parser for the official one.
by_path_name = {(n.get('filePath'), n['type'], n['name'].split('.')[-1]) for n in nodes}
missing_symbols = []
for result_path in [*(ua / 'tmp').glob('ua-file-extract-results-*.json'), *(ua / 'tmp').glob('docs-tests-extract-*.json')]:
    data = json.loads(result_path.read_text())
    for f in data.get('results', []):
        exports = {e['name'] for e in f.get('exports', [])}
        for fn in f.get('functions', []):
            if fn['endLine'] - fn['startLine'] + 1 >= 10 or fn['name'] in exports:
                if (f['path'], 'function', fn['name']) not in by_path_name:
                    missing_symbols.append([f['path'], 'function', fn['name']])
        for cls in f.get('classes', []):
            if cls['endLine'] - cls['startLine'] + 1 >= 20 or len(cls.get('methods', [])) >= 2 or cls['name'] in exports:
                if (f['path'], 'class', cls['name']) not in by_path_name:
                    missing_symbols.append([f['path'], 'class', cls['name']])
if missing_symbols:
    issues.append(f'Missing significant symbols: {len(missing_symbols)}')

snapshot_path = ua / 'tmp/source-hashes-current.json'
if not snapshot_path.exists():
    snapshot_path = ua / 'tmp/source-hashes-before.json'
snapshot = json.loads(snapshot_path.read_text())
changed = [path for path, digest in snapshot.items() if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest]
if changed:
    issues.append('Source files changed during analysis: ' + repr(changed))

report = {
    'issues': issues, 'warnings': warnings,
    'filesExpected': len(inventory), 'fileNodes': len(root_nodes),
    'totalNodes': len(nodes), 'totalEdges': len(edges),
    'nodeTypes': dict(collections.Counter(n['type'] for n in nodes)),
    'edgeTypes': dict(collections.Counter(e['type'] for e in edges)),
    'missingSymbols': missing_symbols, 'changedSources': changed,
}
(ua / 'intermediate/integrity-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(bool(issues))
