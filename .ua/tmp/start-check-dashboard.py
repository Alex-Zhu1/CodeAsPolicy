import json, os, re, subprocess, time, urllib.request
from pathlib import Path

root=Path('/mnt/data_hdd/mhzhu/GPTCode/PyRUA-Lean')
ua=root/'.ua'
assert (ua/'knowledge-graph.json').is_file()
with (ua/'dashboard.log').open('a') as log:
    process=subprocess.Popen(['bash',str(ua/'start-dashboard.sh')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(ua/'dashboard.pid').write_text(str(process.pid)+'\n')
deadline=time.monotonic()+40
url=None
while time.monotonic()<deadline:
    if process.poll() is not None:
        raise RuntimeError(f'Dashboard exited: {process.returncode}; see {ua}/dashboard.log')
    matches=re.findall(r'Dashboard URL: (http://127\.0\.0\.1:\d+/\?token=[a-zA-Z0-9_-]+)',(ua/'dashboard.log').read_text())
    if matches:
        url=matches[-1]
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                html=response.read().decode(); assert response.status==200 and '<div id="root"></div>' in html
            break
        except (OSError,AssertionError): pass
    time.sleep(.2)
else:
    raise RuntimeError('Dashboard startup timed out')
base,token=url.split('/?token=')
checks={}
for endpoint in ['knowledge-graph.json','config.json']:
    with urllib.request.urlopen(f'{base}/{endpoint}?token={token}',timeout=10) as response:
        data=json.load(response)
        checks[endpoint]={'status':response.status}
        if endpoint=='knowledge-graph.json':
            local=json.loads((ua/endpoint).read_text())
            assert data==local,'Dashboard returned graph from a different project'
            checks[endpoint].update(nodes=len(data['nodes']),edges=len(data['edges']),layers=len(data['layers']),tourSteps=len(data['tour']))
        else:
            assert data.get('outputLanguage')=='zh'
            checks[endpoint]['outputLanguage']=data['outputLanguage']
with urllib.request.urlopen(f'{base}/src/main.tsx',timeout=20) as response:
    script=response.read().decode()
    assert response.status==200 and 'createRoot' in script
    checks['frontend-entry']={'status':response.status}
result={'url':url,'pid':process.pid,'checks':checks}
(ua/'dashboard.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
