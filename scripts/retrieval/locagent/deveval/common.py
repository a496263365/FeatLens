"""DevEval localization-only adapter. Ground truth never enters an LLM/tool request."""

# Locate the package from this file, not from the caller's working directory.
from pathlib import Path as _PortablePath
import sys as _portable_sys
_PORTABLE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents
                      if (p / "featlens_paths.py").is_file())
if str(_PORTABLE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_ROOT))
from featlens_paths import (package_path as _package_path,
                            external_path as _external_path,
                            model_location as _model_location,
                            evaluation_python as _evaluation_python,
                            temporary_path as _temporary_path)

import ast,json,re,hashlib,tokenize,time,importlib.util,os
from pathlib import Path
ROOT=(_package_path());OUT=(_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks')
AGENTLESS=(_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/vendor/Agentless')
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def jl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(s):return s.split('(')[0].replace('.__init__.','.').lstrip('.')
def task_by_id(i):return next(t for t in jl((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/tasks.jsonl')) if t['example_id']==i)
def attempt_dir(t,method):
 import uuid
 if os.environ.get('LOCALIZATION_ATTEMPT_DIR'):
  p=Path(os.environ['LOCALIZATION_ATTEMPT_DIR']).resolve()
  assert p.is_relative_to((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/tasks'))
 else:
  p=(_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/tasks')/f"task_{t['example_id']:06d}"/method/'attempts'/(time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
 p.mkdir(parents=True,exist_ok=True)
 if (p/'result.json').exists() or (p/'api').exists():raise FileExistsError(p)
 return p

def event(path,data):
 import datetime
 payload=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),**data)
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with open(path,'a') as f:f.write(json.dumps(payload,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def prepare(t):
 import fcntl,shutil,uuid
 d=(_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/tasks')/f"task_{t['example_id']:06d}";d.mkdir(parents=True,exist_ok=True)
 with open(d/'prepare.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  m=d/'mask_manifest.json';dst=d/'source'/t['project_path']
  if m.exists():
   meta=json.loads(m.read_text());assert meta['task_id']==t['example_id']
   target=dst/Path(t['completion_path']).relative_to(t['project_path'])
   assert sha(target)==meta['source_masked_sha256']
   return dst
  if (d/'source').exists():shutil.move(str(d/'source'),str(d/('partial_source_'+uuid.uuid4().hex[:8])))
  p=(_package_path() / 'scripts/retrieval/locagent/deveval/task_masking.py');spec=importlib.util.spec_from_file_location('masker',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  repo,audit=mod.prepare((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/source_original/Source_Code'),t,d)
  return repo

def visible_files(repo):
 excluded={'tests','test','test_data','testing','.git','.tox','.venv','venv','__pycache__','build','dist','docs','examples'}
 for p in sorted(Path(repo).rglob('*.py')):
  rel=p.relative_to(repo)
  if set(rel.parts)&excluded or p.name.startswith('test_') or p.name.endswith('_test.py'):continue
  yield p,rel.as_posix()

def structure_and_entities(repo):
 import sys
 if str(AGENTLESS) not in sys.path:sys.path.insert(0,str(AGENTLESS))
 from get_repo_structure.get_repo_structure import parse_python_file
 structure={};entities=[];sources={};errors=[]
 for p,rel in visible_files(repo):
  try:
   with tokenize.open(p) as f:code=f.read()
   tree=ast.parse(code)
  except Exception as e:errors.append(dict(file=rel,error=type(e).__name__));continue
  sources[rel]=code
  classes,funcs,lines=parse_python_file(str(p),file_content=code)
  cur=structure
  for part in rel.split('/')[:-1]:cur=cur.setdefault(part,{})
  cur[rel.split('/')[-1]]={'classes':classes,'functions':funcs,'text':lines}
  parts=rel[:-3].split('/')
  if parts and parts[0]=='src':parts=parts[1:]
  if parts[-1]=='__init__':parts=parts[:-1]
  module='.'.join(parts);lines=code.splitlines()
  def walk(node,prefix=''):
   for n in ast.iter_child_nodes(node):
    if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
     qual=prefix+n.name;kind='class' if isinstance(n,ast.ClassDef) else 'function'
     entities.append(dict(file=rel,name=qual,kind=kind,symbol=(module+'.'+qual).lstrip('.'),start=n.lineno,end=n.end_lineno,code='\n'.join(lines[n.lineno-1:n.end_lineno])))
     walk(n,qual+'.')
    elif isinstance(n,(ast.Assign,ast.AnnAssign)) and not prefix:
     ns=n.targets if isinstance(n,ast.Assign) else [n.target]
     for target in ns:
      for name in ast.walk(target):
       if isinstance(name,ast.Name):entities.append(dict(file=rel,name=name.id,kind='variable',symbol=(module+'.'+name.id).lstrip('.'),start=n.lineno,end=n.end_lineno,code='\n'.join(lines[n.lineno-1:n.end_lineno])))
    else:walk(n,prefix)
  walk(tree)
 return structure,entities,sources,errors

def requirement(t,repo):
 rel=Path(t['completion_path']).relative_to(t['project_path']).as_posix()
 with tokenize.open(Path(repo)/rel) as f:lines=f.read().splitlines()
 a,b=t['signature_position'];sig='\n'.join(lines[a-1:b])
 # No dependency list, tests, expected implementation or reference summaries.
 return f"Dependency localization task, not bug fixing. Identify existing repository functions, classes, APIs or variables that should be reused or inspected to implement the missing target below. Do not generate code or tests. Do not just report the already-known target location.\nTarget file: {rel}\nTarget symbol: {t['namespace']}\nSignature:\n{sig}\nRequirement: {t['requirement']['Functionality']}\nArguments: {t['requirement']['Arguments']}"

def parse_locations(raw,sources,entities,t):
 if isinstance(raw,dict):text='\n'.join(k+'\n'+('\n'.join(v) if isinstance(v,list) else str(v)) for k,v in raw.items())
 else:text=str(raw)
 found_files=[];selected=[];unresolved=[];current=None
 byfile={f:[e for e in entities if e['file']==f] for f in sources}
 for line in text.splitlines():
  line=line.strip().strip('`').strip().strip('"\'')
  if not line or line in {'python','json'}:continue
  m=re.match(r'^(?:[-*]\s*)?(.+?\.py)(?::(.*))?$',line)
  if m:
   path=m.group(1).strip();matches=[f for f in sources if f==path or path.endswith('/'+f)]
   if len(matches)==1:
    current=matches[0]
    if current not in found_files:found_files.append(current)
    tail=m.group(2)
    if not tail:continue
    line='function: '+tail
   else:current=None;unresolved.append(line);continue
  if current is None:continue
  m=re.match(r'^(function|class|variable|method|line):\s*(.+)',line,re.I)
  if not m:continue
  kind,name=m.groups();kind=kind.lower();name=name.strip().split('(')[0]
  if kind=='line':
   nums=[int(n) for n in re.findall(r'\d+',name)]
   for n in nums:
    candidates=[e for e in byfile[current] if e['start']<=n<=e['end']]
    if candidates:selected.append(min(candidates,key=lambda e:e['end']-e['start']))
    else:unresolved.append(current+':'+line)
  else:
   candidates=[e for e in byfile[current] if (e['name']==name or e['symbol']==name) and (kind not in ['function','method'] or e['kind']=='function')]
   if len(candidates)==1:selected+=candidates
   else:unresolved.append(current+':'+line)
 unique={e['symbol']:e for e in selected if canonical(e['symbol'])!=canonical(t['namespace'])}
 return dict(files=found_files,entities=list(unique.values()),unresolved=unresolved)

def score(t,locations,sources):
 deps=set(sum(t['dependency'].values(),[]));assert deps
 selected=locations['entities'];symbols={canonical(e['symbol']) for e in selected};exact={d for d in deps if canonical(d) in symbols}
 # Same legacy-compatible variable/module relaxation, deduplicated. Ground truth only used here.
 p=(_package_path() / 'results/ablation/deveval/feature_and_summary/inputs')/t['project_path']/'report-enre.json'
 enre=json.loads(p.read_text());categories={}
 for v in enre.get('variables',[]):categories.setdefault(v.get('qualifiedName'),set()).add(v.get('category'))
 hits=set(exact)
 for d in deps-hits:
  cats=categories.get(d,set());name=d.rsplit('.',1)[-1]
  if 'Variable' in cats and any(name in e['code'] for e in selected):hits.add(d)
  elif 'Unresolved Attribute' in cats and any(e['symbol'].startswith(d.rsplit('.',1)[0]+'.') and 'self.'+name in e['code'] for e in selected):hits.add(d)
  elif cats&{'Module','Package'} and any(e['symbol'].startswith(d) for e in selected):hits.add(d)
 return dict(dr=len(hits)/len(deps),exact_entity_dr=len(exact)/len(deps),dependency_count=len(deps),hit_dependencies=sorted(hits),exact_hit_dependencies=sorted(exact),returned_entities=len(selected),returned_files=len(locations['files']),unresolved_locations=len(locations['unresolved']),returned_code_chars=sum(len(e['code']) for e in selected))

class Client:
 def __init__(self,logdir,model='deepseek_v3_2',max_calls=20):
  from dotenv import load_dotenv
  from openai import OpenAI
  load_dotenv(_package_path('.env'),override=True)
  self.secret=os.environ['OPENAI_API_KEY']
  self.client=OpenAI(base_url=os.environ['OPENAI_BASE_URL'],api_key=self.secret,timeout=120,max_retries=0)
  self.model=model;self.logdir=Path(logdir);self.logdir.mkdir(parents=True,exist_ok=False)
  self.calls=[];self.max_calls=max_calls;self.logical_calls=0;self.physical=[]
 def safe(self,text):return str(text).replace(self.secret,'[REDACTED_API_KEY]')
 def chat(self,messages,tools=None):
  from openai import APIConnectionError,APITimeoutError
  if self.logical_calls>=self.max_calls:raise RuntimeError('Logical call budget exhausted')
  self.logical_calls+=1;n=self.logical_calls;folder=self.logdir/f'call_{n:03d}';folder.mkdir()
  kw=dict(model=self.model,messages=messages,temperature=0,max_tokens=4096,extra_body={'thinking':{'type':'disabled'}})
  if tools:kw['tools']=tools
  from context_budget import estimate,INPUT,TOTAL,OUTPUT,SAFETY
  estimated=estimate(messages,tools)
  if estimated>INPUT:raise ValueError(f'Request exceeds local context budget: {estimated}>{INPUT}')
  dump(folder/'context_budget.json',dict(estimated_input_tokens=estimated,input_limit=INPUT,total_limit=TOTAL,output_reserve=OUTPUT,safety_margin=SAFETY))
  dump(folder/'request.json',kw)
  for attempt in range(1,3):
   start=time.time();event(self.logdir.parent/'events.jsonl',dict(event='api_request_started',logical_call=n,physical_attempt=attempt,request_file=str(folder/'request.json')))
   try:
    ret=self.client.chat.completions.create(**kw)
   except Exception as e:
    status=getattr(e,'status_code',None);info=dict(logical_call=n,physical_attempt=attempt,status='failed',error_type=type(e).__name__,error=self.safe(e),http_status=status,request_id=getattr(e,'request_id',None),seconds=time.time()-start,usage=None)
    dump(folder/f'attempt_{attempt:02d}_error.json',info);self.physical.append(info);dump(self.logdir.parent/'usage_partial.json',self.usage());event(self.logdir.parent/'events.jsonl',dict(event='api_request_failed',**info))
    retry=isinstance(e,(APIConnectionError,APITimeoutError)) or status in [408,409,429] or (isinstance(status,int) and status>=500)
    if attempt<2 and retry:time.sleep(1);continue
    raise RuntimeError(self.safe(f'{type(e).__name__}: {e}')) from None
   response=ret.model_dump();info=dict(logical_call=n,physical_attempt=attempt,status='succeeded',seconds=time.time()-start,request_id=getattr(ret,'_request_id',None),response=response)
   dump(folder/f'attempt_{attempt:02d}_response.json',info);self.calls.append(info);self.physical.append(info)
   dump(self.logdir.parent/'usage_partial.json',self.usage());event(self.logdir.parent/'events.jsonl',dict(event='api_request_succeeded',logical_call=n,physical_attempt=attempt,seconds=info['seconds'],response_file=str(folder/f'attempt_{attempt:02d}_response.json'),usage=response.get('usage')))
   return ret
 def usage(self):
  success=[p for p in self.physical if p['status']=='succeeded'];usage=[p['response'].get('usage') for p in success];known=[u for u in usage if isinstance(u,dict)]
  failed=[p for p in self.physical if p['status']=='failed']
  return dict(calls=self.logical_calls,successful_calls=len(success),physical_requests=len(self.physical),failed_requests=len(failed),prompt_tokens=sum(u.get('prompt_tokens',0) or 0 for u in known),completion_tokens=sum(u.get('completion_tokens',0) or 0 for u in known),total_tokens=sum(u.get('total_tokens',0) or 0 for u in known),seconds=sum(p['seconds'] for p in self.physical),unknown_usage_requests=len(failed)+len(usage)-len(known),usage_scope='Known returned usage only; failed/time-out request billing unknown',sdk_automatic_retries=0,max_physical_attempts_per_logical_call=2)
