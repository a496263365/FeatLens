
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

import json,csv,time,os,argparse,hashlib
from pathlib import Path
O=(_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks');CFG=json.loads(((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/config.json')).read_text());TS=[json.loads(s) for s in ((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/tasks.jsonl')).read_text().splitlines()];T={t['example_id']:t for t in TS}
def dump(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def csvout(p,rows):
 if not rows:return
 with open(p,'w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def alive():
 try:
  pid=json.loads(((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/controller_identity.json')).read_text())['pid']
  return str((_package_path() / 'scripts/retrieval/locagent/deveval/controller.py')).encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
 except (OSError,KeyError,json.JSONDecodeError):return False
p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
if args.wait:
 while not ((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/ALL_JOBS_TERMINAL.json')).exists():
  if not alive():dump((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/SUMMARY_WAIT_INTERRUPTED.json'),dict(reason='controller no longer live and no terminal marker',timestamp=time.time()));raise SystemExit(1)
  time.sleep(60)
rows=[];outcomes={};errors=[]
for tid,t in T.items():
 for method in CFG['methods']:
  md=(_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/tasks')/f'task_{tid:06d}'/method;sp=md/'job_summary.json'
  if not sp.exists():continue
  s=json.loads(sp.read_text());assert s['task_id']==tid and s['method']==method;outcomes[(tid,method)]=s
  if s['state'] not in ['completed','budget_exhausted','context_limited']:continue
  dest=Path(s['attempt_dir']);r=json.loads((dest/'result.json').read_text());f=r['final_stage'];assert r['namespace']==t['namespace'];assert 0<=f['dr']<=1
  deps=set(sum(t['dependency'].values(),[]));assert abs(f['dr']-len(set(f['hit_dependencies']))/len(deps))<1e-12
  locs=r['final_locations']['entities'];assert all(e['symbol'].split('(')[0]!=t['namespace'] for e in locs)
  successes=list((dest/'api').glob('call_*/attempt_*_response.json'));failures=list((dest/'api').glob('call_*/attempt_*_error.json'))
  assert len(successes)==r['usage']['successful_calls'] and len(failures)==r['usage']['failed_requests']
  for call in (dest/'api').glob('call_*'):assert (call/'request.json').exists()
  source_manifest=json.loads((md.parent/'mask_manifest.json').read_text());target=md.parent/'source'/t['completion_path'];assert hashlib.sha256(target.read_bytes()).hexdigest()==source_manifest['source_masked_sha256']
  rows.append(dict(task_id=tid,project=t['project_path'],namespace=t['namespace'],method=method,status=s['state'],dr=f['dr'],exact_dr=f['exact_entity_dr'],entities=f['returned_entities'],files=f['returned_files'],unresolved=f['unresolved_locations'],prompt_tokens=r['usage']['prompt_tokens'],completion_tokens=r['usage']['completion_tokens'],physical_requests=r['usage']['physical_requests'],unknown_usage_requests=r['usage']['unknown_usage_requests'],seconds=s['seconds'],attempt_dir=str(dest)))
summary=[]
for method in CFG['methods']:
 rs=[r for r in rows if r['method']==method];ss=[s for (i,m),s in outcomes.items() if m==method];n=len(rs)
 summary.append(dict(method=method,expected_tasks=len(T),terminal_tasks=len(ss),scored_tasks=n,failed_tasks=len(ss)-n,pending_tasks=len(T)-len(ss),budget_exhausted_tasks=sum(r['status']=='budget_exhausted' for r in rs),completed_only_mean_dr=sum(r['dr'] for r in rs)/n if n else None,completed_only_exact_dr=sum(r['exact_dr'] for r in rs)/n if n else None,mean_entities=sum(r['entities'] for r in rs)/n if n else None,known_prompt_tokens_all_terminal_attempts=sum(s.get('usage',{}).get('prompt_tokens',0) for s in ss),known_completion_tokens_all_terminal_attempts=sum(s.get('usage',{}).get('completion_tokens',0) for s in ss),unknown_usage_requests=sum(s.get('usage',{}).get('unknown_usage_requests',0) for s in ss)))
csvout((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/task_results.csv'),rows);csvout((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/summary.csv'),summary)
paired={r['task_id'] for r in rows}
coverage=dict(expected_tasks_per_method=len(T),expected_jobs=len(CFG['methods'])*len(T),terminal_jobs=len(outcomes),verified_scored_jobs=len(rows),paired_scored_tasks=len(paired),all_jobs_terminal=len(outcomes)==len(CFG['methods'])*len(T),all_jobs_scored=len(rows)==len(CFG['methods'])*len(T),methods=summary,failures=[s for s in outcomes.values() if s['state'] not in ['completed','budget_exhausted','context_limited']])
dump((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/coverage_verified.json'),coverage)
lines=['# DeepSeek-V3.2 定位全量运行','', '全任务已结束。' if coverage['all_jobs_terminal'] else '**运行中：以下不是全量最终分数。**','', '|方法|已评分/1146|失败|待运行/运行中|预算耗尽|已完成任务平均DR|','|---|---:|---:|---:|---:|---:|']
for s in summary:
 dr='—' if s['completed_only_mean_dr'] is None else f"{s['completed_only_mean_dr']:.2%}"
 lines.append(f"|{s['method']}|{s['scored_tasks']}|{s['failed_tasks']}|{s['pending_tasks']}|{s['budget_exhausted_tasks']}|{dr}|")
lines+=['','所有失败和预算耗尽单列。若未全部评分，不将完成子集均值称为全量结果。token仅为已知API返回用量，失败计费可能未知。历史适配、文件范围差异和映射局限保持不变，见PROTOCOL.md。']
((_package_path() / 'results/retrieval/deveval/locagent/dependency_tasks/REPORT.md')).write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in coverage.items() if k not in ['methods','failures']},indent=2))
