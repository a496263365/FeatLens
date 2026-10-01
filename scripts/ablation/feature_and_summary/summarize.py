"""Aggregate only complete tasks; report full benchmark coverage explicitly."""

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

import json,csv,hashlib,collections,sys
from pathlib import Path
import numpy as np
O=(_package_path() / 'results/ablation/deveval/feature_and_summary')
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2))
def csvout(p,rs):
 if not rs:return
 with open(p,'w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def canon(s):return str(s).split('(',1)[0].lstrip('.').replace('.__init__.','.')
tasks=jl((_package_path() / 'data/deveval/tasks_all_1430.jsonl'));eligible={t['example_id']:t for t in tasks if any(t['dependency'].values())}
plans=json.loads(((_package_path() / 'results/ablation/deveval/feature_and_summary/project_plan.json')).read_text())
scope=json.loads(((_package_path() / 'results/ablation/deveval/feature_and_summary/primary_scope.json')).read_text());selected=set(scope['projects'])
plans=[p for p in plans if p['project'] in selected]
tasks=[t for t in tasks if t['project_path'] in selected]
eligible={i:t for i,t in eligible.items() if t['project_path'] in selected}
assert len(plans)==90 and len(tasks)==1430 and len(eligible)==1146
plan={p['project']:p for p in plans}
arms=['feature_promote','feature_no_promote','feature_no_target_feature','summary_dense','summary_dense_no_samefeature','code_dense_bge','code_dense_bge_no_samefeature']
expected={(a,s) for a in arms for s in ['seed','final15']}|{('summary_dense_top15','direct15'),('code_dense_bge_top15','direct15')}
allrows=[];complete=[];incomplete=[];audits=[]
for project in sorted(plan):
 p=(_package_path() / 'results/ablation/deveval/feature_and_summary/project_results')/project;req=plan[project]['eligible_tasks']
 if not req:continue
 if not (p/'run_completed.json').exists():
  partial=jl(p/'task_metrics.jsonl') if (p/'task_metrics.jsonl').exists() else []
  incomplete.append(dict(project=project,eligible_tasks=req,partial_metric_rows=len(partial)));continue
 rs=jl(p/'task_metrics.jsonl');by=collections.defaultdict(list)
 assert len(rs)==16*req,(project,len(rs),req)
 for r in rs:by[r['task_id']].append(r)
 assert len(by)==req
 for tid,rr in by.items():
  assert tid in eligible and eligible[tid]['project_path']==project
  assert len(rr)==16 and {(r['arm'],r['stage']) for r in rr}==expected
  dep=set(sum(eligible[tid]['dependency'].values(),[]))
  for r in rr:
   assert set(r['dependencies'])==dep
   assert 0<=r['dr']<=1 and abs(r['dr']-len(set(r['hits']))/len(dep))<1e-12
   assert len(r['predictions'])==r['n']
   assert all(canon(s)!=canon(r['namespace']) for s in r['predictions'])
   if r['stage'] in ['final15','direct15']:assert r['n']<=15
  local={(r['arm'],r['stage']):r for r in rr}
  for a,b in [('summary_dense','summary_dense_no_samefeature'),('code_dense_bge','code_dense_bge_no_samefeature'),('feature_no_promote','feature_no_target_feature')]:
   assert local[(a,'seed')]['predictions']==local[(b,'seed')]['predictions']
  for a in ['summary_dense','code_dense_bge']:
   assert local[(a,'seed')]['n']==local[('feature_promote','seed')]['n']
 if '--verify-inputs' in sys.argv:
  m=json.loads((p/'manifest.json').read_text())
  for f,h in m['input_hashes'].items():assert hashlib.sha256(((_package_path() / 'results/ablation/deveval/feature_and_summary/inputs')/project/f).read_bytes()).hexdigest()==h
 for r in rs:r['project_path']=project;r['cohort']='historical_cached' if plan[project]['cached_index_ready'] else 'newly_built'
 allrows.extend(rs);complete.append(project)
 audits.extend(json.loads((p/'audit.json').read_text()))
finished_ids={r['task_id'] for r in allrows};assert len(allrows)==16*len(finished_ids)
coverage=dict(benchmark_tasks=len(tasks),benchmark_projects=len(plans),undefined_dr_tasks=len(tasks)-len(eligible),eligible_tasks=len(eligible),eligible_projects=sum(p['eligible_tasks']>0 for p in plans),completed_tasks=len(finished_ids),completed_projects=len(complete),remaining_tasks=len(eligible)-len(finished_ids),remaining_projects=incomplete,all_eligible_complete=len(finished_ids)==len(eligible),completed_historical_cached_tasks=sum(1 for tid in finished_ids if plan[eligible[tid]['project_path']]['cached_index_ready']),completed_newly_built_tasks=sum(1 for tid in finished_ids if not plan[eligible[tid]['project_path']]['cached_index_ready']))
dump((_package_path() / 'results/ablation/deveval/feature_and_summary/coverage.json'),coverage)
lookup={(r['task_id'],r['arm'],r['stage']):r for r in allrows}
def mean(rs,k):return float(np.mean([r[k] for r in rs])) if rs else None
names={'feature_promote':'Feature（优先入选+same-feature）','feature_no_promote':'Feature（无优先入选，有same-feature）','feature_no_target_feature':'Feature（两者均关闭）','summary_dense':'摘要dense（有same-feature）','summary_dense_no_samefeature':'摘要dense（无same-feature）','code_dense_bge':'BGE代码dense（有same-feature）','code_dense_bge_no_samefeature':'BGE代码dense（无same-feature）'}
summary=[]
for cohort in ['all_completed','historical_cached','newly_built']:
 rr=allrows if cohort=='all_completed' else [r for r in allrows if r['cohort']==cohort]
 if not rr:continue
 for arm in arms:
  seed=[r for r in rr if r['arm']==arm and r['stage']=='seed'];final=[r for r in rr if r['arm']==arm and r['stage']=='final15']
  direct_arm='summary_dense_top15' if arm.startswith('summary') else ('code_dense_bge_top15' if arm.startswith('code') else None)
  direct=[r for r in rr if r['arm']==direct_arm and r['stage']=='direct15'] if direct_arm else []
  summary.append(dict(cohort=cohort,arm=arm,tasks=len(seed),projects=len({r['project_path'] for r in seed}),direct15_dr=mean(direct,'dr'),seed_dr=mean(seed,'dr'),final15_dr=mean(final,'dr'),seed_mean_nodes=mean(seed,'n'),final_mean_nodes=mean(final,'n'),final_exact_dr=mean(final,'exact_dr'),final_micro_dr=sum(len(r['hits']) for r in final)/sum(len(r['dependencies']) for r in final),final_mean_code_chars=mean(final,'chars')))
csvout((_package_path() / 'results/ablation/deveval/feature_and_summary/overall_comparison.csv'),summary)
labels=['# DevEval historical 90-project seven-condition evaluation','', '**COMPLETE**' if coverage['all_eligible_complete'] else '**PARTIAL — original 90-project evaluation still running.**','',f"Benchmark: {len(tasks)} tasks / {len(plans)} projects. DR eligible: {len(eligible)} tasks. Completed: {len(finished_ids)} tasks / {len(complete)} projects. Undefined DR (no reference dependencies): {len(tasks)-len(eligible)} tasks.", '', '| 条件 | 直接Top-15 DR | Seed平均节点数 | Seed DR | Final Top-15 DR |','|---|---:|---:|---:|---:|']
for r in summary:
 if r['cohort']!='all_completed':continue
 direct='—' if r['direct15_dr'] is None else f"{r['direct15_dr']:.2%}"
 labels.append(f"| {names[r['arm']]} | {direct} | {r['seed_mean_nodes']:.2f} | {r['seed_dr']:.2%} | {r['final15_dr']:.2%} |")
labels+=['','Task-macro unique-hit relaxed DR, not Pass@1. All conditions paired on the same completed tasks. Zero-dependency tasks are undefined, not failures or perfect recalls. Feature direct Top-15 was not evaluated. Empty mapped seeds return empty context and are retained. Dense seed size follows promoted-feature seed, not equal token budget. Same-file bonus retained; body-based similar-method hint disabled.','', 'Primary scope is the original 90 historical projects, checked against the archived stage summary. The seven supplementary projects are excluded by user instruction; completed or partial supplementary artifacts are retained separately. Cached-index diagnostic, NOT a verified target-isolated or exact paper reproduction. Original DevEval has 1825 tasks; this selected subset has 1430, of which 1146 have defined DR.','', '## Remaining projects','```json',json.dumps(incomplete,ensure_ascii=False,indent=2),'```']
((_package_path() / 'results/ablation/deveval/feature_and_summary/REPORT.md')).write_text('\n'.join(labels))
# Per-project numerical records for checking heterogeneity.
pr=[]
for project in complete:
 for arm in arms:
  rs=[r for r in allrows if r['project_path']==project and r['arm']==arm and r['stage']=='final15']
  pr.append(dict(project=project,arm=arm,tasks=len(rs),final_dr=mean(rs,'dr'),exact_dr=mean(rs,'exact_dr')))
csvout((_package_path() / 'results/ablation/deveval/feature_and_summary/per_project.csv'),pr)
dump((_package_path() / 'results/ablation/deveval/feature_and_summary/audit_summary.json'),dict(tasks=len(audits),empty_promoted_seeds=sum(a['seed_budget']==0 for a in audits),target_feature_found=sum(a['target_feature_found'] for a in audits),target_code_nonempty=sum(a['raw_target_code_chars']>0 for a in audits)))
print(json.dumps({k:v for k,v in coverage.items() if k!='remaining_projects'},indent=2))
if coverage['all_eligible_complete']:
 dump((_package_path() / 'results/ablation/deveval/feature_and_summary/COMPLETION_VERIFIED.json'),dict(**coverage,records=len(allrows),all_conditions_verified=True,input_hashes_verified='--verify-inputs' in sys.argv))
