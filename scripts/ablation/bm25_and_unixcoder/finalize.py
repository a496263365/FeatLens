
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

import json,time,csv,sys,collections
from pathlib import Path
import numpy as np
O=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder');R=(_package_path());OLD=(_package_path() / 'results/ablation/deveval/feature_and_summary');UX=R/'results/deveval90_unixcoder_bm25_20260928'
while not ((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/run_finished.json')).exists():
 live=False
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if str((_package_path() / 'scripts/ablation/bm25_and_unixcoder/run_corrected.py')).encode() in (proc/'cmdline').read_bytes().split(b'\0'):live=True;break
  except OSError:pass
 if not live:raise SystemExit('Corrected worker exited before completion; do not publish a partial main table')
 time.sleep(15)
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def dump(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
projects=json.loads(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/primary_scope.json')).read_text())['projects'];assert len(projects)==90
rows=[];verified=0
for p in projects:
 dest=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/project_results')/p;done=json.loads((dest/'run_completed.json').read_text());assert done['graph_encoder']=='original BGE-code-v1'
 old=jl((_package_path() / 'results/ablation/deveval/feature_and_summary/project_results')/p/'task_metrics.jsonl');ux=jl(UX/'project_results'/p/'task_metrics.jsonl');fixed=jl(dest/'task_metrics.jsonl')
 old_idx={(r['task_id'],r['arm'],r['stage']):r for r in old};ux_idx={(r['task_id'],r['arm'],r['stage']):r for r in ux}
 ts=jl((_package_path() / 'results/ablation/deveval/feature_and_summary/inputs')/p/'filtered.jsonl');assert len(fixed)==2*len(ts)
 # Reuse only invariant old graph arms and MiniLM direct reference.
 keep={'feature_promote','feature_no_promote','feature_no_target_feature','summary_dense','summary_dense_no_samefeature','summary_dense_top15'}
 selected=[r for r in old if r['arm'] in keep]
 selected += [r for r in ux if (r['arm'] in {'code_dense_unixcoder','code_dense_unixcoder_no_samefeature'} and r['stage']=='seed') or r['arm'] in {'code_dense_unixcoder_top15','bm25_description_top15','bm25_code_top15','bm25_signature_top15'}]
 selected+=fixed
 for t in ts:
  i=t['example_id']
  for a in ['feature_promote','feature_no_promote','feature_no_target_feature','summary_dense','summary_dense_no_samefeature']:
   assert old_idx[(i,a,'seed')]['predictions']==ux_idx[(i,a,'seed')]['predictions'];verified+=1
  for a in ['code_dense_unixcoder','code_dense_unixcoder_no_samefeature']:
   assert ux_idx[(i,a,'seed')]['n']==old_idx[(i,'feature_promote','seed')]['n']
 assert len(selected)==19*len(ts)
 assert len({(r['task_id'],r['arm'],r['stage']) for r in selected})==len(selected)
 for r in selected:
  assert 0<=r['dr']<=1
  assert abs(r['dr']-len(set(r['hits']))/len(r['dependencies']))<1e-12
  r=dict(r,project_path=p);rows.append(r)
assert len(rows)==19*1146 and len({r['task_id'] for r in rows})==1146
arms=['feature_promote','feature_no_promote','feature_no_target_feature','summary_dense','summary_dense_no_samefeature','code_dense_unixcoder','code_dense_unixcoder_no_samefeature','bm25_description_top15','bm25_code_top15','bm25_signature_top15']
summary=[]
def avg(rr,key):return float(np.mean([r[key] for r in rr])) if rr else None
for a in arms:
 seed=[r for r in rows if r['arm']==a and r['stage']=='seed'];final=[r for r in rows if r['arm']==a and r['stage']=='final15']
 da='summary_dense_top15' if a.startswith('summary_dense') else 'code_dense_unixcoder_top15' if a.startswith('code_dense_unixcoder') else a if a.startswith('bm25') else None
 direct=[r for r in rows if r['arm']==da and r['stage']=='direct15'] if da else []
 summary.append(dict(arm=a,tasks=1146,projects=90,direct15_dr=avg(direct,'dr'),seed_dr=avg(seed,'dr'),final15_dr=avg(final,'dr'),mean_seed_nodes=avg(seed,'n'),graph_semantic_encoder='original BGE-code-v1' if seed else 'not applicable'))
with open((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/comparison_90projects.csv'),'w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/combined_task_metrics.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
labels=['Feature：位置提升✓，same-feature✓','Feature：位置提升✗，same-feature✓','Feature：位置提升✗，same-feature✗','摘要dense：same-feature✓','摘要dense：same-feature✗','代码UniXcoder：same-feature✓','代码UniXcoder：same-feature✗','BM25摘要','BM25代码','BM25签名']
lines=['# 固定原FeatLens图处理的对照（修正版）','', '90个原项目、1146个DR有效任务全部完成。代码候选检索改为UniXcoder；所有图处理保留原FeatLens图评分后端和参数，未随候选检索替换图编码器。','', '| 条件 | 直接Top-15 DR | Seed DR | 原图处理后Top-15 DR |','|---|---:|---:|---:|']
for label,r in zip(labels,summary):
 vals=['—' if r[k] is None else f"{r[k]:.2%}" for k in ['direct15_dr','seed_dr','final15_dr']]
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','仅在指定的消融条件中关闭位置提升或same-feature；其余固定。BM25三行不接图，所以Seed/Final不适用。摘要与feature检索使用原MiniLM，代码检索使用UniXcoder，图评分保留原BGE-code-v1。','', '复用不变阶段的已验证结果，只重算两组UniXcoder种子的原图处理。统一候选池/目标排除/逐任务DR口径，不是原论文旧汇总的直接复现。旧索引目标隔离仍未核实。']
((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/REPORT.md')).write_text('\n'.join(lines))
dump('COMPLETION_VERIFIED.json',dict(projects=90,tasks=1146,records=len(rows),invariant_seed_records_checked=verified,graph_encoder_held_fixed=True,only_two_code_seed_graph_arms_recomputed=True,all_comparisons_complete=True))
print(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/REPORT.md')).read_text())
