"""Publish joint main table only after both corrected UniXcoder and BM25 finish."""

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

import json,csv,time
from pathlib import Path
import numpy as np
O=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder');B=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph')
def alive(script):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if str((_package_path() / 'scripts/ablation/bm25_and_unixcoder')/script).encode() in (p/'cmdline').read_bytes().split(b'\0'):return True
  except OSError:pass
 return False
while not (((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/COMPLETION_VERIFIED.json')).exists() and ((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/run_finished.json')).exists()):
 if not ((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/COMPLETION_VERIFIED.json')).exists() and not (alive('run_corrected.py') or alive('finalize.py')):raise SystemExit('UniXcoder main result not verified; no worker remains')
 if not ((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/run_finished.json')).exists() and not alive('run_bm25_graph.py'):raise SystemExit('BM25 graph not completed; no worker remains')
 time.sleep(20)
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def canon(s):return str(s).split('(',1)[0].lstrip('.').replace('.__init__.','.')
projects=json.loads(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/primary_scope.json')).read_text())['projects'];main=jl((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/combined_task_metrics.jsonl'));lookup={(r['task_id'],r['arm'],r['stage']):r for r in main}
assert len(main)==1146*19
new=[];checks=0
for project in projects:
 d=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/project_results')/project;done=json.loads((d/'run_completed.json').read_text());assert done['graph_encoder']=='unchanged original BGE-code-v1' and done['same_feature'] is True
 rows=jl(d/'task_metrics.jsonl');assert len(rows)==9*done['tasks']
 local={(r['task_id'],r['arm'],r['stage']):r for r in rows};assert len(local)==len(rows)
 for r in rows:
  key=(r['task_id'],r['arm'],r['stage'])
  assert 0<=r['dr']<=1 and abs(r['dr']-len(set(r['hits']))/len(r['dependencies']))<1e-12
  assert all(canon(s)!=canon(r['namespace']) for s in r['predictions'])
  if r['stage']=='direct15':
   old=lookup[key];assert old['predictions']==r['predictions'] and old['dr']==r['dr'];checks+=1
  else:
   if r['stage']=='seed':assert r['n']==lookup[(r['task_id'],'feature_promote','seed')]['n']
   else:assert r['stage']=='final15' and r['n']<=15
   assert key not in lookup;new.append(r)
rows=main+new;assert len(rows)==25*1146 and len(new)==6*1146 and checks==3*1146
arms=['feature_promote','feature_no_promote','feature_no_target_feature','summary_dense','summary_dense_no_samefeature','code_dense_unixcoder','code_dense_unixcoder_no_samefeature','bm25_description_top15','bm25_code_top15','bm25_signature_top15']
labels=['FeatLens：位置提升✓，same-feature✓','Feature：位置提升✗，same-feature✓','Feature：位置提升✗，same-feature✗','摘要dense：same-feature✓','摘要dense：same-feature✗','代码UniXcoder：same-feature✓','代码UniXcoder：same-feature✗','BM25摘要：same-feature✓','BM25代码：same-feature✓','BM25签名：same-feature✓']
summary=[]
def mean(rr,k):return float(np.mean([r[k] for r in rr])) if rr else None
for a,label in zip(arms,labels):
 seed=[r for r in rows if r['arm']==a and r['stage']=='seed'];final=[r for r in rows if r['arm']==a and r['stage']=='final15']
 da='summary_dense_top15' if a.startswith('summary_dense') else 'code_dense_unixcoder_top15' if a.startswith('code_dense_unixcoder') else a if a.startswith('bm25') else None
 direct=[r for r in rows if r['arm']==da and r['stage']=='direct15'] if da else []
 assert len(seed)==len(final)==1146
 if da:assert len(direct)==1146
 summary.append(dict(method=label,arm=a,tasks=1146,projects=90,direct15_dr=mean(direct,'dr'),seed_dr=mean(seed,'dr'),final15_dr=mean(final,'dr'),mean_seed_nodes=mean(seed,'n'),mean_final_nodes=mean(final,'n'),final_exact_dr=mean(final,'exact_dr')))
with open((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/ablation_metrics.csv'),'w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/joint_task_metrics.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
lines=['# 原90项目联合消融大表（图设置固定）','', '已完成：90项目、1146个DR有效任务。284个无依赖任务不计DR。UniXcoder与BM25新增图实验均通过核验后才生成本表。','', '| 条件 | 直接Top-15 DR | 初始Seed DR | 原图处理后Top-15 DR |','|---|---:|---:|---:|']
for r in summary:
 vals=['—' if r[k] is None else f"{r[k]:.2%}" for k in ['direct15_dr','seed_dr','final15_dr']]
 lines.append('| '+r['method']+' | '+' | '.join(vals)+' |')
lines += ['', '## 固定设置','- 所有图方法共用原FeatLens图扩展、BGE-code-v1语义评分和PPR；不是把图编码器换成UniXcoder的那轮。','- 代码候选检索使用原UniXcoder；Feature/摘要候选检索保留MiniLM；BM25保留原分词与BM25Okapi默认参数。','- BM25三组图实验均保留same-feature及same-file加分，仅检索策略变化。','- BM25与dense初始种子数量逐任务匹配带位置提升的Feature种子；直接Top-15和Seed不是同一预算。','- 最终最多15个节点，不是等token预算。similar-method提示在全部条件中禁用。','- 旧索引目标隔离仍未核实；本次是统一候选/评估口径的检索对照，不是原论文旧指标直接拼接，也不测Pass@1。']
((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/联合消融大表.md')).write_text('\n'.join(lines))
verification=dict(projects=90,tasks=1146,records=len(rows),bm25_direct_reproduction_checks=checks,bm25_seed_budget_matched=True,graph_configuration_fixed=True,all_ten_rows_complete=True)
((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/JOINT_COMPLETION_VERIFIED.json')).write_text(json.dumps(verification,indent=2))
print(json.dumps(verification,indent=2));print(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/联合消融大表.md')).read_text())
