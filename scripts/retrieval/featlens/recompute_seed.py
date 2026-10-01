
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

import json,csv,ast,hashlib,importlib.util,collections,time
from pathlib import Path
import networkx as nx
O=(_package_path() / 'results/retrieval/combined/featlens_scores/seed_scores');P=(_package_path() / 'results/retrieval/combined/featlens_scores');B=(_package_path() / 'results/retrieval/deveval/featlens_and_rag')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
metrics=load('new_helpers',(_package_path() / 'scripts/retrieval/featlens/dependency_matching.py'));legacy=load('old_graph',(_package_path() / 'scripts/retrieval/featlens/reference_graph_scorer.py'));legacy.DEBUG=False
root=ast.parse(((_package_path() / 'scripts/retrieval/featlens/scoring_source.py')).read_text());funcs=[n for n in root.body if isinstance(n,ast.FunctionDef) and n.name in ['norm','canon','score_context']];ns={'metrics':metrics};exec(compile(ast.Module(body=funcs,type_ignores=[]),'unchanged_new_score','exec'),ns);score=ns['score_context'];canon=ns['canon']
archived={r['project_name']:r for r in csv.DictReader(open((_package_path() / 'results/retrieval/combined/featlens_scores/graph_scope_reference.csv'))) if r['project_name']!='ALL'}
projects=list(csv.DictReader(open((_package_path() / 'results/retrieval/combined/featlens_scores/per_project.csv'))));rows=[];project_rows=[];hashes={};keys=collections.Counter();feature_file_inventory=[];start=time.time()
for pr in projects:
 project=pr['project'];p=B/project;tasks=[json.loads(s) for s in (_package_path('data/deveval/projects')/project/'filtered.jsonl').read_text().splitlines() if s.strip()]
 metrics.clear_enre_elements();metrics.load_enre_elements(str(p/'report-enre.json'));legacy.clear_enre_elements();legacy.load_enre_elements(str(p/'report-enre.json'));local=[]
 feature=p/'diagnostic_***feature.jsonl';hashes[str(feature)]=sha(feature)
 records=[json.loads(s) for s in feature.read_text().splitlines() if s.strip()];assert len(records)==len(tasks)
 found_keys=set()
 for r in records:
  found_keys.update(r.get('feature',{}).keys());keys.update(r.get('feature',{}).keys())
 feature_file_inventory.append(dict(project=project,records=len(records),feature_result_keys=sorted(found_keys),has_feature_top15='top15' in found_keys))
 for i,t in enumerate(tasks,1):
  f=p/'graphs'/f'task_{i}_ori.gml';assert f.exists();hashes[str(f)]=sha(f);g=nx.read_gml(f);ctx=[dict(a) for _,a in g.nodes(data=True)]
  dep=sum([t['dependency'].get(k,[]) for k in ['intra_class','intra_file','cross_file']],[]);unique=sorted(set(dep));new=score(ctx,unique) if unique else None
  oldctx,_=legacy.load_context_code_list_from_gml(str(f));old=legacy.compute_task_recall(dep,oldctx)
  row=dict(project=project,project_task_id=i,namespace=t['namespace'],gt=len(unique),seed_nodes=len(g),old_seed_hits=old['dependency_hit'],new_seed_dr=new['dr'] if new else None,new_seed_hits=len(new['hits']) if new else 0,seed_contains_target=any(canon(x.get('method_signature') or x.get('sig',''))==canon(t['namespace']) for x in ctx),graph_path=str(f))
  rows.append(row);local.append(row)
 ref=archived[project.split('/')[-1]];gt=sum(r['gt'] for r in local);oldhits=sum(r['old_seed_hits'] for r in local)
 assert gt==int(ref['sum_gt']) and oldhits==int(ref['ori_match']),(project,oldhits,ref['ori_match'])
 project_rows.append(dict(project=project,tasks=len(local),gt=gt,old_seed_hits=oldhits,old_seed_project_dr=round(oldhits/gt,6)))
 print(project,len(rows),'checked',round(time.time()-start,1),'s',flush=True)
valid=[r for r in rows if r['gt']];assert len(rows)==1430 and len(valid)==1146
oldweighted=round(sum(r['old_seed_project_dr']*r['tasks'] for r in project_rows)/1430,6)
seed=sum(r['new_seed_dr'] for r in valid)/1146;final=json.loads(((_package_path() / 'results/retrieval/combined/featlens_scores/summary.json')).read_text())['new_relaxed_task_macro_dr']
summary=dict(projects=90,total_tasks=1430,dr_tasks=1146,legacy_seed_dr_reproduced=oldweighted,new_seed_macro_dr=seed,new_final15_macro_dr=final,direct_feature_top15=None,direct_feature_top15_reason='All original feature diagnostic records use top1/top3 feature CLUSTERS, not top15 functions. No independent Feature direct Top15 result exists in these records.',feature_output_keys=dict(keys),mean_seed_nodes=sum(r['seed_nodes'] for r in valid)/1146,seed_target_node_tasks=sum(r['seed_contains_target'] for r in valid),retrieval_reruns=0,model_calls=0,final_result_provenance=str((_package_path() / 'results/retrieval/combined/featlens_scores/summary.json')))
for filename,data in [('summary.json',summary),('feature_diagnostic_inventory.json',feature_file_inventory),('read_input_hashes.json',hashes)]: (O/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2))
with open((_package_path() / 'results/retrieval/combined/featlens_scores/seed_scores/per_task_seed.csv'),'w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(sha(p)==h for p,h in hashes.items())
((_package_path() / 'results/retrieval/combined/featlens_scores/seed_scores/verification.json')).write_text(json.dumps(dict(original_seed_graphs=1430,all_90_old_seed_counts_reproduced=True,feature_diagnostic_records=1430,no_direct_feature_top15=all(not p['has_feature_top15'] for p in feature_file_inventory),input_files_unchanged=True,exact_new_score_function_used=True),indent=2))
((_package_path() / 'results/retrieval/combined/featlens_scores/seed_scores/REPORT.md')).write_text(f'''# Original historical FeatLens outputs: new-metric seed and final DR

Same original 90 repositories / 1146 DR-defined tasks. Predictions unchanged.

|Row|Direct Feature Top15 DR|Seed DR|Final graph Top15 DR|
|---|---:|---:|---:|
|Historical outputs, new relaxed task-macro scoring|Not evaluated|{seed:.4%}|{final:.4%}|

Feature diagnostic files all have top1/top3 FEATURE CLUSTERS, not top15 function rankings. Do not relabel the variable-size seed as DR@15 or slice arbitrary member order. Mean seed nodes: {summary['mean_seed_nodes']:.4f}.

Input hashes checked unchanged. This supplements the historical-501 diagnostic, not a new leakage-free benchmark result. Historical returned-target and index information-boundary caveats remain.
''')
print(json.dumps(summary,ensure_ascii=False,indent=2))
