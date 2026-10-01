
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
O=(_package_path() / 'results/retrieval/combined/featlens_scores');R=(_package_path());B=(_package_path() / 'results/retrieval/deveval/featlens_and_rag')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name):
 s=importlib.util.spec_from_file_location(name,(_package_path() / 'scripts/retrieval/featlens')/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
legacy=load('reference_graph_scorer');legacy.DEBUG=False;helpers=load('dependency_matching')
# Execute only the exact three pure scoring definitions from frozen new runner.
tree=ast.parse(((_package_path() / 'scripts/retrieval/featlens/scoring_source.py')).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['norm','canon','score_context']];assert len(nodes)==3
ns={'metrics':helpers};exec(compile(ast.Module(body=nodes,type_ignores=[]),str((_package_path() / 'scripts/retrieval/featlens/scoring_source.py')),'exec'),ns);score=ns['score_context'];canon=ns['canon']
archived=list(csv.DictReader(open((_package_path() / 'results/retrieval/combined/featlens_scores/graph_scope_reference.csv'))));original={r['project_name']:r for r in archived if r['project_name']!='ALL'};overall=next(r for r in archived if r['project_name']=='ALL')
fm=list(csv.DictReader(open((_package_path() / 'results/retrieval/combined/featlens_scores/feature_scope_reference.csv'))));name2path={r['project_name']:r['project_dir'] for r in fm if r['project_name'] in original};assert len(name2path)==90
bench=[json.loads(s) for s in ((_package_path() / 'data/deveval/tasks_all_1430.jsonl')).read_text().splitlines() if s.strip()];benchidx={(t['project_path'],t['namespace']):t for t in bench}
hashes={};rows=[];projects=[];disagreements=[];inventory=[];start=time.time()
for name,project in sorted(name2path.items(),key=lambda kv:kv[1]):
 p=B/project;fp=_package_path('data/deveval/projects')/project/'filtered.jsonl';ep=p/'report-enre.json';folder=p/'graphs/pagerank_top_15'
 assert fp.exists() and ep.exists() and folder.is_dir(),project
 for f in [fp,ep]:hashes[str(f)]=sha(f)
 ts=[json.loads(s) for s in fp.read_text().splitlines() if s.strip()];assert len(ts)==int(original[name]['total_tasks'])
 for t in ts:assert t['requirement']==benchidx[(project,t['namespace'])]['requirement'] and t['dependency']==benchidx[(project,t['namespace'])]['dependency']
 legacy.clear_enre_elements();legacy.load_enre_elements(str(ep));helpers.clear_enre_elements();helpers.load_enre_elements(str(ep))
 local=[];files=list(folder.glob('task_*_rank.gml'));assert len(files)==len(ts),(project,len(files),len(ts))
 for i,t in enumerate(ts,1):
  f=folder/f'task_{i}_rank.gml';assert f.exists();hashes[str(f)]=sha(f);g=nx.read_gml(f)
  dep=sum([t['dependency'].get(k,[]) for k in ['intra_class','intra_file','cross_file']],[]);unique=sorted(set(dep))
  old_ctx,_=legacy.load_context_code_list_from_gml(str(f));old=legacy.compute_task_recall(dep,old_ctx)
  # Exact new evaluation input: every stored graph node, no class filtering.
  all_ctx=[dict(d) for _,d in g.nodes(data=True)];new=score(all_ctx,unique) if unique else None
  target_nodes=[n for n,d in g.nodes(data=True) if canon(d.get('method_signature') or d.get('sig',''))==canon(t['namespace'])]
  row=dict(project=project,project_task_id=i,namespace=t['namespace'],dep_count=len(unique),old_hits=old['dependency_hit'],old_task_dr=old['recall'],new_hits=len(new['hits']) if new else 0,new_dr=new['dr'] if new else None,new_exact_dr=new['exact_dr'] if new else None,graph_nodes=len(g),class_nodes=sum(d.get('category')=='Class' for _,d in g.nodes(data=True)),old_scored_functions=len(old_ctx),target_node_count=len(target_nodes),new_hit_list=json.dumps(new['hits'] if new else [],ensure_ascii=False),source_graph=str(f))
  rows.append(row);local.append(row)
 hits=sum(r['old_hits'] for r in local);gt=sum(r['dep_count'] for r in local);ref=original[name];valid=[r for r in local if r['dep_count']]
 pr=dict(project=project,total_tasks=len(local),dr_defined_tasks=len(valid),gt=gt,old_hits=hits,archived_gt=int(ref['sum_gt']),archived_old_hits=int(ref['rank_15_match']),recomputed_legacy_project_dr=round(hits/gt,6),archived_project_dr=float(ref['rank_15_R']),new_macro_dr=sum(r['new_dr'] for r in valid)/len(valid),new_project_micro=sum(r['new_hits'] for r in valid)/gt)
 pr['historical_counts_match']=pr['gt']==pr['archived_gt'] and pr['old_hits']==pr['archived_old_hits'];projects.append(pr)
 if not pr['historical_counts_match']:disagreements.append(pr)
 logs={n:dict(exists=(p/n).is_file(),bytes=(p/n).stat().st_size if (p/n).exists() else None) for n in ['compare_graph_recall.debug.log','compare_graph_recall_report.csv','diagnostic_***feature.jsonl']}
 inventory.append(dict(project=project,graph_directory=str(folder),rank_graph_count=len(files),files=logs))
 print(project,len(rows),'checked',round(time.time()-start,1),'s',flush=True)
def csvout(name,rs):
 with open(O/name,'w') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def dump(name,x):(O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
valid=[r for r in rows if r['dep_count']];assert len(rows)==1430 and len(valid)==1146
weighted=round(sum(p['recomputed_legacy_project_dr']*p['total_tasks'] for p in projects)/1430,6)
summary=dict(projects=90,all_tasks=1430,dr_defined_tasks=1146,undefined_tasks=284,archived_legacy_dr=float(overall['rank_15_R']),recomputed_legacy_dr=weighted,historical_project_counts_matching=sum(p['historical_counts_match'] for p in projects),new_relaxed_task_macro_dr=sum(r['new_dr'] for r in valid)/1146,new_exact_task_macro_dr=sum(r['new_exact_dr'] for r in valid)/1146,new_relaxed_global_micro_dr=sum(r['new_hits'] for r in valid)/sum(r['dep_count'] for r in valid),new_relaxed_projectweighted_dr=sum(p['new_project_micro']*p['total_tasks'] for p in projects)/1430,old_scoring_task_macro_dr=sum(r['old_task_dr'] for r in valid)/1146,target_node_occurrences=sum(r['target_node_count'] for r in rows),graphs_read=len(rows),mean_final_graph_nodes=sum(r['graph_nodes'] for r in valid)/1146,method='Unchanged historical rank15 graph outputs; score_context extracted unchanged from new ablation runner; no retrieval, reranking, class-code restoration or manual edits',elapsed_seconds=time.time()-start)
csvout('per_task.csv',rows);csvout('per_project.csv',projects);dump('summary.json',summary);dump('historical_reproduction_disagreements.json',disagreements);dump('artifact_inventory.json',inventory);dump('input_hashes.json',hashes)
assert not disagreements,'Historical graph outputs no longer fully reproduce archived counts; inspect before calling them original501'
assert weighted==float(overall['rank_15_R']), (weighted,overall['rank_15_R'])
assert all(sha(p)==h for p,h in hashes.items()),'Original input changed during run'
dump('verification.json',dict(all_90_historical_project_hit_counts_reproduced=True,legacy_501178_reproduced=True,new_task_macro_recomputed=True,source_graphs_and_annotations_unchanged=True,model_calls=0,retrieval_reruns=0,graphs=1430,dr_tasks=1146,logs_present={n:sum(x['files'][n]['exists'] for x in inventory) for n in inventory[0]['files']}))
lines=['# 原论文50.1%对应检索结果的新口径重算','',f"历史90项目、1430份rank15图全部读取；其中1146任务参考依赖非空。逐项目旧命中数全部复现历史汇总；旧整体DR重算为{weighted:.6f}。",'', '|同一批历史检索结果|DR|','|---|---:|',f"|原旧评分+原项目任务权重|{weighted:.4%}|",f"|新去重宽松评分+逐任务平均（主答案）|{summary['new_relaxed_task_macro_dr']:.4%}|",f"|新严格符号评分+逐任务平均（附加核验）|{summary['new_exact_task_macro_dr']:.4%}|",'', '新评分上下文与新消融脚本一致：全部已返回节点参与、命中集合去重、保留末尾构造函数名、同样变量/属性/模块宽松规则。类代码仍按原GML保留，不补骨架、不新增节点、不截断或重新排序。', '', '新表48.64%使用另一批重新运行的检索结果及已记录修订，不能与本次历史预测重算混为一谈。新旧主分数之间同时存在匹配和汇总方式变化，不能只归因于某一项。', '', '所有输入哈希和日志覆盖记录保留；本次没有修改旧输出，没有API调用。']
((_package_path() / 'results/retrieval/combined/featlens_scores/REPORT.md')).write_text('\n'.join(lines));print(json.dumps(summary,indent=2))
