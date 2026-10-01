"""Three BM25 seeds through the unchanged FeatLens graph pipeline."""

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

import os,json,time,contextlib,traceback,re,fcntl
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
O=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder');R=(_package_path());B=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph');OLD=(_package_path() / 'results/ablation/deveval/feature_and_summary');UX=R/'results/deveval90_unixcoder_bm25_20260928'
B.mkdir(exist_ok=True)
lock=open((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/controller.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
# Avoid loading a second large model while the previous controlled run uses it.
while not ((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/run_finished.json')).exists():
 live=False
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if str((_package_path() / 'scripts/ablation/bm25_and_unixcoder/run_corrected.py')).encode() in (proc/'cmdline').read_bytes().split(b'\0'):live=True;break
  except OSError:pass
 if not live:raise SystemExit('UniXcoder graph worker absent before completion; inspect first')
 print('WAIT_UNIXCODER_GRAPH',time.strftime('%H:%M:%S'),flush=True);time.sleep(30)
time.sleep(3)
import original_graph_runtime as rt
import pandas as pd
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
embed=rt.CachedBGE();start=time.time();status=[]
projects=json.loads(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/primary_scope.json')).read_text())['projects']
def tok(s):return re.findall(r'\w+',str(s).lower())
def sigtok(s):return tok(re.sub(r'[\(\),.:]',' ',s.split('.')[-1].replace('_',' ')))
for project in projects:
 dest=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/project_results')/project;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'run_completed.json').exists():status.append(dict(project=project,status='already_completed'));continue
 if (dest/'task_metrics.jsonl').exists():status.append(dict(project=project,status='partial_requires_inspection'));continue
 logpath=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/logs')/(project.replace('/','__')+'.log');logpath.parent.mkdir(parents=True,exist_ok=True)
 print('START',project,flush=True);t0=time.time()
 with open(logpath,'w') as log:
  with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
   try:
    p=(_package_path() / 'results/ablation/deveval/feature_and_summary/inputs')/project;tasks=rt.readlines(p/'filtered.jsonl')
    manifest=json.loads(((_package_path() / 'results/ablation/deveval/feature_and_summary/project_results')/project/'manifest.json').read_text())
    for f,h in manifest['input_hashes'].items():assert rt.sha(p/f)==h
    md=pd.read_csv(p/'methods.csv',dtype=str).fillna('');dd=pd.read_csv(p/'methods_with_desc.csv',dtype=str).fillna('');fd=pd.read_csv(p/'features.csv',dtype=str).fillna('')
    mm={rt.norm(r['method_signature']):r for r in md.to_dict('records')}
    desc={rt.norm(r['func_fullName']):r['func_desc'] for r in dd.to_dict('records')}
    valid,q_to_id,id_to_q,adj,rev=rt.graph_ops.load_enre_json(str(p/'report-enre.json'))
    rt.metrics.clear_enre_elements();rt.metrics.load_enre_elements(str(p/'report-enre.json'))
    methods=[s for s in mm if s in q_to_id and valid[q_to_id[s]].get('category')=='Function']
    docs={'bm25_description_top15':[tok(desc.get(s,'')) for s in methods],'bm25_code_top15':[tok(mm[s]['method_code']) for s in methods],'bm25_signature_top15':[sigtok(mm[s]['method_signature']) for s in methods]}
    models={a:BM25Okapi(ds) if any(ds) else None for a,ds in docs.items()}
    tfmap={};members={}
    for r in fd.to_dict('records'):
     tfmap.setdefault(rt.canon(r['method_name']),r['id']);members.setdefault(r['id'],set()).add(rt.norm(r['method_name']))
    old_idx={(r['task_id'],r['arm'],r['stage']):r for r in rt.readlines((_package_path() / 'results/ablation/deveval/feature_and_summary/project_results')/project/'task_metrics.jsonl')}
    ux_idx={(r['task_id'],r['arm'],r['stage']):r for r in rt.readlines(UX/'project_results'/project/'task_metrics.jsonl')}
    embed.prefill([mm[s]['method_code'] for s in methods]);rows=[]
    for task in tasks:
     tid=task['example_id'];target=rt.canon(task['namespace']);dep=rt.deps(task);query=task['requirement']['Functionality']+' '+task['requirement']['Arguments']
     target_ids={i for i,q in id_to_q.items() if rt.canon(q)==target}
     tv={i:v for i,v in valid.items() if i not in target_ids};tq={i:q for i,q in id_to_q.items() if i not in target_ids}
     ta={i:[(j,k) for j,k in es if j not in target_ids] for i,es in adj.items() if i not in target_ids}
     tr={i:[(j,k) for j,k in es if j not in target_ids] for i,es in rev.items() if i not in target_ids}
     tm={s:r for s,r in mm.items() if rt.canon(s)!=target}
     target_rows=[r for s,r in mm.items() if rt.canon(s)==target];target_file=target_rows[0]['func_file'] if target_rows else ''
     same_feature=members.get(tfmap.get(target),set());budget=old_idx[(tid,'feature_promote','seed')]['n']
     def record(arm,stage,ctx):
      assert all(rt.canon(c.get('method_signature') or c.get('sig',''))!=target for c in ctx)
      r=dict(task_id=tid,project=task['project_key'],project_path=project,namespace=task['namespace'],arm=arm,stage=stage,dependencies=dep,predictions=[c.get('method_signature') or c.get('sig','') for c in ctx],**rt.score_context(ctx,dep))
      rows.append(r)
      with open(dest/'task_metrics.jsonl','a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
      return r
     for arm,model in models.items():
      scores=model.get_scores(tok(query)) if model is not None else np.zeros(len(methods))
      order=[methods[i] for i in np.argsort(-scores,kind='stable') if rt.canon(methods[i])!=target]
      direct=record(arm,'direct15',[dict(tm[s],sig=s) for s in order[:15]])
      prior=ux_idx[(tid,arm,'direct15')]
      assert direct['predictions']==prior['predictions'] and direct['dr']==prior['dr']
      selected=order[:budget];g=nx.DiGraph();g.graph.update(target_method=task['namespace'],target_file=target_file)
      for s in selected:
       eid=q_to_id[s];r=tm[s]
       g.add_node(eid,sig=s,category=tv[eid]['category'],method_signature=r['method_signature'],method_code=r['method_code'],func_file=r['func_file'],is_SameFile=r['func_file']==target_file,is_SameFeature=s in same_feature,is_SimilarMethod=False)
      for u in list(g):
       for v,k in ta.get(u,[]):
        if v in g:g.add_edge(u,v,type=k)
      seed=record(arm,'seed',[dict(d) for _,d in g.nodes(data=True)]);assert seed['n']==budget
      folder=dest/'graphs'/arm;folder.mkdir(parents=True,exist_ok=True);nx.write_gml(g,folder/f'task_{tid}_ori.gml')
      if len(g):
       rt.graph_ops.TOP_KS=[15];rt.graph_ops.ENABLE_EXTRA_EXPANDED_NODE_BONUS=True
       rt.graph_ops.process_graph_dir(str(folder),[task],embed,tm,tv,tq,ta,tr,str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code')/project))
       final=nx.read_gml(folder/'pagerank_top_15'/f'task_{tid}_rank.gml')
      else:final=g
      record(arm,'final15',[dict(d) for _,d in final.nodes(data=True)])
    assert len(rows)==9*len(tasks)
    for f,h in manifest['input_hashes'].items():assert rt.sha(p/f)==h
    rt.writej(dest/'run_completed.json',dict(project=project,tasks=len(tasks),rows=len(rows),seconds=time.time()-t0,graph_encoder='unchanged original BGE-code-v1',same_feature=True,same_file=True,similar_method=False,seed_budget='matched to feature_promote per task',direct_results_verified_against_prior=True));ok=True
   except Exception:traceback.print_exc();ok=False
 entry=dict(project=project,status='completed' if ok else 'failed',seconds=time.time()-t0);status.append(entry);rt.writej((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/controller_status.json'),status);print('END',entry,flush=True);embed.codes.clear()
rt.writej((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/bm25_graph/run_finished.json'),dict(seconds=time.time()-start,status=status))
