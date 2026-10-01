
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

import os,json,time,hashlib,contextlib,traceback,fcntl
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
import original_graph_runtime as rt
import pandas as pd
import networkx as nx
O=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder');R=(_package_path());OLD=(_package_path() / 'results/ablation/deveval/feature_and_summary');UX=R/'results/deveval90_unixcoder_bm25_20260928'
lock=open((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/controller.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
scope=json.loads(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/primary_scope.json')).read_text());projects=scope['projects'];embed=rt.CachedBGE();status=[];start=time.time()
for project in projects:
 dest=(_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/project_results')/project;dest.mkdir(parents=True,exist_ok=True)
 if (dest/'run_completed.json').exists():status.append(dict(project=project,status='already_completed'));continue
 if (dest/'task_metrics.jsonl').exists():status.append(dict(project=project,status='partial_requires_inspection'));continue
 print('START',project,flush=True);t0=time.time()
 with open((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/logs')/(project.replace('/','__')+'.log'),'w') as log:
  with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
   try:
    p=(_package_path() / 'results/ablation/deveval/feature_and_summary/inputs')/project;tasks=rt.readlines(p/'filtered.jsonl');md=pd.read_csv(p/'methods.csv',dtype=str).fillna('')
    original_manifest=json.loads(((_package_path() / 'results/ablation/deveval/feature_and_summary/project_results')/project/'manifest.json').read_text())
    ux_manifest=json.loads((UX/'project_results'/project/'manifest.json').read_text())
    for f,h in original_manifest['input_hashes'].items():assert rt.sha(p/f)==h==ux_manifest['input_hashes'][f]
    assert ux_manifest['model']=='microsoft/unixcoder-base'
    method_map={rt.norm(r['method_signature']):r for r in md.to_dict('records')}
    valid,q_to_id,id_to_q,adj,rev=rt.graph_ops.load_enre_json(str(p/'report-enre.json'))
    rt.metrics.clear_enre_elements();rt.metrics.load_enre_elements(str(p/'report-enre.json'))
    methods=[s for s in method_map if s in q_to_id and valid[q_to_id[s]].get('category')=='Function']
    embed.prefill([method_map[s]['method_code'] for s in methods])
    ux_rows=rt.readlines(UX/'project_results'/project/'task_metrics.jsonl');ux_idx={(r['task_id'],r['arm'],r['stage']):r for r in ux_rows}
    rows=[]
    for task in tasks:
     tid=task['example_id'];target=rt.canon(task['namespace']);dep=rt.deps(task)
     target_ids={i for i,q in id_to_q.items() if rt.canon(q)==target}
     tv={i:v for i,v in valid.items() if i not in target_ids};tq={i:q for i,q in id_to_q.items() if i not in target_ids}
     ta={i:[(j,k) for j,k in es if j not in target_ids] for i,es in adj.items() if i not in target_ids}
     tr={i:[(j,k) for j,k in es if j not in target_ids] for i,es in rev.items() if i not in target_ids}
     tm={s:r for s,r in method_map.items() if rt.canon(s)!=target}
     for arm in ['code_dense_unixcoder','code_dense_unixcoder_no_samefeature']:
      source=UX/'project_results'/project/'graphs'/task['project_key']/arm/f'task_{tid}_ori.gml'
      g=nx.read_gml(source)
      assert all(rt.canon(d.get('sig',''))!=target for _,d in g.nodes(data=True))
      seed=ux_idx[(tid,arm,'seed')]
      assert set(seed['predictions'])=={d.get('method_signature') or d.get('sig','') for _,d in g.nodes(data=True)}
      if arm.endswith('no_samefeature'):assert all(not d.get('is_SameFeature',False) for _,d in g.nodes(data=True))
      folder=dest/'graphs'/arm;folder.mkdir(parents=True,exist_ok=True)
      (folder/source.name).write_bytes(source.read_bytes())
      if len(g):
       rt.graph_ops.TOP_KS=[15];rt.graph_ops.ENABLE_EXTRA_EXPANDED_NODE_BONUS=True
       rt.graph_ops.process_graph_dir(str(folder),[task],embed,tm,tv,tq,ta,tr,str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code')/project))
       final=nx.read_gml(folder/'pagerank_top_15'/f'task_{tid}_rank.gml')
      else:final=g
      ctx=[dict(d) for _,d in final.nodes(data=True)]
      assert all(rt.canon(d.get('method_signature') or d.get('sig',''))!=target for d in ctx)
      r=dict(task_id=tid,project=task['project_key'],project_path=project,namespace=task['namespace'],arm=arm,stage='final15',dependencies=dep,predictions=[c.get('method_signature') or c.get('sig','') for c in ctx],**rt.score_context(ctx,dep))
      rows.append(r)
      with open(dest/'task_metrics.jsonl','a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    assert len(rows)==len(tasks)*2
    rt.writej(dest/'run_completed.json',dict(project=project,tasks=len(tasks),rows=len(rows),seconds=time.time()-t0,retrieval_encoder='UniXcoder',graph_encoder='original BGE-code-v1',source_inputs_unchanged=True,source_seeds_unchanged=True))
    ok=True
   except Exception:traceback.print_exc();ok=False
 entry=dict(project=project,status='completed' if ok else 'failed',seconds=time.time()-t0);status.append(entry);rt.writej((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/controller_status.json'),status)
 print('END',entry,flush=True);embed.codes.clear()
rt.writej((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/run_finished.json'),dict(seconds=time.time()-start,status=status))
