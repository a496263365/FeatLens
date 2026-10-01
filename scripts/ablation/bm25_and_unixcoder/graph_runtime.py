#!/usr/bin/env python3
"""50-task paired retrieval diagnostic. Never writes to source datasets/indexes.
Cached indexes may contain target information: NOT a clean benchmark reproduction.
"""

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

import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,csv,random,time,hashlib,copy,argparse,collections
from pathlib import Path
ROOT=(_package_path())
FULL=(_package_path() / 'results/ablation/deveval/feature_and_summary')
sys.path.insert(0,str((_package_path() / 'src')))
import numpy as np
import pandas as pd
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer
from graph import expand_and_rank_graph_exclude_TM_last as graph_ops
# Imported production module changes CUDA_VISIBLE_DEVICES; restore before CUDA init.
os.environ['CUDA_VISIBLE_DEVICES']='6'
from graph.embedding_backends import BGECodeV1Backend
from search.utils import enre_utils as metrics

PROJECT=os.environ.get('FEATLENS_PROJECT','Database/asyncpg')
BASE=(_package_path() / 'results/ablation/deveval/feature_and_summary/inputs')/PROJECT
PROJECT_INPUT=BASE
OUT=(_package_path() / 'results/ablation/deveval/feature_and_summary/project_results')/PROJECT
OUT.mkdir(parents=True,exist_ok=True)
SEED=20260928

def norm(s): return str(s).split('(',1)[0].lstrip('.')
def canon(s): return norm(s).replace('.__init__.','.')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2))
def readlines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def deps(t): return sorted(set(sum([t['dependency'].get(k,[]) for k in ('intra_class','intra_file','cross_file')],[])))

class CachedBGE:
    def __init__(self):
        self.base=BGECodeV1Backend(model_name=_model_location('FEATLENS_BGE_MODEL', 'BAAI/bge-code-v1'),device=torch.device('cuda:0'))
        self.codes={};self.queries={}
    def prefill(self,texts):
        missing=list(dict.fromkeys(s for s in texts if s not in self.codes))
        for i in range(0,len(missing),8):
            batch=missing[i:i+8]
            vs=self.base.encode_code(batch,batch_size=8).detach().cpu()
            for s,v in zip(batch,vs): self.codes[s]=v
            if i%400==0: print('BGE prefill',i,'/',len(missing),flush=True)
    def encode_code(self,texts,batch_size=16):
        self.prefill(texts)
        return torch.stack([self.codes[s] for s in texts]).to(self.base.device)
    def encode_query(self,q):
        if q not in self.queries: self.queries[q]=self.base.encode_query(q).detach().cpu()
        return self.queries[q].to(self.base.device)

def score_context(context,dep):
    # Keep archived evaluator for comparability, but use unique dependency hits below.
    legacy=metrics.compute_task_recall(dep,context)
    hits=set(); exact=set()
    retrieved={canon(c.get('method_signature') or c.get('sig','')) for c in context}
    for d in dep:
        if canon(d) in retrieved: hits.add(d);exact.add(d);continue
        if d in metrics.variables_enre:
            if any(d.split('.')[-1] in c.get('method_code','') for c in context): hits.add(d)
        elif d in metrics.unresolved_attribute_enre:
            cls='.'.join(d.split('.')[:-1]);name=d.split('.')[-1]
            if any(c.get('sig','').startswith(cls+'.') and 'self.'+name in c.get('method_code','') for c in context):hits.add(d)
        elif d in metrics.module_enre or d in metrics.package_enre:
            if any(c.get('sig','').startswith(d) for c in context):hits.add(d)
    return dict(dr=len(hits)/len(dep),exact_dr=len(exact)/len(dep),legacy_dr=legacy['recall'],hits=sorted(hits),exact_hits=sorted(exact),n=len(context),chars=sum(len(c.get('method_code','')) for c in context))

def main():
    start=time.time()
    tasks=readlines(PROJECT_INPUT/'filtered.jsonl')
    assert tasks and len({t['namespace'] for t in tasks})==len(tasks)
    inventory={f:sha(PROJECT_INPUT/f) for f in ['methods.csv','methods_with_desc.csv','features.csv','report-enre.json','filtered.jsonl']}
    writej(OUT/'manifest.json',dict(project=PROJECT,tasks=len(tasks),input_hashes=inventory,script_sha256=sha(__file__),scope='all nonempty-dependency benchmark tasks in project; reused or separately rebuilt cached indexes; not verified target-isolated'))
    (OUT/'tasks.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks))
    print('SAMPLE',collections.Counter(t['project_key'] for t in tasks),flush=True)
    mini=SentenceTransformer(_model_location('FEATLENS_MINILM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),device='cuda:0')
    bge=CachedBGE(); records=[];audits=[]
    for project in sorted({t['project_key'] for t in tasks}):
        p=PROJECT_INPUT;pts=[t for t in tasks if t['project_key']==project]
        md=pd.read_csv(p/'methods.csv',dtype=str).fillna('');fd=pd.read_csv(p/'features.csv',dtype=str).fillna('');dd=pd.read_csv(p/'methods_with_desc.csv',dtype=str).fillna('')
        method_map={norm(r['method_signature']):r for r in md.to_dict('records')}
        descriptions={norm(r['func_fullName']):r['func_desc'] for r in dd.to_dict('records')}
        valid,q_to_id,id_to_q,adj,rev=graph_ops.load_enre_json(str(p/'report-enre.json'))
        metrics.clear_enre_elements();metrics.load_enre_elements(str(p/'report-enre.json'))
        groups=fd.groupby('id')['desc'].first();fids=groups.index.tolist()
        feature_members={fid:list(dict.fromkeys(norm(s) for s in fd.loc[fd.id==fid,'method_name'].tolist())) for fid in fids}
        target_fid={}
        for r in fd.to_dict('records'):target_fid.setdefault(canon(r['method_name']),r['id'])
        methods=[s for s in method_map if s in q_to_id and valid[q_to_id[s]].get('category')=='Function']
        assert methods
        femb=mini.encode(groups.tolist(),normalize_embeddings=True,show_progress_bar=False)
        demb=mini.encode([descriptions.get(s,'') for s in methods],normalize_embeddings=True,show_progress_bar=False)
        print('PROJECT',project,'methods',len(methods),'tasks',len(pts),flush=True)
        # Corpus embeddings are only used for retrieval of OTHER functions.
        codes=[method_map[s]['method_code'] for s in methods]
        bge.prefill(codes)
        code_emb=torch.stack([bge.codes[c] for c in codes]).float().numpy()
        for task in pts:
            tid=task['example_id'];target=canon(task['namespace']);dep=deps(task)
            query=task['requirement']['Functionality']+' '+task['requirement']['Arguments']
            target_ids={i for i,q in id_to_q.items() if canon(q)==target}
            # Mask target before graph construction and expansion, not just final selection.
            tv={i:v for i,v in valid.items() if i not in target_ids}
            tq={i:q for i,q in id_to_q.items() if i not in target_ids}
            ta={i:[(j,k) for j,k in es if j not in target_ids] for i,es in adj.items() if i not in target_ids}
            tr={i:[(j,k) for j,k in es if j not in target_ids] for i,es in rev.items() if i not in target_ids}
            tm={s:r for s,r in method_map.items() if canon(s)!=target}
            qmini=mini.encode([query],normalize_embeddings=True,show_progress_bar=False)[0]
            order=np.argsort(-(femb@qmini),kind='stable');ranked=[fids[i] for i in order]
            tf=target_fid.get(target)
            forced=(([tf] if tf else [])+[f for f in ranked if f!=tf])[:3]
            semantic=ranked[:3]
            def members(fs):return list(dict.fromkeys(s for f in fs for s in feature_members[f] if s in tm and s in q_to_id and q_to_id[s] in tv))
            promoted=members(forced);unpromoted=members(semantic);budget=len(promoted)
            # A zero mapped feature seed is a genuine empty retrieval; record it, do not drop the task.
            def dense(sims,n): return [methods[i] for i in np.argsort(-sims,kind='stable') if canon(methods[i])!=target][:n]
            ds=demb@qmini; cs=code_emb@bge.encode_query(query).detach().cpu().float().numpy()[0]
            arms={'feature_promote':promoted,'feature_no_promote':unpromoted,'feature_no_target_feature':unpromoted,'summary_dense':dense(ds,budget),'code_dense_bge':dense(cs,budget),'summary_dense_no_samefeature':dense(ds,budget),'code_dense_bge_no_samefeature':dense(cs,budget)}
            target_rows=[r for s,r in method_map.items() if canon(s)==target]
            target_file=target_rows[0]['func_file'] if target_rows else ''
            audits.append(dict(task_id=tid,project=project,target=target,target_method_present=bool(target_rows),target_feature_found=tf is not None,target_enre_nodes=len(target_ids),promotion_changes_clusters=forced!=semantic,seed_budget=budget,raw_target_code_chars=sum(len(r['method_code']) for r in target_rows)))
            def evaluate(arm,stage,ctx,**kw):
                assert all(canon(c.get('method_signature') or c.get('sig',''))!=target for c in ctx)
                result=dict(task_id=tid,project=project,namespace=task['namespace'],arm=arm,stage=stage,dependencies=dep,predictions=[c.get('method_signature') or c.get('sig','') for c in ctx],**score_context(ctx,dep),**kw)
                records.append(result)
                with open(OUT/'task_metrics.jsonl','a') as out:out.write(json.dumps(result,ensure_ascii=False)+'\n')
            for arm,selected in arms.items():
                G=nx.DiGraph();G.graph.update(target_method=task['namespace'],target_file=target_file)
                # Only feature promotion varies in no_promote; no_target_feature ALSO removes same-feature bonus.
                same_feature=set(feature_members.get(tf,[])) if arm not in {'feature_no_target_feature','summary_dense_no_samefeature','code_dense_bge_no_samefeature'} else set()
                for s in selected:
                    eid=q_to_id[s];r=tm[s]
                    G.add_node(eid,sig=s,category=tv[eid]['category'],method_signature=r['method_signature'],method_code=r['method_code'],func_file=r['func_file'],is_SameFile=r['func_file']==target_file,is_SameFeature=s in same_feature,is_SimilarMethod=False)
                for u in list(G):
                    for v,k in ta.get(u,[]):
                        if v in G:G.add_edge(u,v,type=k)
                d=OUT/'graphs'/project/arm;d.mkdir(parents=True,exist_ok=True)
                nx.write_gml(G,d/f'task_{tid}_ori.gml')
                evaluate(arm,'seed',[dict(a) for _,a in G.nodes(data=True)])
                if len(G)==0:
                    empty=d/'pagerank_top_15';empty.mkdir(exist_ok=True)
                    nx.write_gml(G,empty/f'task_{tid}_rank.gml')
                    evaluate(arm,'final15',[],graph_seconds=0.0,empty_seed=True)
                    continue
                graph_ops.TOP_KS=[15]
                t0=time.time()
                graph_ops.process_graph_dir(str(d),[task],bge,tm,tv,tq,ta,tr,str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code')/task['project_path']))
                g=nx.read_gml(d/'pagerank_top_15'/f'task_{tid}_rank.gml')
                evaluate(arm,'final15',[dict(a) for _,a in g.nodes(data=True)],graph_seconds=time.time()-t0)
            # Fixed-size direct retrieval reference, distinct from budget-matched graph seeds.
            for arm,scores in [('summary_dense_top15',ds),('code_dense_bge_top15',cs)]:
                selected=dense(scores,15)
                evaluate(arm,'direct15',[dict(tm[s],sig=s) for s in selected])
            print('DONE TASK',tid,project,'elapsed',round(time.time()-start,1),flush=True)
        # Keep cache only for current repository to bound RAM.
        bge.codes.clear()
    assert len(records)==16*len(tasks)
    for f,h in inventory.items():assert sha(PROJECT_INPUT/f)==h
    writej(OUT/'audit.json',audits)
    writej(OUT/'run_completed.json',dict(tasks=len(tasks),rows=len(records),elapsed_seconds=time.time()-start,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print('COMPLETE',len(records),'elapsed',time.time()-start,flush=True)

if __name__=='__main__':
    import fcntl
    lock=open(OUT/'run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
    if (OUT/'run_completed.json').exists():raise SystemExit(0)
    if (OUT/'task_metrics.jsonl').exists(): raise SystemExit('Refusing to overwrite existing run; use a new output directory or explicitly archive previous attempt.')
    main()
