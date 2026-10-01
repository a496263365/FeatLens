"""Evaluate per-project shards; each owns an isolated source copy and parser temp dir."""

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

import argparse, concurrent.futures, collections, json, os, shutil, signal, subprocess, threading, time
from common import *
from postprocess import normalize
EVALPY=_PortablePath(_evaluation_python())
PASS=ROOT/'scripts/metrics/deveval/pass_k.py'
RECALL=ROOT/'scripts/metrics/deveval/parser/recall_k.py'
DEPS=_external_path('FEATLENS_DEPENDENCY_DATA_ROOT', 'external/deveval/dependency_data')
CACHED=ROOT/'results/generation/deveval/locagent/test_source_pass'

def evaluation_env():
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['KEY','TOKEN','SECRET','PASSWORD'])}
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env.update(PATH=str(EVALPY.parent)+':'+env.get('PATH',''),CONDA_PREFIX=str(EVALPY.parent.parent),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    return env

def launch(cmd,log,cwd,limit):
    with open(log,'w') as f:
        p=subprocess.Popen([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,cwd=cwd,env=evaluation_env(),start_new_session=True)
        try:return p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            return 124

def evaluate_project(project,tasks):
    dest=OUT/'evaluation'/slug(project);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return json.loads((dest/'DONE.json').read_text())
    start=time.time()
    jsonl(dest/'tasks.jsonl',tasks)
    comps=[json.loads((OUT/'tasks'/f"{t['idx']:04d}"/'generation.json').read_text()) for t in tasks]
    for r in comps:
        raw=r['completion']; normalized,notes=normalize(raw,r['namespace'])
        r['raw_completion']=raw; r['completion']=normalized; r['postprocess_notes']=notes
    jsonl(dest/'completion.jsonl',comps)
    workspace=dest/'source';target=workspace/project
    if not (dest/'source_prepared.json').exists():
        if target.exists():raise RuntimeError(f'Unfinished source snapshot {target}; inspect before reuse')
        shutil.copytree(REFERENCE/project,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git','.pytest_cache'),ignore_dangling_symlinks=True)
        # Copy cached build eggs to avoid re-downloading or changing the shared evaluator environment.
        if (CACHED/project/'.eggs').is_dir() and not (target/'.eggs').exists():shutil.copytree(CACHED/project/'.eggs',target/'.eggs',ignore_dangling_symlinks=True)
        hashes={t['completion_path']:sha_file(REFERENCE/t['completion_path']) for t in tasks}
        assert all(sha_file(workspace/p)==h for p,h in hashes.items())
        dump(dest/'source_prepared.json',hashes)
    commands={
      'pass':[EVALPY,PASS,'--output_file',dest/'completion.jsonl','--log_file',dest/'test_output.jsonl','--source_code_root',workspace,'--data_file',dest/'tasks.jsonl','--k','1','--n','1','--failure_log',dest/'failure.log'],
      'dir':[EVALPY,RECALL,'--output_file',dest/'completion.jsonl','--log_file',dest/'recall_output.jsonl','--source_code_root',workspace,'--data_file',dest/'tasks.jsonl','--dependency_data_root',DEPS,'--dependency_tmp_dir',dest/'dependency_tmp','--k','1']}
    dump(dest/'commands.json',{k:list(map(str,v)) for k,v in commands.items()})
    statuses={}
    for stage,cmd in commands.items():
        if (dest/f'{stage}_done.json').exists(): statuses[stage]=json.loads((dest/f'{stage}_done.json').read_text());continue
        returncode=launch(cmd,dest/f'{stage}.log',dest,max(300,len(tasks)*90+120))
        # Both evaluators must have restored originals before the next stage.
        expected=json.loads((dest/'source_prepared.json').read_text())
        mismatches=[p for p,h in expected.items() if not (workspace/p).exists() or sha_file(workspace/p)!=h]
        if mismatches:
            dump(dest/f'{stage}_source_restore_issues.json',mismatches)
            for p in mismatches:shutil.copy2(REFERENCE/p,workspace/p)
        statuses[stage]={'returncode':returncode,'source_restore_mismatches':mismatches}
        dump(dest/f'{stage}_done.json',statuses[stage])
    tests=rows(dest/'test_output.jsonl'); recalls=rows(dest/'recall_output.jsonl')
    exp_ns={t['namespace'] for t in tasks};dep_ns={t['namespace'] for t in tasks if any(t['dependency'].values())}
    valid=(set(r['namespace'] for r in tests)==exp_ns and set(r['namespace'] for r in recalls)==dep_ns and all(s['returncode']==0 for s in statuses.values()))
    r={'project_path':project,'tasks':len(tasks),'pass_evaluated':len(tests),'dir_evaluated':len(recalls),'wall_seconds':time.time()-start,'stages':statuses,'complete':valid}
    dump(dest/'DONE.json' if valid else dest/'INCOMPLETE.json',r)
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    grouped=collections.defaultdict(list)
    for t in rows(OUT/'inputs/eval_tasks.jsonl'):grouped[t['project_path']].append(t)
    submitted=set();completed={};errors={};pending={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(completed)+len(errors)<len(grouped):
            for project,ts in grouped.items():
                if project in submitted:continue
                if all((OUT/'tasks'/f"{t['idx']:04d}"/'generation.json').exists() for t in ts):
                    pending[pool.submit(evaluate_project,project,ts)]=project;submitted.add(project)
            for f in list(pending):
                if not f.done():continue
                project=pending.pop(f)
                try:
                    r=f.result();completed[project]=r;print(f'evaluation {len(completed)}/{len(grouped)} {project} complete={r["complete"]}',flush=True)
                except Exception as e:
                    import traceback
                    errors[project]=traceback.format_exc();print(errors[project],flush=True)
            dump(OUT/'evaluation_progress.json',{'submitted':len(submitted),'completed':len(completed),'errors':errors,'projects':completed})
            if len(completed)+len(errors)==len(grouped):break
            time.sleep(15)
    ordered=rows(OUT/'inputs/eval_tasks.jsonl');tests={};recalls={}
    for project in grouped:
        d=OUT/'evaluation'/slug(project)
        tests.update({r['namespace']:r for r in rows(d/'test_output.jsonl')});recalls.update({r['namespace']:r for r in rows(d/'recall_output.jsonl')})
    allc={r['namespace']:r for project in grouped for r in rows(OUT/'evaluation'/slug(project)/'completion.jsonl')}
    jsonl(OUT/'completions/gpt-5-mini_evaluated.jsonl',[allc[t['namespace']] for t in ordered if t['namespace'] in allc])
    jsonl(OUT/'gpt-5-mini_test_output.jsonl',[tests[t['namespace']] for t in ordered if t['namespace'] in tests])
    jsonl(OUT/'gpt-5-mini_recall_output.jsonl',[recalls[t['namespace']] for t in ordered if t['namespace'] in recalls])
    dump(OUT/'EVALUATION_COMPLETE.json',{'projects':len(completed),'errors':errors,'pass_tasks':len(tests),'dir_tasks':len(recalls),'complete':not errors and len(tests)==1430 and len(recalls)==1146})
if __name__=='__main__':main()
