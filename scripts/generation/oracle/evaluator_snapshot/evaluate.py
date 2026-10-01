"""Isolated per-project DevEval evaluation with full subprocess output logs.

Metric semantics match DevEval: run all annotated tests; macro dependency recall
from the benchmark's analyzer. Drains logs to disk (no subprocess pipe deadlock),
enforces the original 30-second task test budget, and restores source in finally.
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

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import tokenize
import traceback
import fcntl
from postprocess import normalize_first_line

ROOT=(_package_path())
MODELS=['deepseek_v3_2','gpt_5_mini']


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def safe_env():
    # Generated code must not inherit gateway credentials or incidental tokens.
    keep={'PATH','LANG','LC_ALL','TZ','USER','LOGNAME','LD_LIBRARY_PATH','CONDA_PREFIX'}
    env={k:v for k,v in os.environ.items() if k in keep}
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    env['OMP_NUM_THREADS']='1';env['OPENBLAS_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
    return env


def subprocess_logged(cmd,cwd,log,timeout):
    started=time.time()
    env=safe_env()
    # Test programs get an isolated home/tmp, not the researcher's credentials.
    test_home=Path(cwd)/'.oracle_test_home';test_home.mkdir(exist_ok=True)
    env['HOME']=str(test_home);env['TMPDIR']=str(test_home)
    with log.open('wb') as f:
        try:
            proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError as e:
            return {'returncode':None,'seconds':time.time()-started,'error':str(e),'timeout':False}
        try:
            import psutil
            while proc.poll() is None:
                if time.time()-started>max(0.1,timeout):raise subprocess.TimeoutExpired(cmd,timeout)
                try:
                    if psutil.Process(proc.pid).memory_info().rss>5*1024**3:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        return {'returncode':proc.returncode,'seconds':time.time()-started,'timeout':False,'oom':True}
                except psutil.NoSuchProcess:pass
                time.sleep(0.05)
            code=proc.returncode
            return {'returncode':code,'seconds':time.time()-started,'timeout':False}
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
            return {'returncode':proc.returncode,'seconds':time.time()-started,'timeout':True}


def tests(task,project,out):
    out.mkdir(parents=True,exist_ok=True);started=time.time();attempts=[]
    if not task.get('tests'):return {'Result':'NoTests','seconds':0,'attempts':[]}
    for i,test in enumerate(task['tests']):
        remaining=30-(time.time()-started)
        if remaining<=0:return {'Result':'TimeOut','seconds':time.time()-started,'attempts':attempts}
        cmd=[sys.executable,'setup.py','pytest','--addopts',test]
        log=out/f'test_{i:03d}.log'
        result=subprocess_logged(cmd,project,log,remaining)
        attempts.append({'test':test,'command':cmd,'log':str(log),**result})
        if result['timeout']:flag='TimeOut'
        elif result['returncode']!=0:flag='Error'
        else:continue
        return {'Result':flag,'seconds':time.time()-started,'attempts':attempts}
    return {'Result':'Pass','seconds':time.time()-started,'attempts':attempts}


def dir_one(args):
    record=read(args.record); task=record['task']; project=args.source/task['project_path']
    parser_dir=(_package_path() / 'scripts/metrics/deveval/parser')
    sys.path.insert(0,str(parser_dir))
    try:
        from add_func_call import process
        target=args.source/task['completion_path']
        analyzer=Path(str(_external_path('FEATLENS_DEPENDENCY_DATA_ROOT', 'external/deveval/dependency_data', '')))/task['project_path']/'analyzer_result.pkl'
        process(target_object=str(project),func_object_root=str(project),func_path=str(target),
                analyzer_result=str(analyzer),target_root=str(args.out/'analyzer'))
        dependency_file=args.out/'analyzer'/Path(task['completion_path']).relative_to(task['project_path']).with_suffix('.json')
        candidates=list((args.out/'analyzer').rglob(Path(task['completion_path']).with_suffix('.json').name))
        if dependency_file.exists():candidates=[dependency_file]+[p for p in candidates if p!=dependency_file]
        attributes=None
        for path in candidates:
            data=read(path)
            if task['namespace'] in data:attributes=data[task['namespace']];break
        if attributes is None:raise RuntimeError('Analyzer output missing target namespace')
        generated={'intra_class':[x['name'] for x in attributes['in_class']],
                   'intra_file':[x['name'] for x in attributes['in_file']],
                   'cross_file':[x['name'] for x in attributes['in_object']]}
        save(args.out/'parsed.json',{'status':'parsed','generated_dependency':generated})
    except Exception as e:
        traceback.print_exc()
        save(args.out/'parsed.json',{'status':'analysis_error','error':str(e),'generated_dependency':None})


def finalize_completions_if_ready(run):
    if len(list((run/'evaluation').glob('*/DONE.json')))!=90:return
    with (run/'normalized_assembly.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (run/'normalized_assembly.json').exists():return
        audit={}
        for model in MODELS:
            path=run/f'{model}_completion.jsonl'
            original=rows(path)
            norm={r['idx']:r for r in [read(p) for p in (run/'normalized'/model).glob('task_*.json')]}
            assert len(original)==len(norm)==1430
            backup=run/f'{model}_completion_before_format.jsonl'
            if not backup.exists():shutil.copy2(path,backup)
            for r in original:
                n=norm[r['idx']]
                assert n['original_completion_sha256']==hashlib.sha256(r['completion'].encode()).hexdigest()
                r['completion']=n['completion'];r['format_action']=n['format_action']
            tmp=path.with_name(path.name+'.tmp')
            tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in original));tmp.replace(path)
            audit[model]={'before_format_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),
                          'evaluated_completion_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'format_changes':sum(r['format_action']=='first_line_indent_only_unique_syntax_repair' for r in original)}
        save(run/'normalized_assembly.json',audit)


def project_worker(args):
    run=args.run; key=args.project.replace('/','__'); out=run/'evaluation'/key
    out.mkdir(parents=True,exist_ok=True)
    workspace_lock=(out/'workspace.lock').open('w')
    fcntl.flock(workspace_lock,fcntl.LOCK_EX)
    records=[r for r in rows(run/'prepared/prepared.jsonl') if r['task']['project_path']==args.project]
    models={} if args.ground_truth_only else {m:{r['idx']:r for r in rows(run/f'{m}_completion.jsonl')} for m in MODELS}
    manifest=read(run/'manifest.json'); source=Path(manifest['source_root'])/args.project
    # Every project worker has an independent source tree, and no two worker
    # processes modify the same project. Ground truth and both models run serially.
    dst=out/'workspace'/args.project
    if not dst.exists():
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,dst,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','.pytest_cache'))
    save(out/'worker_identity.json',{'pid':os.getpid(),'python':sys.executable,'source':str(source),'project':args.project})
    source_hashes={}
    for row in records:
        task=row['task'];i=row['idx'];rel=Path(task['completion_path']).relative_to(args.project)
        original=(source/rel).read_bytes(); target=dst/rel
        # Resume after an interrupted task from the immutable source, not a leftover completion.
        target.write_bytes(original)
        source_hashes[str(rel)]=hashlib.sha256(original).hexdigest()
        gtpath=out/'ground_truth'/f'task_{i:04d}.json'
        if not gtpath.exists():
            gt=tests(task,dst,out/'ground_truth'/f'task_{i:04d}_logs')
            save(gtpath,{'idx':i,'namespace':task['namespace'],**gt})
        if args.ground_truth_only:continue
        with tokenize.open(source/rel) as f:original_text=f.read();encoding=f.encoding
        for model in MODELS:
            raw_completion=models[model][i]['completion']
            completion,format_action=normalize_first_line(raw_completion)
            save(run/'normalized'/model/f'task_{i:04d}.json',{'idx':i,'namespace':task['namespace'],'completion':completion,
                 'format_action':format_action,'original_completion_sha256':hashlib.sha256(raw_completion.encode()).hexdigest()})
            pass_path=out/'pass'/model/f'task_{i:04d}.json'
            dir_path=out/'dir'/model/f'task_{i:04d}.json'
            needs_dir=row['dependency_count']>0
            if pass_path.exists() and (not needs_dir or dir_path.exists()):continue
            body=textwrap.indent(textwrap.dedent(completion),' '*task['indent'])
            lines=original_text.splitlines(keepends=True);a,b=task['body_position']
            patched=''.join(lines[:a-1])+ '\n'+body+'\n'+''.join(lines[b:])
            syntax_error=None
            try:
                if not completion.strip():raise ValueError('Empty completion (generation failure)')
                compile(patched,str(target),'exec')
            except (SyntaxError,ValueError) as e:syntax_error=str(e)
            try:
                target.write_text(patched,encoding=encoding)
                if not pass_path.exists():
                    if syntax_error:result={'Result':'SyntaxError','error':syntax_error,'seconds':0,'attempts':[]}
                    else:result=tests(task,dst,out/'pass'/model/f'task_{i:04d}_logs')
                    save(pass_path,{'idx':i,'namespace':task['namespace'],'completion_sha256':hashlib.sha256(completion.encode()).hexdigest(),**result})
                if needs_dir and not dir_path.exists():
                    work=out/'dir'/model/f'task_{i:04d}_work';work.mkdir(parents=True,exist_ok=True)
                    if syntax_error:
                        parsed={'status':'syntax_error','error':syntax_error,'generated_dependency':None}
                    else:
                        record=work/'input.json';save(record,{'task':task})
                        cmd=[sys.executable,str(run/'code/evaluate.py'),'--dir-one','--record',str(record),
                             '--source',str(out/'workspace'),'--out',str(work)]
                        proc=subprocess_logged(cmd,work,work/'analyzer.log',120)
                        if proc['timeout']:parsed={'status':'analysis_timeout','generated_dependency':None}
                        elif (work/'parsed.json').exists():parsed=read(work/'parsed.json')
                        else:parsed={'status':'analysis_error','generated_dependency':None,'process':proc}
                    refs=set(d for ds in task['dependency'].values() for d in ds)
                    preds=set(d for ds in (parsed.get('generated_dependency') or {}).values() for d in ds)
                    save(dir_path,{'idx':i,'namespace':task['namespace'],'recall':len(refs&preds)/len(refs),
                         'reference_count':len(refs),'hit_dependencies':sorted(refs&preds),
                         'completion_sha256':hashlib.sha256(completion.encode()).hexdigest(),**parsed})
            finally:
                target.write_bytes(original)
                assert hashlib.sha256(target.read_bytes()).hexdigest()==source_hashes[str(rel)]
        print('task_done',i,task['namespace'],flush=True)
    save(out/'source_restore_hashes.json',source_hashes)
    if args.ground_truth_only:
        save(out/'GT_DONE.json',{'project':args.project,'tasks':len(records)})
        return
    save(out/'DONE.json',{'project':args.project,'tasks':len(records),'source_restored':True})
    finalize_completions_if_ready(run)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--project')
    p.add_argument('--ground-truth-only',action='store_true')
    p.add_argument('--dir-one',action='store_true');p.add_argument('--record',type=Path)
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.dir_one:dir_one(args)
    else:project_worker(args)


if __name__=='__main__':main()
