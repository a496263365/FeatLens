"""Read-only localization audit plus dataset-specific progress/final summaries.

Does not launch workers, rerun inference, or overwrite raw trajectories.
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
import collections
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import time
import tokenize
from pathlib import Path

import common

OUT = (_package_path() / 'results/retrieval/evocodebench/locagent')
CONFIG = json.loads(((_package_path() / 'results/retrieval/evocodebench/locagent/config.json')).read_text())
TASKS = {t['example_id']: t for t in common.jl((_package_path() / 'results/retrieval/evocodebench/locagent/tasks.jsonl'))}
GOOD = {'completed', 'budget_exhausted', 'context_limited'}


def atomic(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data)
    temp.replace(path)


def dump(path, obj):
    atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


def csvout(path, records):
    if not records:
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live_controller():
    try:
        identity = json.loads(((_package_path() / 'results/retrieval/evocodebench/locagent/controller_identity.json')).read_text())
        cmd = (Path('/proc') / str(identity['pid']) / 'cmdline').read_bytes().split(b'\0')
        return str((_package_path() / 'scripts/retrieval/locagent/evocodebench/controller.py')).encode() in cmd
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def read_selected():
    records, failures, summaries = [], [], []
    for tid, task in TASKS.items():
        path = (_package_path() / 'results/retrieval/evocodebench/locagent/tasks') / f'task_{tid:06d}' / 'locagent' / 'job_summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        assert summary['task_id'] == tid and summary['method'] == 'locagent'
        summaries.append(summary)
        if summary['state'] not in GOOD:
            failures.append(summary)
            continue
        result_path = Path(summary['result_file'])
        assert result_path.resolve().is_relative_to((_package_path() / 'results/retrieval/evocodebench/locagent/tasks'))
        result = json.loads(result_path.read_text())
        assert result['namespace'] == task['namespace'] and result['task_id'] == tid
        assert result['status'] == summary['state'] and result['model'] == CONFIG['model']
        stage = result['final_stage']
        assert stage['dependency_count'] == len(set(sum(task['dependency'].values(), [])))
        assert abs(stage['dr'] - len(set(stage['hit_dependencies'])) / stage['dependency_count']) < 1e-12
        assert summary['dr'] == stage['dr'] and 0 <= stage['exact_entity_dr'] <= stage['dr'] <= 1
        usage = result['usage']
        records.append(dict(
            task_id=tid, project=task['project_path'], namespace=task['namespace'],
            status=result['status'], dr=stage['dr'], exact_dr=stage['exact_entity_dr'],
            entities=stage['returned_entities'], files=stage['returned_files'],
            unresolved=stage['unresolved_locations'], calls=usage['calls'],
            prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
            total_tokens=usage['total_tokens'], unknown_usage_requests=usage['unknown_usage_requests'],
            seconds=summary['seconds'], result_file=str(result_path),
        ))
    return records, failures, summaries


def mean(records, field):
    return sum(r[field] for r in records) / len(records) if records else None


def publish(final=False):
    records, failures, summaries = read_selected()
    by_project = []
    for project in CONFIG['project_names']:
        selected = [r for r in records if r['project'] == project]
        expected = sum(t['project_path'] == project for t in TASKS.values())
        bad = sum(r.get('project') == project for r in failures)
        by_project.append(dict(
            project=project, expected=expected, scored=len(selected), failed=bad,
            pending=expected-len(selected)-bad, dr=mean(selected, 'dr'),
            exact_dr=mean(selected, 'exact_dr'), mean_entities=mean(selected, 'entities'),
        ))
    states = dict(collections.Counter(r['status'] for r in records))
    report = dict(
        dataset='EvoCodeBench', model=CONFIG['model'], expected_tasks=109, all_tasks=123,
        zero_dependency_tasks=14, projects=5, terminal_tasks=len(summaries), scored_tasks=len(records),
        failed_tasks=len(failures), remaining=109-len(summaries),
        all_jobs_terminal=len(summaries)==109, all_tasks_scored=len(records)==109,
        dr=mean(records, 'dr'), exact_dr=mean(records, 'exact_dr'),
        mean_entities=mean(records, 'entities'), mean_calls=mean(records, 'calls'),
        current_result_prompt_tokens=sum(r['prompt_tokens'] for r in records),
        current_result_completion_tokens=sum(r['completion_tokens'] for r in records),
        states=states, updated_at=datetime.datetime.now().astimezone().isoformat(),
        metric='unchanged DevEval task-macro relaxed and exact-entity DR; final reported locations; no Top-K cap',
        status='FINAL' if final else 'RUNNING',
    )
    csvout((_package_path() / 'results/retrieval/evocodebench/locagent/task_results.csv'), records)
    csvout((_package_path() / 'results/retrieval/evocodebench/locagent/per_project.csv'), by_project)
    dump((_package_path() / 'results/retrieval/evocodebench/locagent/summary.json'), report)
    dump((_package_path() / 'results/retrieval/evocodebench/locagent/failures.json'), failures)
    simple = {k:v for k,v in report.items() if k not in {'states'}}
    csvout((_package_path() / 'results/retrieval/evocodebench/locagent/summary.csv'), [simple])
    lines = ['# EvoCodeBench LocAgent：5轮 / 32K / 10并发', '',
             f"更新时间：{report['updated_at']}",
             f"已评分 {len(records)}/109；失败 {len(failures)}；未结束 {report['remaining']}。",
             '**全量运行结束**' if final else '**运行中：完成子集均值不是全量结果。**', '',
             '|项目|已评分/应评分|宽松DR|严格实体DR|失败|', '|---|---:|---:|---:|---:|']
    for row in by_project:
        dr = '—' if row['dr'] is None else f"{row['dr']:.2%}"
        exact = '—' if row['exact_dr'] is None else f"{row['exact_dr']:.2%}"
        lines.append(f"|{row['project']}|{row['scored']}/{row['expected']}|{dr}|{exact}|{row['failed']}|")
    if records:
        lines += ['', f"任务平均宽松DR：**{report['dr']:.2%}**；严格实体DR：{report['exact_dr']:.2%}。",
                  f"平均返回实体：{report['mean_entities']:.2f}；平均模型调用：{report['mean_calls']:.2f}。"]
    lines += ['', '原五项目共有123任务，14个无参考依赖任务DR未定义，不按零分或满分处理。',
              '只做最终依赖定位，不生成/修复/测试代码。5次调用包含最终答案；32K是本地估计，不是供应商精确tokenizer。',
              '使用与DevEval相同匹配函数，包括完整类表示和宽松变量匹配的既有局限。',
              '当前结果token与前序中断尝试成本分开；raw请求/响应和错误记录均保留。']
    atomic((_package_path() / 'results/retrieval/evocodebench/locagent/EVO_PROGRESS.md'), '\n'.join(lines)+'\n')
    if final:
        atomic((_package_path() / 'results/retrieval/evocodebench/locagent/REPORT.md'), '\n'.join(lines)+'\n')
    return records, summaries, report


def verify_final(records, summaries):
    spec = importlib.util.spec_from_file_location('verify_mask', (_package_path() / 'scripts/retrieval/locagent/evocodebench/task_masking.py'))
    masker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(masker)
    hashes, problems, budget_checks = {}, [], 0
    for row in records:
        tid = row['task_id']; task = TASKS[tid]
        d = Path(row['result_file']).parent
        result = json.loads((d/'result.json').read_text())
        locs = result['final_locations']
        assert common.score(task, locs, {}) == result['final_stage'], tid
        assert all(common.canonical(e['symbol']) != common.canonical(task['namespace']) for e in locs['entities'])
        taskdir = (_package_path() / 'results/retrieval/evocodebench/locagent/tasks')/f'task_{tid:06d}'
        maskmeta = json.loads((taskdir/'mask_manifest.json').read_text())
        source = taskdir/'source'/task['completion_path']
        assert sha(source)==maskmeta['source_masked_sha256']
        with tokenize.open(source) as handle:
            code=handle.read()
        node=masker.locate(code,task)
        assert len(node.body)==1 and isinstance(node.body[0],ast.Pass)
        for entity in locs['entities']:
            file=taskdir/'source'/task['project_path']/entity['file']
            assert file.resolve().is_relative_to((taskdir/'source'/task['project_path']).resolve())
            with tokenize.open(file) as handle:
                lines=handle.read().splitlines()
            assert '\n'.join(lines[entity['start']-1:entity['end']]) == entity['code']
        responses = list((d/'api').glob('call_*/attempt_*_response.json'))
        errors = list((d/'api').glob('call_*/attempt_*_error.json'))
        assert len(responses)==result['usage']['successful_calls']
        assert len(errors)==result['usage']['failed_requests']
        assert result['usage']['calls']<=5
        for call in (d/'api').glob('call_*'):
            request=json.loads((call/'request.json').read_text())
            b=json.loads((call/'context_budget.json').read_text())
            assert request['model']==CONFIG['model'] and request['temperature']==0
            assert request['max_tokens']==4096
            assert request['extra_body']=={'thinking':{'type':'disabled'}}
            assert b['estimated_input_tokens']<=b['input_limit']==27648
            budget_checks+=1
        for f in [d/'result.json',d/'trajectory.json',taskdir/'mask_manifest.json']:
            hashes[str(f.relative_to(OUT))]=sha(f)
    input_proof=json.loads(((_package_path() / 'results/retrieval/evocodebench/locagent/INPUTS_VERIFIED.json')).read_text())
    for rel,h in input_proof['frozen_files'].items():
        assert sha(OUT/rel)==h,rel
    for path,h in input_proof['source_hashes'].items():
        assert sha(Path(path))==h,path
    # Retain cost of early/orphan attempts separately; no cherry-picked scores.
    selected={str(Path(r['result_file']).parent) for r in records}
    cost=collections.defaultdict(lambda:collections.Counter())
    for d in ((_package_path() / 'results/retrieval/evocodebench/locagent/tasks')).glob('task_*/locagent/attempts/*'):
        cohort='selected_results' if str(d) in selected else 'other_attempts'
        for p in (d/'api').glob('call_*/attempt_*_response.json'):
            x=json.loads(p.read_text());u=x['response'].get('usage') or {}
            cost[cohort]['successful_responses']+=1
            cost[cohort]['prompt_tokens']+=u.get('prompt_tokens',0) or 0
            cost[cohort]['completion_tokens']+=u.get('completion_tokens',0) or 0
        cost[cohort]['error_responses']+=len(list((d/'api').glob('call_*/attempt_*_error.json')))
        for call in (d/'api').glob('call_*'):
            if (call/'request.json').exists() and not list(call.glob('attempt_*_response.json')) and not list(call.glob('attempt_*_error.json')):
                cost[cohort]['requests_without_returned_usage_or_error']+=1
    dump((_package_path() / 'results/retrieval/evocodebench/locagent/cost_all_attempts.json'),dict(cost))
    dump((_package_path() / 'results/retrieval/evocodebench/locagent/selected_result_hashes.json'),hashes)
    proof=dict(scored_tasks=len(records),terminal_tasks=len(summaries),expected_tasks=109,
               full_coverage=len(records)==109,score_recomputed_for_every_result=True,
               target_masking_verified=True,selected_entity_code_verified=True,
               budget_checks=budget_checks,inputs_and_frozen_code_unchanged=True,
               no_task_rerun_in_verification=True,issues=problems)
    dump((_package_path() / 'results/retrieval/evocodebench/locagent/COMPLETION_VERIFIED.json'),proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    with open((_package_path() / 'results/retrieval/evocodebench/locagent/publisher.lock'),'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            terminal=((_package_path() / 'results/retrieval/evocodebench/locagent/ALL_JOBS_TERMINAL.json')).exists()
            records,summaries,report=publish(final=terminal)
            if terminal:
                assert len(summaries)==109
                verify_final(records,summaries)
                print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
                return
            if not args.wait:
                print(json.dumps(report,ensure_ascii=False,indent=2))
                return
            if not live_controller():
                dump((_package_path() / 'results/retrieval/evocodebench/locagent/PUBLISHER_NEEDS_ATTENTION.json'),dict(reason='No live controller and no terminal marker'))
                raise SystemExit('Controller absent; do not claim completion')
            time.sleep(30)


if __name__=='__main__':
    main()
