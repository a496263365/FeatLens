
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

from pathlib import Path
import csv
import json
import subprocess
import psutil
from subprocess import run
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser
import textwrap
from func_timeout import func_set_timeout
import func_timeout

# Root directory containing all DevEval project source code
SOURCE_CODE_ROOT = Path(str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', '')))
# Task file to evaluate, such as data.jsonl for the full DevEval set or a project-specific jsonl subset
DATA_JSONL = Path(str(_package_path('data/deveval/tasks_all_1430.jsonl')))
# DATA_JSONL = Path('<external_root>/testRepoSummaryOut/211/mrjob/filtered.jsonl')
# Output file to evaluate; LLM completion records must contain namespace and completion fields
# The pass@k evaluation CSV uses the same base name as OUTPUT_FILE and contains per-project and overall pass@k
OUTPUT_FILE = Path(str(_package_path('results/generation/deveval/repograph/deepseek_v3_2_combined_completion.jsonl')))
# OUTPUT_FILE = Path('<external_root>/outputData/devEvalCompletionOut/mrjob/0316_cut/bm25_rag_completion.jsonl')
# The evaluation output file contains a Result field indicating whether each task passed
LOG_FILE = Path(str(_package_path('results/generation/deveval/repograph/combined_repograph_test_output.jsonl')))
# LOG_FILE = Path('<external_root>/outputData/devEvalCompletionOut/mrjob/0316_cut/bm25_rag_test_output.jsonl')


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path, default=OUTPUT_FILE)
    parser.add_argument('--log_file', type=Path, default=LOG_FILE)
    parser.add_argument('--source_code_root', type=Path, default=SOURCE_CODE_ROOT)
    parser.add_argument('--data_file', type=Path, default=DATA_JSONL) # data.jsonl
    parser.add_argument('--k', type=str, default='1') # k in pass_at_k
    parser.add_argument('--n', type=int, default=1) # number of completions per task
    parser.add_argument('--failure_log', type=Path, default=Path('failure_details.log'),
                        help='Log file for failed test details (error messages, stderr, etc.)')
    return parser.parse_args()


def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code


def _read_process_output(process):
    """Read stdout/stderr from process; communicate() also waits for process to end (no zombie)."""
    try:
        out, err = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
    process.wait()  # no-op if already ended; makes wait semantics explicit
    return (
        (out or b'').decode('utf-8', errors='replace'),
        (err or b'').decode('utf-8', errors='replace'),
    )


@func_set_timeout(30)
def execution_tests(args, data):
    project_path = os.path.join(args.source_code_root, data['project_path'])
    command = ['python', 'setup.py', 'pytest', '--addopts']
    for test in data['tests']:
        process = subprocess.Popen(command + [test], cwd=project_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while True:
                process_id = process.pid
                process_memory = psutil.Process(process_id).memory_info().rss
                if process_memory > 5 * 1024 * 1024 * 1024:  # 5GB memory usage per test
                    process.terminate()
                    out, err = _read_process_output(process)
                    msg = f"[OOM] test: {test}\nstdout:\n{out}\nstderr:\n{err}".strip()
                    return ('OOM', msg)
                return_code = process.poll()
                if return_code is not None:
                    if return_code != 0:
                        out, err = _read_process_output(process)
                        msg = f"test: {test}\nstdout:\n{out}\nstderr:\n{err}".strip() or f"Process exited with code {return_code}"
                        return ('Error', msg)
                    else:
                        break
        except Exception as e:
            process.terminate()
            out, err = _read_process_output(process)
            msg = f"Exception: {e}\ntest: {test}\nstdout:\n{out}\nstderr:\n{err}".strip()
            return ('Error', msg)
        finally:
            process.terminate()
            process.wait()
    return ('Pass', '')


def compute_pass_at_k(n, c, k):
    """
    n: total number of completions per task
    c: number of completions that pass all tests
    k: k in pass_at_k
    """
    if n - c < k:
        return 1
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))


def SetUp_evaluation(args, data, completion):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])

    # rename the original completion file as tmp_completion
    run(['cp', completion_path, completion_tmp_path])

    # write the new completion file
    sos, eos = data['body_position'][0]-1, data['body_position'][1]
    with open(completion_path, 'r') as f:
        file_lines = f.readlines()
    file_lines = file_lines[:sos] + ['\n', completion, '\n'] + file_lines[eos:]
    with open(completion_path, 'w') as f:
        f.write(''.join(file_lines))


def TearDown_evaluation(args, data):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['mv', completion_tmp_path, completion_path])


def check_correctness(args, data):
    """Returns (flag, detail) where flag is 'Pass'|'Error'|'TimeOut'|'OOM', detail is error message when not Pass."""
    completion = data['completion']
    if completion == "    pass\n":
        return ('Error', 'Completion is placeholder "pass" only')
    completion = adjust_indent(completion, data['indent'])

    SetUp_evaluation(args, data, completion)
    try:
        flag, detail = execution_tests(args, data)
    except func_timeout.exceptions.FunctionTimedOut as e:
        flag, detail = 'TimeOut', str(e)
    TearDown_evaluation(args, data)
    return (flag, detail)


def report_results(args, benchmark_data):
    if not os.path.exists(args.log_file):
        raise ValueError(f'{args.log_file} does not exist')
    
    # Collect passed completions for each namespace
    passed_completion = {}
    with open(args.log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            if 'pass' in js:
                js['Result'] = js['pass']
            if js['Result'] == 'Pass':
                namespace, completion = js['namespace'], js['completion']
                if namespace not in passed_completion:
                    passed_completion[namespace] = set()
                passed_completion[namespace].add(completion)

    # Iterate through all completions and count the number of passed completions for each namespace
    results = {}
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in benchmark_data:
                continue
            if namespace not in results:
                results[namespace] = 0
            if namespace in passed_completion and completion in passed_completion[namespace]:
                results[namespace] += 1

    # Group namespaces by project (first segment of namespace, e.g. mrjob.job.MRJob.xxx -> mrjob)
    project_to_namespaces = {}
    for namespace in results:
        project = namespace.split('.')[0] if '.' in namespace else namespace
        if project not in project_to_namespaces:
            project_to_namespaces[project] = []
        project_to_namespaces[project].append(namespace)

    # Compute Pass@k for overall and per project
    k_list = [int(k) for k in args.k.split(',')]
    k_list = [k for k in k_list if k <= args.n]
    overall_pass_at_k = {}
    project_pass_at_k = {project: {} for project in project_to_namespaces}

    for k in k_list:
        overall_pass_at_k[k] = np.mean([compute_pass_at_k(args.n, results[ns], k) for ns in results])
        print(f'pass_at_{k}: {overall_pass_at_k[k]*100}%')
        for project, namespaces in project_to_namespaces.items():
            if namespaces:
                project_pass_at_k[project][k] = np.mean([compute_pass_at_k(args.n, results[ns], k) for ns in namespaces])
            else:
                project_pass_at_k[project][k] = 0.0

    # Write CSV: same directory as output_file, filename with .csv extension
    csv_path = args.output_file.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['project'] + [f'pass_at_{k}' for k in k_list]
        writer.writerow(header)
        for project in sorted(project_to_namespaces.keys()):
            row = [project] + [f'{project_pass_at_k[project][k] * 100:.2f}%' for k in k_list]
            writer.writerow(row)
        row = ['overall'] + [f'{overall_pass_at_k[k] * 100:.2f}%' for k in k_list]
        writer.writerow(row)
    print(f'Results by project written to: {os.path.abspath(csv_path)}')


def load_finished_data(args):
    finished_data = {}
    if os.path.exists(args.log_file):
        with open(args.log_file, 'r') as f:
            for line in f:
                js = json.loads(line)
                namespace, completion = js['namespace'], js['completion']
                if namespace not in finished_data:
                    finished_data[namespace] = set()
                finished_data[namespace].add(completion)
    return finished_data


def main(args):
    finished_data = load_finished_data(args)

    todo_output_data = []
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in finished_data:
                todo_output_data.append(js)
                finished_data[namespace] = set()
                finished_data[namespace].add(completion)
            elif completion not in finished_data[namespace]: 
                todo_output_data.append(js)
                finished_data[namespace].add(completion)         
    del finished_data
    print("TODO Completions: ", len(todo_output_data))
    print("Failure details log: ", os.path.abspath(args.failure_log))

    benchmark_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js

    # Create failure_log at start so it always appears in the directory; failures append below
    if not args.failure_log.exists():
        with open(args.failure_log, 'w', encoding='utf-8') as fl:
            fl.write(json.dumps({"_info": "Failure details (one JSON object per line per failed completion)."}, ensure_ascii=False) + "\n")

    with open(args.log_file, 'a') as f:
        for output in tqdm(todo_output_data):
            if output['namespace'] in benchmark_data:
                data = benchmark_data[output['namespace']]
                data['completion'] = output['completion']
                flag, detail = check_correctness(args, data)
                output['Result'] = flag
                f.write(json.dumps(output) + '\n')
                f.flush()
                if flag != 'Pass':
                    entry = {
                        'namespace': output['namespace'],
                        'result': flag,
                        'message': detail,
                        'completion_preview': (output['completion'][:500] + '...') if len(output['completion']) > 500 else output['completion'],
                    }
                    with open(args.failure_log, 'a', encoding='utf-8') as fl:
                        fl.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        fl.flush()

    report_results(args, benchmark_data)


def test_ground_truth(args):
    data = open(args.data_file, 'r').readlines()
    output_f = open('failed_samples.jsonl', 'w')

    if not args.failure_log.exists():
        with open(args.failure_log, 'w', encoding='utf-8') as fl:
            fl.write(json.dumps({"_info": "Ground-truth failure details (one JSON object per line)."}, ensure_ascii=False) + "\n")
    print("Failure details log: ", os.path.abspath(args.failure_log))

    for line in tqdm(data):
        js = json.loads(line)
        tests = set(js['tests'])
        js['tests'] = list(tests)
        try:
            flag, detail = execution_tests(args, js)
        except func_timeout.exceptions.FunctionTimedOut as e:
            flag, detail = 'TimeOut', str(e)
        if flag != 'Pass':
            print(js['namespace'])
            output_f.write(json.dumps(js) + '\n')
            entry = {
                'namespace': js['namespace'],
                'result': flag,
                'message': detail,
            }
            with open(args.failure_log, 'a', encoding='utf-8') as fl:
                fl.write(json.dumps(entry, ensure_ascii=False) + '\n')
                fl.flush()


if __name__ == '__main__':
    args = get_parser()
    if args.output_file is None:
        test_ground_truth(args)
    else:
        main(args)