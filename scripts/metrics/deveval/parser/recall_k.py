
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
from subprocess import run
from tqdm import tqdm
import os
from argparse import ArgumentParser
import textwrap
from add_func_call import process


# Root directory containing all DevEval project source code
SOURCE_CODE_ROOT = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', ''))
# Root directory of the dependency data required by DevEval recall@k evaluation; this must be downloaded
DEPENDENCY_DATA_ROOT = str(_external_path('FEATLENS_DEPENDENCY_DATA_ROOT', 'external/deveval/dependency_data', ''))
# Task file to evaluate, such as data.jsonl for the full DevEval set or a project-specific jsonl subset
# DATA_JSONL = '<external_root>/outputData/devEvalCompletionOut/0303_small_cross_test/combined_filtered_cross.jsonl'
DATA_JSONL = str(_package_path('data/deveval/tasks_all_1430.jsonl'))
# Output file to evaluate; LLM completion records must contain namespace and completion fields
# The recall@k evaluation CSV uses the OUTPUT_FILE base name plus a _recall.csv suffix and reports per-project and overall recall@k
# OUTPUT_FILE = '<external_root>/outputData/devEvalCompletionOut/0303_small_cross_test/combined_no_context_completion.jsonl'
OUTPUT_FILE = str(_package_path('results/generation/deveval/repograph/deepseek_v3_2_combined_completion.jsonl'))
# The evaluation output contains each task's recall information, such as dependencies correctly identified in generated code
# LOG_FILE = '<external_root>/outputData/devEvalCompletionOut/0303_small_cross_test/combined_no_context_recall_output.jsonl'
LOG_FILE = str(_package_path('results/generation/deveval/repograph/combined_repograph_recall_output.jsonl'))


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', default=OUTPUT_FILE, type=str)
    parser.add_argument('--log_file', default=LOG_FILE, type=str) 
    parser.add_argument('--k', default='1', type=str)
    parser.add_argument('--source_code_root', default=SOURCE_CODE_ROOT, type=str)
    parser.add_argument('--dependency_data_root', default=DEPENDENCY_DATA_ROOT, type=str)
    parser.add_argument('--data_file', default=DATA_JSONL, type=str)
    # This defaults to the current directory; it is a temporary directory used during recall@k evaluation and is cleared afterward
    parser.add_argument('--dependency_tmp_dir', type=str, default='dependency_data_tmp')
    return parser.parse_args()

def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code


def compute_recall(generated_dependency, reference_dependency):
    reference = []
    for _type, _list in reference_dependency.items():
        reference.extend(_list)
    if generated_dependency is None:
        return 0
    prediction = []
    for _type, _list in generated_dependency.items():
        prediction.extend(_list)
    reference = set(reference)
    prediction = set(prediction)
    recall = len(reference.intersection(prediction)) / len(reference)
    return recall
    

def report_results(args, k_list, output_data, benchmark_data):
    if not os.path.exists(args.log_file):
        raise ValueError("Output file not found")
    
    parse_results = dict()
    with open(args.log_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in parse_results:
                parse_results[namespace] = dict()
            parse_results[namespace][completion] = js['generated_dependency']

    results = {}
    for namespace, outputs in output_data.items():
        for output in outputs:
            completion = output['completion']
            if namespace in parse_results and namespace in benchmark_data:
                generated_dependency = parse_results[namespace][completion]
                data = benchmark_data[namespace]
                reference_dependency = data['dependency']
                recall = compute_recall(generated_dependency, reference_dependency)
                if namespace not in results:
                    results[namespace] = []
                results[namespace].append(recall)

    # Group namespaces by project (first segment of namespace, e.g. mrjob.job.MRJob.xxx -> mrjob)
    project_to_namespaces = {}
    for namespace in results:
        project = namespace.split('.')[0] if '.' in namespace else namespace
        if project not in project_to_namespaces:
            project_to_namespaces[project] = []
        project_to_namespaces[project].append(namespace)

    # Compute Recall@k overall and per project
    overall_recall_at_k = {}
    project_recall_at_k = {project: {} for project in project_to_namespaces}
    num_namespaces = len(results)

    for k in k_list:
        if num_namespaces == 0:
            overall_recall_at_k[k] = 0.0
        else:
            overall_recall_at_k[k] = sum(max(recall_list[:k]) for recall_list in results.values()) / num_namespaces
        print(f"Recall@{k}: {overall_recall_at_k[k]*100}%\n")
        for project, namespaces in project_to_namespaces.items():
            if namespaces:
                project_recall_at_k[project][k] = sum(max(results[ns][:k]) for ns in namespaces) / len(namespaces)
            else:
                project_recall_at_k[project][k] = 0.0

    # Write CSV: same directory as output_file, filename with _recall.csv suffix
    output_path = Path(args.output_file)
    csv_path = output_path.parent / (output_path.stem + '_recall.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['project'] + [f'recall_at_{k}' for k in k_list]
        writer.writerow(header)
        for project in sorted(project_to_namespaces.keys()):
            row = [project] + [f'{project_recall_at_k[project][k] * 100:.2f}%' for k in k_list]
            writer.writerow(row)
        row = ['overall'] + [f'{overall_recall_at_k[k] * 100:.2f}%' for k in k_list]
        writer.writerow(row)
    print(f"Results by project written to: {os.path.abspath(csv_path)}")


def SetUp_evaluation(args, data):
    completion = adjust_indent(data['completion'], data['indent'])
    completion_path = os.path.join(args.source_code_root, data['completion_path'])
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])

    # rename the original completion file as tmp_completion
    run(['cp', completion_path, completion_tmp_path])

    # write the new completion file
    sos, eos = data['body_position'][0]-1, data['body_position'][1]
    file_lines = []
    with open(completion_path, 'r') as f:
        file_lines = f.readlines()
    file_lines = file_lines[:sos] + ['\n', completion, '\n'] + file_lines[eos:]
    with open(completion_path, 'w') as f:
        f.write(''.join(file_lines))


def parse_dependency(args, data):
    project_root = os.path.join(args.source_code_root, data['project_path'])
    file_to_parse = os.path.join(args.source_code_root, data['completion_path'])
    output_path = os.path.join(args.dependency_tmp_dir, data['project_path'])
    analyzer_result_path = os.path.join(args.dependency_data_root, data['project_path'], 'analyzer_result.pkl')
    try:
        process(target_object=project_root, func_object_root=project_root, func_path=file_to_parse,
                analyzer_result=analyzer_result_path, target_root=output_path)
    except Exception as e:
        return False
    return True


def extract_dependency(args, data):
    generated_dependency = {'intra_class': [], 'intra_file': [], 'cross_file': []}
    dependency_path = os.path.join(args.dependency_tmp_dir, data['completion_path'].replace('.py', '.json'))
    if not os.path.exists(dependency_path):
        return generated_dependency
    with open(dependency_path, 'r') as f:
        dependency_data = json.load(f)
    if data['namespace'] not in dependency_data:
        return generated_dependency
    attributes = dependency_data[data['namespace']]
    for _item in attributes['in_class']:
        generated_dependency['intra_class'].append(_item['name'])
    for _item in attributes['in_file']:
        generated_dependency['intra_file'].append(_item['name'])
    for _item in attributes['in_object']:
        generated_dependency['cross_file'].append(_item['name'])
    return generated_dependency


def TearDown_evaluation(args, data):
    completion_path = os.path.join(args.source_code_root, data['completion_path'])
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    dependency_tmp_path = os.path.join(args.dependency_tmp_dir, data['project_path'])

    run(['mv', completion_tmp_path, completion_path])
    run(['rm', '-rf', dependency_tmp_path])


def is_standalone(data):
    dependency = data['dependency']
    if len(dependency['intra_class']) + len(dependency['intra_file']) + len(dependency['cross_file']) == 0:
        return True
    return False


def load_finished_data(args):
    finished_data = dict()
    if os.path.exists(args.log_file):
        with open(args.log_file, 'r') as f:
            for line in f:
                js = json.loads(line)
                if js['namespace'] not in finished_data:
                    finished_data[js['namespace']] = set()
                finished_data[js['namespace']].add(js['completion'])
    return finished_data


def main():
    args = get_parser()

    # Parse the k values
    k_list = []
    for _k in args.k.split(','):
        k_list.append(int(_k))
    max_k = max(k_list)

    # Load the completion data and finished data
    finished_data = load_finished_data(args)
    output_data = dict()
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            if namespace not in output_data:
                output_data[namespace] = []
            if len(output_data[namespace]) < max_k: # only consider max_k completions
                output_data[namespace].append(js)
    
    benchmark_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
    
    # Skip the finished data, deuplicate completions, and standalone completions
    todo_output_data = []
    for namespace, outputs in output_data.items():
        assert len(outputs) == max_k, print(len(outputs))
        if namespace in benchmark_data:
            data = benchmark_data[namespace]
            if is_standalone(data):
                continue
            for output in outputs:   # only consider max_k completions
                completion = output['completion']
                if namespace not in finished_data:
                    todo_output_data.append(output)
                    finished_data[namespace] = set()
                    finished_data[namespace].add(completion)
                elif completion not in finished_data[namespace]:
                    todo_output_data.append(output)
                    finished_data[namespace].add(completion)
    print(f"TODO Completions: {len(todo_output_data)}\n")

    # release memory
    del finished_data
            
    with open(args.log_file, 'a') as f:
        for output in tqdm(todo_output_data):
            if output['completion'] == "    pass\n":
                output['generated_dependency'] = None
            else:
                data = benchmark_data[output['namespace']]
                data['completion'] = output['completion']
                SetUp_evaluation(args, data)
                if parse_dependency(args, data) == True:
                    generated_dependency = extract_dependency(args, data)
                    output['generated_dependency'] = generated_dependency
                else:
                    output['generated_dependency'] = None
                TearDown_evaluation(args, data)
            f.write(json.dumps(output) + '\n')
            f.flush()
    
    report_results(args, k_list, output_data, benchmark_data)

if __name__ == '__main__':
    main()