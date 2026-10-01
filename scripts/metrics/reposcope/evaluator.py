
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
import shutil
import subprocess
import sys
import time
from abc import abstractmethod
from collections import defaultdict
from itertools import product

import black
import editdistance
from black import InvalidInput
from tqdm import tqdm

from src.config import Config, CONSTANTS
from src.generate.tokenizer import Tokenizer
from src.utils import Tools, FilePathBuilder


class BaseEvaluator:
    @classmethod
    @abstractmethod
    def evaluate(cls, prediction_path: str, result_path: str, copy_config: bool = True):
        pass

    @staticmethod
    def wait_process_and_report_result(process, copy_config: bool = True, result_path: str = '', prediction_path: str = '', task_key: str = ''):
        print('=' * 20 + 'Start evaluating' + '=' * 20)
        log_path = FilePathBuilder.evaluate_log_path(result_path)
        total = len(Tools.load_jsonl(prediction_path))
        bar = tqdm(total=total)
        ninety_seconds_ago = time.time()
        ninety_seconds_ago_num = 0
        while True:
            current = len(set(map(lambda x: x[task_key], Tools.load_jsonl(result_path))))
            bar.n = current
            bar.refresh()
            if current >= total:
                break
            if time.time() - ninety_seconds_ago >= 90:
                if current == ninety_seconds_ago_num:
                    print(f'No sample is evaluated in the past 90 seconds. We\'ll Re-run the evaluating script.')
                    process.kill()
                    process.wait()
                    return False
                ninety_seconds_ago = time.time()
                ninety_seconds_ago_num = current
            time.sleep(1)
        bar.close()
        stdout, stderr = process.communicate()
        Tools.write_file(f'[[stdout]]\n{stdout}[[stderr]]{stderr}', log_path)
        print(stdout)
        if len(stderr.strip()) > 0:
            print(f'Evaluation maybe failed, check log in {log_path}', file=sys.stderr)
        print('=' * 20 + 'End evaluating' + '=' * 20)
        if copy_config:
            shutil.copy(Config.config_path, FilePathBuilder.config_archive_path(result_path))
        return True

class CoderEvalEvaluator(BaseEvaluator):
    @classmethod
    def evaluate(cls, prediction_path: str, result_path: str, copy_config: bool = True):
        predictions = Tools.load_jsonl(prediction_path)
        n = len(predictions[0]['generate_results'])
        Tools.dump_jsonl([], result_path)
        base_eval_path = str(_external_path('FEATLENS_EVAL_WORKSPACE', 'work/evaluation', ''))
        proc = subprocess.Popen(
            ['docker', 'run', '--rm', '--name', 'code_graph_codereval',
             '-w', base_eval_path,
             '-v', f'{prediction_path}:{base_eval_path}/generation.jsonl',
             '-v', f'{result_path}:{base_eval_path}/generation.jsonl_out.jsonl',
             'codereval', '/bin/bash',
             '-c', f'python {base_eval_path}/PythonExec.py {base_eval_path}/generation.jsonl {n}'
             ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, text=True
        )
        cls.wait_process_and_report_result(
            proc, copy_config=copy_config, result_path=result_path, prediction_path=prediction_path, task_key='_id'
        )

class DevEvalEvaluator(BaseEvaluator):
    @classmethod
    def evaluate(cls, prediction_path: str, result_path: str, copy_config: bool = True):
        evaluate_script_base_path = f'{Config.Data.repos_path}_for_evaluate'
        while True:
            proc = subprocess.Popen(
                f'source ~/anaconda3/etc/profile.d/conda.sh && conda activate deveval && '
                f'python {os.path.join(evaluate_script_base_path, "pass_k.py")} '
                f'--output_file {prediction_path} --log_file {result_path} '
                f'--source_code_root {os.path.join(evaluate_script_base_path, "Source_Code")} '
                f'--data_file {os.path.join(evaluate_script_base_path, "data.jsonl")} '
                '--n 1 --k 1',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True, executable='/bin/bash'
            )
            if cls.wait_process_and_report_result(
                proc, copy_config=copy_config, result_path=result_path, prediction_path=prediction_path, task_key='namespace'
            ):
                break

class RepoEvalEvaluator(BaseEvaluator):
    @staticmethod
    def compute_em(target: str, predictions: list[str], passk: int):
        def format_code(code: str):
            mode = black.Mode()
            try:
                return black.format_str(code, mode=mode)
            except (InvalidInput, IndentationError):
                for line in code.splitlines():
                    if len(line.strip()) > 0:
                        try:
                            return black.format_str(line, mode=mode)
                        except InvalidInput:
                            pass
                return code

        formatted_target = format_code(target)
        em_scores = []
        for prediction in predictions[:passk]:
            if format_code(prediction) == formatted_target:
                em_scores.append(1)
            else:
                em_scores.append(0)
        return sum(em_scores) / len(em_scores)

    @staticmethod
    def compute_es(target: str, predictions: list[str], passk: int):
        target_lines = [line.strip() for line in target.splitlines() if line.strip()]
        target_str = '\n'.join(target_lines)
        es_scores = []
        for prediction in predictions[:passk]:
            prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
            prediction_str = '\n'.join(prediction_lines)
            es_scores.append(
                1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
            )
        return sum(es_scores) / len(es_scores)

    @classmethod
    def evaluate(cls, prediction_path: str, result_path: str, copy_config: bool = True):
        predictions = Tools.load_jsonl(prediction_path)
        em_scores = []
        es_scores = []
        results = []
        for prediction in predictions:
            generate_results = prediction['generate_results']
            ground_truth = prediction['ground_truth']
            em_scores.append(cls.compute_em(ground_truth, generate_results, len(generate_results)))
            es_scores.append(cls.compute_es(ground_truth, generate_results, len(generate_results)))
            results.append({
                'EM': em_scores[-1],
                'ES': es_scores[-1],
            })
        Tools.dump_jsonl(results, result_path)
        log_path = FilePathBuilder.evaluate_log_path(result_path)
        log = f'EM: {sum(em_scores) / len(em_scores):.4f}\nES: {sum(es_scores) / len(es_scores):.4f}\n'
        print(log)
        Tools.write_file(log, log_path)
        if copy_config:
            shutil.copy(Config.config_path, FilePathBuilder.config_archive_path(result_path))



class EvaluateResultWrapper:
    fragment_sizes = Config.Retrieval.fragment_sizes
    slice_sizes = Config.Retrieval.slice_sizes
    dataset = Config.Data.dataset

    @classmethod
    def evaluate(cls, prediction_path: str = '', evaluate_result_path: str = ''):
        for fragment_size, slice_size in product(EvaluateResultWrapper.fragment_sizes, EvaluateResultWrapper.slice_sizes):
            if len(prediction_path + evaluate_result_path) > 0:
                copy_config = False
                raw_prediction_path = prediction_path.replace('extracted.jsonl', 'raw_prediction.jsonl')
            else:
                prediction_path = FilePathBuilder.extracted_prediction_path(fragment_size, slice_size)
                evaluate_result_path = FilePathBuilder.evaluated_prediction_path(fragment_size, slice_size)
                raw_prediction_path = FilePathBuilder.raw_prediction_path(fragment_size, slice_size)
                copy_config = True
            if Tools.check_and_reuse(evaluate_result_path) and cls.dataset != CONSTANTS.deveval:
                print(f'Skip evaluation, check cached result in {evaluate_result_path}')
            else:
                print('Evaluating codes...')
                if cls.dataset == CONSTANTS.codereval:
                    evaluator = CoderEvalEvaluator
                elif cls.dataset == CONSTANTS.deveval:
                    evaluator = DevEvalEvaluator
                elif cls.dataset == CONSTANTS.repoeval:
                    evaluator = RepoEvalEvaluator
                else:
                    raise NotImplementedError()
                evaluator.evaluate(prediction_path, evaluate_result_path, copy_config=copy_config)
                print(f'Finish evaluation, check result in {evaluate_result_path}')

            log_path = FilePathBuilder.evaluate_log_path(evaluate_result_path)
            if cls.dataset == CONSTANTS.codereval:
                counts = defaultdict(int)
                for item in Tools.load_jsonl(evaluate_result_path):
                    for idx, generation in enumerate(item['generate_results']):
                        if generation['is_pass']:
                            counts[idx] += 1
                print(dict(counts))
                Tools.append_write_file(f'\n[[iter pass]]\t{dict(counts)}', log_path)
            token_len = []
            for line in Tools.load_jsonl(raw_prediction_path):
                if 'history' in line:
                    token_len.append(sum(len(Tokenizer.tokenize(history['query'])) for history in line['history']))
                else:
                    token_len.append(len(Tokenizer.tokenize(line['prompt'])))
            # print(sum(token_len) / len(token_len))
            Tools.append_write_file(f'\n[[prompt avg token length]]\t{sum(token_len) / len(token_len)}', log_path)


if __name__ == '__main__':
    EvaluateResultWrapper.evaluate()