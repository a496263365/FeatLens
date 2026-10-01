import json
import re
import textwrap
import tokenize
from collections import defaultdict
from io import BytesIO
from itertools import product
from json import JSONDecodeError

from src.config import Config, CONSTANTS
from src.utils import Tools, FilePathBuilder
from src.utils.dataset_loader import DatasetLoader


class Extractor:
    @staticmethod
    def _extract_function(func_name: str, content: str, include_signature: bool = True) -> str:
        def _parse_tokens(_content: str):
            try:
                return list(tokenize.tokenize(BytesIO(_content.encode('utf-8')).readline))[1:]
            except Exception as e:
                if len(_content.split('\n')) == 1:
                    print(f'{e}')
                return _parse_tokens('\n'.join(_content.split('\n')[:-1]))

        tokens = _parse_tokens(content)
        start = [(0, 0)]
        end = [(0, 0)]
        indents = 0
        find_func = find_body = False
        for i, token in enumerate(tokens):
            if token.type == tokenize.NAME and token.string == func_name and tokens[i - 1].string == 'def':
                find_func = True
                if include_signature:
                    start.append(tokens[i - 1].start)
            elif find_func and token.type == tokenize.INDENT:
                indents += 4
                if not find_body and not include_signature:
                    start.append(token.start)
                find_body = True
            elif find_func and token.type == tokenize.DEDENT:
                indents -= 4
                if indents == 0:
                    end.append(token.start)
                    find_func = False
                    find_body = False
        lines = content.split('\n')[start[len(end) - 1][0] - 1: end[-1][0] - 1]
        return '\n'.join(lines).strip('\n')

    @staticmethod
    def extract_function(signature: str, generation: str, include_signature: bool = True) -> str:
        signature = signature.rstrip().replace('\t', '    ')
        signature = textwrap.dedent(signature)
        generation = generation.replace('\t', '    ')
        func_name = re.findall(r'def\s+([_\w]+?)\s*\(', signature)[0]
        def _remove_space(string: str) -> str:
            return ''.join(string.strip().split(' '))
        def_line = re.findall(r'def\s+.*', signature)[0]
        signature_prefix = _remove_space(def_line.split('(')[0])
        if Config.Generate.is_chat_llm and signature_prefix in _remove_space(generation):
            filtered_lines = []
            try:
                for line in generation.split('\n')[::-1]:
                    filtered_lines.append(line)
                    if signature_prefix in _remove_space(line) and '`' not in line:
                        spaces = len(line) - len(line.lstrip())
                        signature_lines = []
                        while True:
                            signature_lines.append(filtered_lines.pop(-1))
                            if signature_lines[-1].strip().endswith(':'):
                                break
                        parameters = iter(re.findall(r'[,(]\s*(\*{0,2}[_\w]+?)\s*[:,)]', '\n'.join(signature_lines)))
                        try:
                            signature = re.sub(
                                r'([,(]\s*)\*{0,2}[_\w]+?(\s*[:,)])',
                                lambda match: f'{match.group(1)}{next(parameters)}{match.group(2)}', signature
                            )
                        except StopIteration:
                            pass
                        import_lines = list(map(
                            lambda l: ' ' * 4 + l, filter(
                                lambda l: re.fullmatch(r'import\s+[ \w,.*]+', l) is not None or
                                          re.fullmatch(r'from\s+[ \w,.*]+\s+import\s+[ \w,.*]+', l) is not None,
                                map(lambda l: l.strip(), generation.split('\n'))
                            )
                        ))
                        function_in_prefix = '\n'.join([
                            signature, *import_lines, *map(lambda l: l[spaces:], filtered_lines[::-1])
                        ])
                        break
                else:
                    raise RuntimeError('No function detected')
            except Exception as e:
                print(e)
                function_in_prefix = f'{signature}\n' + ' ' * 4 + 'pass'
        else:
            if re.search(r'```.*?\n.*\n```', generation, re.DOTALL):
                generation = re.findall(r'```.*?\n(.*)\n```', generation, re.DOTALL)[-1]
            spaces = 0
            generation_lines = generation.splitlines()
            for line in generation_lines:
                if len(line.strip()) > 0:
                    spaces = len(line) - len(line.lstrip())
                    break
            function_in_prefix = f'{signature}\n' + '\n'.join(map(lambda l: ' ' * 4 + l[spaces:], generation_lines))
        function_in_prefix = textwrap.dedent(function_in_prefix)
        return Extractor._extract_function(func_name, function_in_prefix, include_signature)


class ExtractCodeWrapper:
    fragment_sizes = Config.Retrieval.fragment_sizes
    slice_sizes = Config.Retrieval.slice_sizes
    dataset = Config.Data.dataset
    api_level = Config.Data.api_level

    @classmethod
    def extract(cls, output_path: str = '', std_output_path: str = '', raw_prediction_path: str = ''):
        for fragment_size, slice_size in product(cls.fragment_sizes, cls.slice_sizes):
            if len(output_path + std_output_path) == 0:
                output_path = FilePathBuilder.extracted_prediction_path(fragment_size, slice_size)
                std_output_path = FilePathBuilder.standard_extracted_prediction_path(fragment_size, slice_size)
            if Tools.check_and_reuse(output_path) and Tools.check_and_reuse(std_output_path):
                continue
            if len(raw_prediction_path) > 0:
                raw_predictions = Tools.load_jsonl(raw_prediction_path)
            else:
                raw_predictions = Tools.load_jsonl(FilePathBuilder.raw_prediction_path(fragment_size, slice_size))
            task_id_to_predictions = defaultdict(list)
            for item in raw_predictions:
                task_id_to_predictions[item['task_id']].append(item)
            extracted_results = []
            std_extracted_results = []
            print('Extracting codes...')
            for task in DatasetLoader.load_tasks():
                task_id = task.task_id
                task_extracted_results = []
                include_signature = True
                if cls.api_level:
                    def for_evaluate_builder(codes: list):
                        return [{
                            'task_id': task_id,
                            'generate_results': codes,
                            'ground_truth': task.ref_code
                        }]

                elif cls.dataset == CONSTANTS.codereval:
                    def for_evaluate_builder(codes: list):
                        return [{
                            '_id': task_id,
                            'generate_results': codes,
                        }]

                elif cls.dataset == CONSTANTS.deveval:
                    def for_evaluate_builder(codes: list):
                        results = []
                        for c in codes:
                            results.append({
                                'namespace': task_id,
                                'completion': c,
                            })
                        return results

                    include_signature = False
                else:
                    raise NotImplementedError()
                for raw_prediction in task_id_to_predictions[task_id]:
                    try:
                        signature = raw_prediction['metadata']['task_input']
                    except KeyError:
                        signature = task.task_input
                    if cls.api_level:
                        try:
                            extracted = re.findall(r'```.*?\n(.*)\n```', raw_prediction['generation'], re.DOTALL)[0]
                        except IndexError:
                            extracted = raw_prediction['generation']
                    else:
                        extracted = Extractor.extract_function(
                            signature, raw_prediction['generation'], include_signature=include_signature
                        )
                    task_extracted_results.append(extracted)
                    std_extracted_results.append({
                        'task_id': task_id,
                        'extracted_generation': extracted,
                        **raw_prediction
                    })
                extracted_results.extend(for_evaluate_builder(task_extracted_results))

            Tools.dump_jsonl(extracted_results, output_path)
            Tools.dump_jsonl(std_extracted_results, std_output_path)

class ExtractPlanWrapper:
    def __init__(self, fragment_size, slice_size):
        self.raw_plans = {}
        if Config.Method.type == CONSTANTS.OURS and Config.Method.context_include_strategy == CONSTANTS.AUTO:
            raw_plan_predictions = Tools.load_jsonl(FilePathBuilder.plan_prediction_path(CONSTANTS.CONTEXT_PLAN, fragment_size, slice_size))
            for prediction_item in raw_plan_predictions:
                self.raw_plans[prediction_item['task_id']] = prediction_item

    def extract_and_map_plan(self, task_id: str):
        if self.raw_plans == {}:
            return
        raw_plan = self.raw_plans[task_id]['generation'].strip()
        raw_plan_lines = raw_plan.splitlines()
        for start_line in range(len(raw_plan_lines)):
            for end_line in range(start_line, len(raw_plan_lines)):
                span = '\n'.join(raw_plan_lines[start_line: end_line + 1])
                try:
                    extracted_plan = json.loads(span)
                    assert 'INVOKER' in extracted_plan and 'API' in extracted_plan
                    Config.Method.include_callers = extracted_plan.get('INVOKER', False)
                    Config.Method.include_call_chains = extracted_plan.get('API', False)
                    print(f'Context of {task_id} has been changed to: {extracted_plan}')
                    return
                except (JSONDecodeError, AssertionError, TypeError):
                    pass

if __name__ == '__main__':
    function = 'def a():'
    gen = """
        if a > 0:
            return 1
    """
    Extractor.extract_function(function, gen)

