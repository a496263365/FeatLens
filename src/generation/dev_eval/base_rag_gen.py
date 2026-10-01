
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
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import pandas as pd

from llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import DevEvalTask, parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import resolve_signature
from utils.task_recall import compute_task_recall, load_enre_elements


SOURCE_CODE_DIR = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', ''))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
DIAGNOSTIC_JSONL = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/diagnostic_bm25_code.jsonl'))
OUTPUT_COMPLETION_PATH = str(_package_path('results/generation/deveval/bm25/deepseek_v3_2/completion.jsonl'))

# Large language model used for code generation
MODEL_NAME = "deepseek-v3.2"
MODEL_BACKEND_CHOICE = "openai"

# RAG data source; supported values include "bm25", "unixcoder", "feature", and "feature+bm25"
RAG_DATA_SOURCE = "bm25"

DEBUG = True  # Whether to print debug information to the console
DEBUG_LOG_FULL = True  # When DEBUG is True, write the complete prompt and completions to the log file
GENERATION_FLAG = True  # Whether to perform code generation; defaults to True, set to False to compute context recall only

PROMPT_TEMPLATE = (
    "Please complete the function in the given Python code"
    "located at the end of the instuction based on relevant repository information.\n\n"
    "Constraints:\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n"
    "- You can reference the code fragments from the repo to help you complete the function!\n\n"
    "Here are some relevant code fragments from the repo:\n"
    "{{context_code_in_prompt}}\n\n\n\n"
    "Input Code (You should complete):\n"
    "```Python\n"
    "{{signature}}\n\n"
    "{{requirement_comment}}\n\n"
    "```\n\n"
    "Completed Code:\n"
)


# Global variable storing all method information so complete code can be retrieved by signature
method_sig_to_info = {}

# Load all previously processed method information; batch runs clear and reload it for each project
def load_methods_info(methods_csv: str) -> None:
    method_sig_to_info.clear()
    df_methods = pd.read_csv(methods_csv)
    print("Loading METHODS_CSV...")

    for index, row in df_methods.iterrows():
        sig = str(row['method_signature'])
        method_sig_to_info[sig] = row.to_dict()

    print(f"Loaded {len(method_sig_to_info)} methods from CSV.")


# Load previous search results, including match indicators
def load_diagnostic_result(diagnostic_jsonl: str) -> list[Dict[str, Any]]:
    print("Loading diagnostic_jsonl...")
    diag_records = []
    
    with open(diagnostic_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    return diag_records


# Obtain the corresponding task search results for later use as LLM context
def get_searched_context_code(
    task: DevEvalTask, diag_record: Dict[str, Any], rag_data_source: str
) -> list[Dict[str, Any]]:
    task_namespace = task.namespace
    preds = []
    try:
        if rag_data_source == "bm25":
            preds = diag_record["bm25_code"]["top15"]["predictions"]
        elif rag_data_source == "unixcoder":
            preds = diag_record["unixcoder_code"]["top15"]["predictions"]
        elif rag_data_source == "feature":
            preds = diag_record["feature"]["top3"]["predictions"]
        elif rag_data_source == "feature+bm25":
            # Select the top 10 first because later steps expand and filter the initial search results
            preds = diag_record["hybrid"]["recall_top3_clusters"]["rank_top10"]["predictions"]
        
        # Remove methods in preds whose method equals task_namespace, namely the ground-truth code to complete
        filtered_preds = [p for p in preds if (p['method'] if '(' not in p['method'] else p['method'].split('(')[0]) != task_namespace]
        if len(filtered_preds) != len(preds):
            print(f"Attention! {len(preds) - len(filtered_preds)} items filtered out for task {task_namespace}")
        preds = filtered_preds
    except KeyError:
        print(f"Skipping task {task_namespace}: structure not matching data['feature']['top3']['predictions']")
        return []
    
    # Use the complete method signature only
    pred_signatures = [p['method'] for p in preds]
    context_code_list = []
    for sig in pred_signatures:
        # Find the method details corresponding to the signature in the method CSV
        if sig in method_sig_to_info:
            csv_info = method_sig_to_info[sig]
            context_code = {
                'method_signature': str(csv_info.get('method_signature', '')),
                'func_file': str(csv_info.get('func_file', '')),
                'method_code': str(csv_info.get('method_code', '')),
            }
            context_code_list.append(context_code)
        else:
            print(f"Warning! Signature {sig} not found in methods csv")
    
    return context_code_list


# Join retrieved code fragments as context for the prompt
def assemble_context_code_into_prompt(context_code_list: list[Dict[str, Any]]) -> str:
    context_code_in_prompt = ""
    max_len = 3000
    for context_code in context_code_list:
        method_code = str(context_code.get("method_code", ""))
        if len(method_code) > max_len:
            method_code = method_code[:max_len] + "..."
        context_code_in_prompt += (
            f"{context_code.get('func_file', '')}\n"
            f"{method_code}\n\n"
        )
    return context_code_in_prompt


# Format the requirement text as a multiline comment appended below the function signature
def format_requirement_as_comment(requirement_text: str) -> str:
    if not requirement_text:
        return ""
    lines = requirement_text.splitlines()
    indent_str = "    "  # Use four spaces per indentation level
    delimiter = '"""'  # Represent multiline comments with triple quotes
    escaped_delimiter = '\\"\\"\\"'
    lines = [line.replace(delimiter, escaped_delimiter) for line in lines]

    content = "\n".join(indent_str + line if line else indent_str for line in lines)
    return f"{indent_str}{delimiter}\n{content}\n{indent_str}{delimiter}\n"


def build_prompt(signature: str, requirement_comment: str, context_code_in_prompt: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
        .replace("{{context_code_in_prompt}}", context_code_in_prompt)
    )


def generate_completions(
    *,
    filtered_path: str,
    source_code_dir: str,
    methods_csv: str,
    enre_json: str,
    diagnostic_jsonl: str,
    rag_data_source: str,
    output_jsonl: str,
    debug_log_path_override: Optional[str] = None,
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    timeout_s: float,
    max_tasks: Optional[int],
    sleep_s: float,
) -> None:

    load_methods_info(methods_csv)
    load_enre_elements(enre_json)

    # Clear the output JSONL file
    with open(output_jsonl, "w", encoding="utf-8"):
        pass

    client = make_client(
        backend=backend,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    diag_records = load_diagnostic_result(diagnostic_jsonl)

    debug_log_path = None
    if DEBUG and DEBUG_LOG_FULL:
        debug_log_path = debug_log_path_override or (os.path.splitext(output_jsonl)[0] + "_debug.log")
        with open(debug_log_path, "w", encoding="utf-8") as _:
            pass

    processed = 0
    recall_sum = 0.0
    recall_count = 0
    recall_none_count = 0
    for record in iter_jsonl(filtered_path):
        task = parse_task(record)

        diag_record = diag_records[processed]

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )

        # Obtain the code-search results for this task
        searched_context_code_list = get_searched_context_code(task, diag_record, rag_data_source)
        print(f"code len: {len(searched_context_code_list)}")
        context_code_in_prompt = assemble_context_code_into_prompt(searched_context_code_list)

        # Compute context recall for this task
        recall_info = compute_task_recall(task.dependency, searched_context_code_list)
        if recall_info["recall"] is None:
            recall_none_count += 1
        else:
            recall_sum += float(recall_info["recall"])
            recall_count += 1

        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(signature=signature, requirement_comment=requirement_comment,
                                context_code_in_prompt=context_code_in_prompt)

        if DEBUG:
            print(f"[debug] namespace={task.namespace}", file=sys.stderr)
            print(f"[debug] file={abs_file}", file=sys.stderr)
            print(f"[debug] signature_position={task.signature_position}", file=sys.stderr)
            print("[debug] signature:\n" + preview_text(signature), file=sys.stderr)
            print("[debug] requirement_comment:\n" + preview_text(requirement_comment), file=sys.stderr)
            print("[debug] prompt:\n" + preview_text(prompt), file=sys.stderr)
            if DEBUG_LOG_FULL and debug_log_path:
                sep = "=" * 80
                with open(debug_log_path, "a", encoding="utf-8") as logf:
                    logf.write(f"\n{sep}\n")
                    logf.write(f"Task idx={processed}  namespace={task.namespace}\n")
                    logf.write(f"file={abs_file}\n")
                    logf.write(f"{sep}\n\n")
                    logf.write("--- Full prompt (complete) ---\n\n")
                    logf.write(prompt)
                    logf.write("\n\n")
                    logf.write("--- Context only (context_code_in_prompt) ---\n\n")
                    logf.write(context_code_in_prompt)
                    logf.write("\n\n")
                    logf.flush()

        if GENERATION_FLAG:
            raw_completion = client.generate(prompt)
            extracted_completion = extract_code_from_markdown(raw_completion)
            completion = keep_only_completion(
                extracted_completion,
                signature=signature,
                requirement_comment=requirement_comment,
                requirement_text=task.requirement_text,
            )

            if DEBUG:
                print("[debug] raw_completion:\n" + preview_text(raw_completion), file=sys.stderr)
                print("[debug] final_completion:\n" + preview_text(completion), file=sys.stderr)
                if DEBUG_LOG_FULL and debug_log_path:
                    with open(debug_log_path, "a", encoding="utf-8") as logf:
                        logf.write("--- Raw completion from LLM ---\n\n")
                        logf.write(raw_completion)
                        logf.write("\n\n--- Final completion (after postprocess) ---\n\n")
                        logf.write(completion)
                        logf.write("\n\n")
                        logf.flush()
        else:
            completion = ""

        if GENERATION_FLAG:
            write_jsonl_line(output_jsonl, {
                "namespace": task.namespace,
                "completion": completion,
                "idx": processed,
                "dependency": task.dependency,
                "recall": recall_info,
            })

        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
        if max_tasks is not None and processed >= max_tasks:
            break
    
    mean_recall = (recall_sum / recall_count) if recall_count > 0 else None
    print(
        json.dumps(
            {
                "recall_mean": mean_recall,
                "tasks_with_dependency": recall_count,
                "tasks_without_dependency": recall_none_count,
                "tasks_total": processed,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--filtered_path", default=FILTERED_PATH)
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--methods_csv", default=METHODS_CSV, help="Methods CSV 路径")
    p.add_argument("--enre_json", default=ENRE_JSON, help="ENRE 报告 JSON 路径")
    p.add_argument("--diagnostic_jsonl", default=DIAGNOSTIC_JSONL, help="检索结果 jsonl 路径")
    p.add_argument("--rag_data_source", default=RAG_DATA_SOURCE,
                   choices=["bm25", "unixcoder", "feature", "feature+bm25"],
                   help="RAG 数据源：bm25 / unixcoder / feature / feature+bm25")
    p.add_argument("--output", default=OUTPUT_COMPLETION_PATH)
    p.add_argument("--debug_log", default="", help="DEBUG 且 DEBUG_LOG_FULL 时的全量日志路径，默认 <output>_debug.log")
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None
    generate_completions(
        filtered_path=args.filtered_path,
        source_code_dir=args.source_code_dir,
        methods_csv=args.methods_csv,
        enre_json=args.enre_json,
        diagnostic_jsonl=args.diagnostic_jsonl,
        rag_data_source=args.rag_data_source,
        output_jsonl=args.output,
        debug_log_path_override=(args.debug_log.strip() or None),
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        timeout_s=args.timeout_s,
        max_tasks=max_tasks,
        sleep_s=args.sleep_s,
    )


if __name__ == "__main__":
    main()
