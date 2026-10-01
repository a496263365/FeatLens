
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
import os
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import resolve_signature


SOURCE_CODE_DIR = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', ''))
FILTERED_PATH = str(_package_path('data/deveval/projects/Internet/boto/filtered.jsonl'))
OUTPUT_COMPLETION_PATH = str(_package_path('results/generation/deveval/no_context/deepseek_v3_2/completion.jsonl'))
MODEL_NAME = "deepseek-v3.2"
MODEL_BACKEND_CHOICE = "openai"

DEBUG = True  # Whether to print debug information to the console
DEBUG_LOG_FULL = True  # When DEBUG is True, also write all content, including the complete prompt and generated code, to the log file

PROMPT_TEMPLATE = (
    "Please complete the function in the given Python code.\n\n"
    "Constraints:\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n\n"
    "Input Code:\n"
    "```Python\n"
    "{{signature}}\n\n"
    "{{requirement_comment}}\n\n"
    "```\n\n"
    "Completed Code:\n"
)


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


def build_prompt(signature: str, requirement_comment: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
    )


def generate_completions(
    *,
    filtered_path: str,
    source_code_dir: str,
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

    # In DEBUG mode, write the complete prompt and raw/final completion to the log for inspection
    debug_log_path = None
    if DEBUG and DEBUG_LOG_FULL:
        debug_log_path = debug_log_path_override or (os.path.splitext(output_jsonl)[0] + "_debug.log")
        with open(debug_log_path, "w", encoding="utf-8") as _:
            pass

    processed = 0
    for record in iter_jsonl(filtered_path):
        task = parse_task(record)

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )
        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(signature=signature, requirement_comment=requirement_comment)

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
                    logf.flush()

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
            print("[debug] extracted_completion:\n" + preview_text(extracted_completion), file=sys.stderr)
            print("[debug] final_completion:\n" + preview_text(completion), file=sys.stderr)
            if DEBUG_LOG_FULL and debug_log_path:
                with open(debug_log_path, "a", encoding="utf-8") as logf:
                    logf.write("--- Raw completion from LLM ---\n\n")
                    logf.write(raw_completion)
                    logf.write("\n\n--- Final completion (after postprocess) ---\n\n")
                    logf.write(completion)
                    logf.write("\n\n")
                    logf.flush()

        write_jsonl_line(output_jsonl, {
            "namespace": task.namespace,
            "completion": completion,
            "idx": processed
        })

        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
        if max_tasks is not None and processed >= max_tasks:
            break


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--filtered_path", default=FILTERED_PATH)
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--output", default=OUTPUT_COMPLETION_PATH)
    p.add_argument("--debug_log", default="", help="当 DEBUG 且 DEBUG_LOG_FULL 时：全量日志路径；默认 <output>_debug.log")
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
