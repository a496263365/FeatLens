"""Generate DevEval completions with LocAgent-retrieved entities as context.

This script intentionally reuses the same prompting/post-processing protocol as
``graph_rag_gen.py`` so that LocAgent can be compared with the other RAG
baselines without changing the generation setup.

The only LocAgent-specific adaptation is context assembly:

* functions and variables are inserted verbatim from ``result.json``;
* class entities that contain the target symbol are converted to skeletons,
  because the raw LocAgent payload may otherwise include the target body;
* the target symbol itself is always excluded.

The original LocAgent result files are never modified.
"""

from __future__ import annotations

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
import concurrent.futures
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import DevEvalTask, parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import get_class_skeleton, resolve_signature
from utils.task_recall import compute_task_recall


DEFAULT_FILTERED_PATH = (
    str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl'))
)
DEFAULT_SOURCE_CODE_DIR = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', ''))
DEFAULT_LOCAGENT_ROOT = (
    str(_package_path('results/retrieval/deveval/locagent/dependency_tasks'))
)
DEFAULT_OUTPUT = (
    str(_package_path('results/locagent_generation_20260930_deepseek_v3_2_completion.jsonl'))
)

PROMPT_TEMPLATE = (
    "You will complete a Python function body based on the requirement and "
    "relevant repository context.\n\n"
    "You will be given some context which may be helpful for you to complete "
    "the function.\n"
    "The context is organized by file. Functions are shown in full when "
    "available; classes are shown as skeletons (signatures + ... for method "
    "bodies).\n\n"
    "Use this context to write a correct and consistent completion following "
    "the signature and requirement.\n\n"
    "**Constraints:**\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n\n"
    "=== Repository context ===\n\n"
    "{{context_code_in_prompt}}\n\n"
    "=== Input code (You should complete!!!): ===\n\n"
    "```Python\n"
    "{{signature}}\n\n"
    "{{requirement_comment}}\n\n"
    "```\n\n"
    "**Completed Code:**\n"
)


@dataclass(frozen=True)
class LocAgentTask:
    task_id: int
    namespace: str
    project_path: str
    completion_path: str
    result_path: str


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_locagent_index(root: str, wanted_namespaces: Iterable[str]) -> Dict[str, LocAgentTask]:
    """Map DevEval namespaces to their LocAgent task and result files."""
    tasks_path = os.path.join(root, "tasks.jsonl")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"LocAgent tasks.jsonl not found: {tasks_path}")

    wanted = set(wanted_namespaces)
    index: Dict[str, LocAgentTask] = {}
    for record in iter_jsonl(tasks_path):
        namespace = record.get("namespace")
        if namespace not in wanted:
            continue

        task_id = record.get("full_run_task_id") or record.get("task_id")
        if not isinstance(task_id, int):
            raise ValueError(f"Invalid LocAgent task id for {namespace}: {task_id!r}")

        task_dir = os.path.join(root, "tasks", f"task_{task_id:06d}", "locagent")
        summary_path = os.path.join(task_dir, "job_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"LocAgent summary not found: {summary_path}")
        summary = _read_json(summary_path)
        attempt_dir = summary.get("attempt_dir")
        if not isinstance(attempt_dir, str) or not attempt_dir:
            raise ValueError(f"LocAgent summary has no attempt_dir: {summary_path}")
        result_path = os.path.join(attempt_dir, "result.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"LocAgent result not found: {result_path}")

        index[namespace] = LocAgentTask(
            task_id=task_id,
            namespace=namespace,
            project_path=str(record.get("project_path", "")),
            completion_path=str(record.get("completion_path", "")),
            result_path=result_path,
        )

    missing = wanted - set(index)
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"LocAgent results missing for {len(missing)} requested tasks: {sample}"
        )
    return index


def _is_target_symbol(entity: Dict[str, Any], task: DevEvalTask) -> bool:
    return str(entity.get("symbol", "")) == task.namespace


def _is_target_ancestor_class(entity: Dict[str, Any], task: DevEvalTask) -> bool:
    symbol = str(entity.get("symbol", ""))
    return (
        entity.get("kind") == "class"
        and symbol
        and (task.namespace == symbol or task.namespace.startswith(symbol + "."))
    )


def _class_skeleton_or_placeholder(
    entity: Dict[str, Any],
    source_code_dir: str,
) -> Tuple[str, str]:
    """Return a safe class representation and how it was produced."""
    symbol = str(entity.get("symbol", ""))
    file_path = str(entity.get("file", ""))
    if symbol and file_path:
        skeleton = get_class_skeleton(source_code_dir, file_path, symbol)
        if skeleton:
            return skeleton, "skeleton"

    name = str(entity.get("name") or symbol or "UnknownClass")
    return f"class {name}:\n    ...", "placeholder"


def build_locagent_context(
    *,
    task: DevEvalTask,
    entities: Iterable[Dict[str, Any]],
    project_root: str,
    class_mode: str = "target-safe",
    max_entity_chars: int = 0,
    max_context_chars: int = 0,
) -> Tuple[str, Dict[str, Any]]:
    """Assemble LocAgent entities into one prompt-ready context string."""
    sections: List[str] = []
    stats: Dict[str, Any] = {
        "returned_entities": 0,
        "included_entities": 0,
        "skipped_target_entities": 0,
        "class_skeleton_entities": 0,
        "class_placeholder_entities": 0,
        "truncated_entities": 0,
        "context_truncated": False,
        "returned_code_chars": 0,
        "included_code_chars": 0,
    }
    seen: set[Tuple[str, str]] = set()
    parts: List[str] = []
    current_chars = 0

    for entity in entities:
        stats["returned_entities"] += 1
        raw_code = entity.get("code") or ""
        stats["returned_code_chars"] += len(raw_code)

        if _is_target_symbol(entity, task):
            stats["skipped_target_entities"] += 1
            continue

        kind = str(entity.get("kind", "entity"))
        symbol = str(entity.get("symbol", ""))
        file_path = str(entity.get("file", ""))
        if kind == "class" and class_mode == "skeleton":
            code, how = _class_skeleton_or_placeholder(entity, project_root)
            stats[
                "class_skeleton_entities" if how == "skeleton" else "class_placeholder_entities"
            ] += 1
        elif _is_target_ancestor_class(entity, task):
            code, how = _class_skeleton_or_placeholder(entity, project_root)
            stats["class_skeleton_entities" if how == "skeleton" else "class_placeholder_entities"] += 1
        else:
            code = str(raw_code)

        if not code.strip():
            continue
        if max_entity_chars and len(code) > max_entity_chars:
            code = code[:max_entity_chars] + "\n# ... entity truncated ..."
            stats["truncated_entities"] += 1

        key = (symbol or file_path, code)
        if key in seen:
            continue
        seen.add(key)

        header = f"--- File: {file_path} ---\n"
        if symbol:
            header += f"### {symbol} ({kind})\n"
        section = header + code.rstrip() + "\n"
        if max_context_chars and current_chars + len(section) > max_context_chars:
            remaining = max_context_chars - current_chars
            if remaining > 0:
                parts.append(section[:remaining] + "\n# ... context truncated ...\n")
                stats["context_truncated"] = True
            break
        parts.append(section)
        current_chars += len(section)
        stats["included_entities"] += 1
        stats["included_code_chars"] += len(code)

    context = "\n".join(parts).strip()
    stats["context_chars"] = len(context)
    return context, stats


def format_requirement_as_comment(requirement_text: str) -> str:
    if not requirement_text:
        return ""
    indent_str = "    "
    delimiter = '"""'
    escaped_delimiter = '\\"\\"\\"'
    lines = [line.replace('"""', escaped_delimiter) for line in requirement_text.splitlines()]
    content = "\n".join(indent_str + line if line else indent_str for line in lines)
    return f"{indent_str}{delimiter}\n{content}\n{indent_str}{delimiter}\n"


def build_prompt(signature: str, requirement_comment: str, context_code_in_prompt: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
        .replace("{{context_code_in_prompt}}", context_code_in_prompt)
    )


def _context_recall_rows(entities: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for entity in entities:
        rows.append(
            {
                "method_signature": entity.get("symbol", ""),
                "func_file": entity.get("file", ""),
                "method_code": entity.get("code", "") or "",
            }
        )
    return rows


_BLOCK_PREFIXES = (
    "if ",
    "elif ",
    "else:",
    "for ",
    "while ",
    "try:",
    "except",
    "finally:",
    "with ",
    "def ",
    "class ",
)


def normalize_completion_indentation(text: str) -> str:
    """Normalize the first-line indentation glitch produced by some models.

    ``pass_k.py`` calls ``textwrap.dedent`` before inserting the completion.
    If a model emits a comment/statement with one leading space but the rest of
    the block with four spaces, dedent leaves three spaces on the following
    lines and the inserted body becomes invalid Python.
    """
    if not text:
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return text
    first = nonempty[0]
    first_line = lines[first]
    first_indent = len(first_line) - len(first_line.lstrip(" "))
    rest = [i for i in nonempty if i != first]
    if not rest:
        return text
    rest_min = min(len(lines[i]) - len(lines[i].lstrip(" ")) for i in rest)

    # A block header legitimately has a smaller indent than its body. A bare
    # comment or simple statement with a smaller indent than every following
    # line is instead the common model formatting glitch. Fix only that first
    # line; leave the remaining indentation for pass_k.py's textwrap.dedent.
    if first_indent < rest_min and not first_line.lstrip().startswith(_BLOCK_PREFIXES):
        lines[first] = (" " * rest_min) + first_line.lstrip(" ")
    return "\n".join(lines)


def generate_completions(
    *,
    filtered_path: str,
    source_code_dir: str,
    locagent_root: str,
    output_jsonl: str,
    debug_log_path_override: Optional[str],
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    timeout_s: float,
    max_tasks: Optional[int],
    sleep_s: float,
    max_entity_chars: int,
    max_context_chars: int,
    class_mode: str,
    workers: int,
    debug: bool,
    debug_log_full: bool,
) -> None:
    wanted_namespaces: set[str] = set()
    for record in iter_jsonl(filtered_path):
        namespace = record.get("namespace")
        completion_path = record.get("completion_path")
        if not isinstance(namespace, str) or not isinstance(completion_path, str):
            raise ValueError("Every DevEval record must contain namespace and completion_path")
        wanted_namespaces.add(namespace)

    locagent_index = load_locagent_index(locagent_root, wanted_namespaces)
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
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

    debug_log_path = None
    if debug and debug_log_full:
        debug_log_path = debug_log_path_override or (
            os.path.splitext(output_jsonl)[0] + "_debug.log"
        )
        with open(debug_log_path, "w", encoding="utf-8"):
            pass

    records: List[Tuple[int, Dict[str, Any]]] = []
    for idx, record in enumerate(iter_jsonl(filtered_path)):
        records.append((idx, record))
        if max_tasks is not None and len(records) >= max_tasks:
            break

    def process_one(item: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, record = item
        task = parse_task(record)
        locagent_task = locagent_index[task.namespace]
        result = _read_json(locagent_task.result_path)
        final_locations = result.get("final_locations") or {}
        entities = final_locations.get("entities") or []

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )
        context_code_in_prompt, context_stats = build_locagent_context(
            task=task,
            entities=entities,
            project_root=os.path.join(source_code_dir, locagent_task.project_path),
            class_mode=class_mode,
            max_entity_chars=max_entity_chars,
            max_context_chars=max_context_chars,
        )
        recall_info = compute_task_recall(task.dependency, _context_recall_rows(entities))
        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(
            signature=signature,
            requirement_comment=requirement_comment,
            context_code_in_prompt=context_code_in_prompt,
        )
        last_err: Optional[Exception] = None
        raw_completion = ""
        for attempt in range(4):
            try:
                raw_completion = client.generate(prompt)
                break
            except Exception as e:
                last_err = e
                if attempt == 3:
                    raise
                time.sleep(min(30.0, 5.0 * (2 ** attempt)))
        extracted_completion = extract_code_from_markdown(raw_completion)
        completion = keep_only_completion(
            extracted_completion,
            signature=signature,
            requirement_comment=requirement_comment,
            requirement_text=task.requirement_text,
        )
        completion = normalize_completion_indentation(completion)
        if sleep_s > 0:
            time.sleep(sleep_s)
        return {
            "idx": idx,
            "namespace": task.namespace,
            "completion": completion,
            "dependency": task.dependency,
            "recall": recall_info,
            "locagent_task_id": locagent_task.task_id,
            "locagent_context_stats": context_stats,
            "debug_abs_file": abs_file,
            "debug_prompt": prompt,
            "debug_context": context_code_in_prompt,
            "debug_raw_completion": raw_completion,
        }

    processed = 0
    recall_sum = 0.0
    recall_count = 0
    recall_none_count = 0
    skipped_target_total = 0
    skeleton_total = 0

    def handle_result(result: Dict[str, Any]) -> None:
        nonlocal processed, recall_sum, recall_count, recall_none_count
        nonlocal skipped_target_total, skeleton_total
        recall_info = result["recall"]
        context_stats = result["locagent_context_stats"]
        if recall_info["recall"] is None:
            recall_none_count += 1
        else:
            recall_sum += float(recall_info["recall"])
            recall_count += 1
        skipped_target_total += context_stats["skipped_target_entities"]
        skeleton_total += context_stats["class_skeleton_entities"]

        if debug:
            print(f"[debug] namespace={result['namespace']}", file=sys.stderr)
            print(f"[debug] file={result['debug_abs_file']}", file=sys.stderr)
            print(
                f"[debug] context_stats={json.dumps(context_stats, ensure_ascii=False)}",
                file=sys.stderr,
            )
            print("[debug] prompt:\n" + preview_text(result["debug_prompt"]), file=sys.stderr)
        if debug_log_full and debug_log_path:
            with open(debug_log_path, "a", encoding="utf-8") as logf:
                sep = "=" * 80
                logf.write(f"\n{sep}\n")
                logf.write(
                    f"Task idx={result['idx']}  namespace={result['namespace']}\n"
                )
                logf.write(f"LocAgent task_id={result['locagent_task_id']}\n")
                logf.write(f"file={result['debug_abs_file']}\n")
                logf.write(f"{json.dumps(context_stats, ensure_ascii=False)}\n")
                logf.write(f"{sep}\n\n")
                logf.write("--- Full prompt (complete) ---\n\n")
                logf.write(result["debug_prompt"])
                logf.write("\n\n--- Context only ---\n\n")
                logf.write(result["debug_context"])
                logf.write("\n\n")
                logf.write("--- Raw completion from LLM ---\n\n")
                logf.write(result["debug_raw_completion"])
                logf.write("\n\n--- Final completion (after postprocess) ---\n\n")
                logf.write(result["completion"])
                logf.write("\n\n")
                logf.flush()

        write_jsonl_line(
            output_jsonl,
            {
                "namespace": result["namespace"],
                "completion": result["completion"],
                "idx": result["idx"],
                "dependency": result["dependency"],
                "recall": result["recall"],
                "locagent_task_id": result["locagent_task_id"],
                "locagent_context_stats": context_stats,
            },
        )
        processed += 1

    if workers <= 1:
        for item in records:
            handle_result(process_one(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {executor.submit(process_one, item): item for item in records}
            for future in concurrent.futures.as_completed(future_to_item):
                handle_result(future.result())

    mean_recall = (recall_sum / recall_count) if recall_count > 0 else None
    print(
        json.dumps(
            {
                "model": model,
                "recall_mean": mean_recall,
                "tasks_with_dependency": recall_count,
                "tasks_without_dependency": recall_none_count,
                "tasks_total": processed,
                "skipped_target_entities": skipped_target_total,
                "class_skeleton_entities": skeleton_total,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--filtered_path", default=DEFAULT_FILTERED_PATH)
    p.add_argument("--source_code_dir", default=DEFAULT_SOURCE_CODE_DIR)
    p.add_argument("--locagent_root", default=DEFAULT_LOCAGENT_ROOT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--debug_log", default="")
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default="openai")
    p.add_argument("--model", default="deepseek-v3.2")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=180.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--max_entity_chars", type=int, default=0)
    p.add_argument("--max_context_chars", type=int, default=0)
    p.add_argument("--class_mode", choices=["target-safe", "skeleton"], default="target-safe")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    generate_completions(
        filtered_path=args.filtered_path,
        source_code_dir=args.source_code_dir,
        locagent_root=args.locagent_root,
        output_jsonl=args.output,
        debug_log_path_override=(args.debug_log.strip() or None),
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
        timeout_s=args.timeout_s,
        max_tasks=args.max_tasks if args.max_tasks > 0 else None,
        sleep_s=args.sleep_s,
        max_entity_chars=max(0, args.max_entity_chars),
        max_context_chars=max(0, args.max_context_chars),
        class_mode=args.class_mode,
        workers=max(1, args.workers),
        debug=not args.quiet,
        debug_log_full=True,
    )


if __name__ == "__main__":
    main()
