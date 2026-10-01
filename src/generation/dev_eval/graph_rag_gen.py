
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
import html
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
from numpy import append

from utils.global_var_extract import get_used_globals_in_file
from llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import DevEvalTask, parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import resolve_signature, get_class_skeleton
from utils.task_recall import compute_task_recall, load_enre_elements


SOURCE_CODE_DIR = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', ''))
FILTERED_PATH = str(_package_path('data/deveval/projects/Security/diffprivlib/filtered.jsonl'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/Security/diffprivlib/report-enre.json'))
GRAPH_DIR_PATH = str(_package_path('results/retrieval/deveval/featlens_and_rag/Security/diffprivlib/graphs/pagerank_top_15'))
OUTPUT_COMPLETION_PATH = str(_package_path('results/generation/deveval/featlens/deepseek_v3_2/completion.jsonl'))
# Project root directory corresponding to the RepoSummary output
GRAPH_PROJECT_PATH = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', 'Security/diffprivlib'))

# Large language model used for code generation
MODEL_NAME = "deepseek-v3.2"
MODEL_BACKEND_CHOICE = "openai"
DEBUG = True  # Whether to print debug information to the console
DEBUG_LOG_FULL = True  # When DEBUG is True, also write the complete prompt and other full content to the log to inspect prompt construction
GENERATION_FLAG = True  # Whether to perform code generation; defaults to True, set to False to compute context recall only

PROMPT_TEMPLATE = (
    "You will complete a Python function body based on the requirement and relevant repository context.\n\n"
    "You will be given some context which may be helpful for you to complete the function.\n"
    "The context is organized by file. For each file you will see:\n"
    "- **Important module-level variable and class names** used by the code below.\n"
    "- **Relations** between entities (e.g. A Call B, C Define D) to understand repository-level code structure.\n"
    "- **Code**: functions are shown in full; classes are shown as skeletons (signatures + ... for method bsodies).\n\n"
    "Use this context to write a correct and consistent completion following the signature and requirement.\n\n"
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


def _unescape(s):
    if s is None or (isinstance(s, float) and (s != s)):
        return ""
    if isinstance(s, str):
        return html.unescape(s)
    return str(s)


def _edge_kind(attrs: Dict) -> str:
    """GML edges may use 'kind' or 'type' for relation type."""
    k = attrs.get("kind") or attrs.get("type")
    return _unescape(k) if k is not None else ""


def load_graph_and_group_by_file(
    graph_gml_path: str,
    project_path: str,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Tuple[str, str, str]]], Dict[str, Set[str]]]:
    """
    Load GML graph and return:
    1) file_to_nodes: func_file -> list of node dicts (sig, category, method_signature, func_file, method_code, etc.)
    2) file_to_relations: func_file -> list of (sig_src, kind, sig_tgt)
    3) file_to_used_globals: func_file -> set of used global names (from static analysis)
    """
    if not os.path.exists(graph_gml_path):
        return defaultdict(list), defaultdict(list), defaultdict(set)

    G = nx.read_gml(graph_gml_path)
    id_to_node = {}
    # Use method_signature as unique identifier (sig can duplicate for overloads e.g. two write(...))
    node_method_sigs = {}
    file_to_nodes = defaultdict(list)

    for node_id, attrs in G.nodes(data=True):
        sig = _unescape(attrs.get("sig"))
        category = _unescape(attrs.get("category")) or "Function"
        method_sig = _unescape(attrs.get("method_signature")) or sig
        func_file = _unescape(attrs.get("func_file"))
        method_code = _unescape(attrs.get("method_code"))
        if not func_file:
            continue
        id_to_node[node_id] = {
            "sig": sig,
            "category": category,
            "method_signature": method_sig,
            "func_file": func_file,
            "method_code": method_code or "",
        }
        node_method_sigs[node_id] = method_sig
        file_to_nodes[func_file].append(id_to_node[node_id])

    # Relations
    file_to_relations = defaultdict(list)
    for u, v, attrs in G.edges(data=True):
        kind = _edge_kind(attrs)
        if not kind:
            continue
        ms_u = node_method_sigs.get(u)
        ms_v = node_method_sigs.get(v)
        if ms_u is None or ms_v is None:
            continue
        for f in (id_to_node.get(u, {}).get("func_file"), id_to_node.get(v, {}).get("func_file")):
            if f:
                file_to_relations[f].append((ms_u, kind, ms_v))

    # Deduplicate relations per file
    for f in file_to_relations:
        file_to_relations[f] = list(dict.fromkeys(file_to_relations[f]))

    # Used globals per file (tree-sitter)
    project_root = project_path
    file_to_used_globals = defaultdict(set)
    for func_file, nodes in file_to_nodes.items():
        node_keys = []
        for n in nodes:
            cat = n.get("category") or "Function"
            sig = n.get("sig") or ""
            short = sig.split(".")[-1] if sig else ""
            method_sig = n.get("method_signature") or ""
            if short:
                node_keys.append((cat, short, method_sig))
        if node_keys:
            used = get_used_globals_in_file(project_root, func_file, node_keys)
            file_to_used_globals[func_file] = used

    return dict(file_to_nodes), dict(file_to_relations), dict(file_to_used_globals)


def get_node_display_code(
    node: Dict,
    project_path: str,
) -> str:
    """Return code to show: full for Function, skeleton for Class."""
    category = (node.get("category") or "Function").strip()
    func_file = node.get("func_file") or ""
    method_code = (node.get("method_code") or "").strip()
    sig = (node.get("sig") or "").strip()
    project_root = project_path

    if category == "Class":
        skeleton = get_class_skeleton(project_root, func_file, sig)
        if skeleton:
            return skeleton
        return method_code or f"# Class: {sig}"
    return method_code or f"# {node.get('method_signature', sig)}"


def assemble_graph_context_into_prompt(
    file_to_nodes: Dict[str, List[Dict]],
    file_to_relations: Dict[str, List[Tuple[str, str, str]]],
    file_to_used_globals: Dict[str, Set[str]],
    project_path: str
) -> str:
    """Build the context string: by file, with globals, relations, and code."""
    lines = []
    for func_file in sorted(file_to_nodes.keys()):
        nodes = file_to_nodes[func_file]
        rels = file_to_relations.get(func_file, [])
        used_globals = file_to_used_globals.get(func_file, set())

        lines.append(f"--- File: {func_file} ---")
        if used_globals:
            lines.append("**Important module-level variable and class names** used here:")
            lines.append(", ".join(sorted(used_globals)))
        if rels:
            lines.append("**Important Relations** between code entities:")
            for s, k, t in rels:
                lines.append(f"  {s} {k} {t}")
        lines.append("**Important Code Context**:")
        for node in nodes:
            display = get_node_display_code(node, project_path)
            if display:
                lines.append(display)
                lines.append("")
        lines.append("")
    return "\n".join(lines).strip()


def format_requirement_as_comment(requirement_text: str) -> str:
    if not requirement_text:
        return ""
    lines = requirement_text.splitlines()
    indent_str = "    "  # Use four spaces per indentation level
    delimiter = '"""'  # Represent multiline comments with triple quotes
    escaped = '\\"\\"\\"'
    lines = [line.replace(delimiter, escaped) for line in lines]
    content = "\n".join(indent_str + line if line else indent_str for line in lines)
    return f"{indent_str}{delimiter}\n{content}\n{indent_str}{delimiter}\n"


def build_prompt(signature: str, requirement_comment: str, context_code_in_prompt: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
        .replace("{{context_code_in_prompt}}", context_code_in_prompt)
    )


def graph_context_to_flat_list(file_to_nodes: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """Convert file->nodes to a flat list of context dicts (for compute_task_recall)."""
    out = []
    for _f, nodes in sorted(file_to_nodes.items()):
        for n in nodes:
            out.append({
                "method_signature": n.get("sig"),
                "func_file": n.get("func_file"),
                "method_code": n.get("method_code") or "",
            })
    return out


def generate_completions(
    *,
    filtered_path: str,
    source_code_dir: str,
    graph_dir_path: str,
    project_path: str,
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

    processed = 0
    recall_sum = 0.0
    recall_count = 0
    recall_none_count = 0

    # In DEBUG mode, write the complete prompt, context, and completion to the log to inspect construction
    debug_log_path = None
    if DEBUG and DEBUG_LOG_FULL:
        debug_log_path = debug_log_path_override or (os.path.splitext(output_jsonl)[0] + "_debug.log")
        # Clear or create the log file and overwrite it for this run
        with open(debug_log_path, "w", encoding="utf-8") as _:
            pass

    for record in iter_jsonl(filtered_path):
        task = parse_task(record)
        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )

        # Obtain the code-search results for this task in graph form
        graph_gml_path = os.path.join(graph_dir_path, f"task_{processed + 1}_rank.gml")
        file_to_nodes, file_to_relations, file_to_used_globals = load_graph_and_group_by_file(
            graph_gml_path, project_path
        )

        context_code_in_prompt = assemble_graph_context_into_prompt(
            file_to_nodes,
            file_to_relations,
            file_to_used_globals,
            project_path
        )

        # Convert the graph context to a flat list for computing context recall for this task
        flat_list = graph_context_to_flat_list(file_to_nodes)
        recall_info = compute_task_recall(task.dependency, flat_list)
        if recall_info["recall"] is None:
            recall_none_count += 1
        else:
            recall_sum += float(recall_info["recall"])
            recall_count += 1

        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(
            signature=signature,
            requirement_comment=requirement_comment,
            context_code_in_prompt=context_code_in_prompt,
        )

        if DEBUG:
            print(f"[debug] namespace={task.namespace}", file=sys.stderr)
            print(f"[debug] file={abs_file}", file=sys.stderr)
            print("[debug] prompt (preview):\n" + preview_text(prompt), file=sys.stderr)
            # Write all content to the log file to inspect prompt construction
            if DEBUG_LOG_FULL and debug_log_path:
                with open(debug_log_path, "a", encoding="utf-8") as logf:
                    sep = "=" * 80
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
    p.add_argument("--graph_dir_path", default=GRAPH_DIR_PATH, help="Directory containing task_1_rank.gml, ...")
    p.add_argument("--project_path", default=GRAPH_PROJECT_PATH, help="Project root for func_file in GML")
    p.add_argument("--output", default=OUTPUT_COMPLETION_PATH)
    p.add_argument("--debug_log", default="", help="When DEBUG and DEBUG_LOG_FULL: path for full prompt log; default is <output>_debug.log")
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
    project_path = args.project_path
    generate_completions(
        filtered_path=args.filtered_path,
        source_code_dir=args.source_code_dir,
        graph_dir_path=args.graph_dir_path,
        project_path=project_path,
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
    load_enre_elements(ENRE_JSON)
    main()
