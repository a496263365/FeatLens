"""
批量运行 base_rag_gen：从 Excel 读取项目列表，按项目依次执行 load_methods_info + load_enre_elements + generate_completions。

Excel 表格要求（与 batch_run_graph_rag 相同格式可复用）：
  - 至少两列：项目名称（project_name / 项目名称）、项目根目录（project_root / 项目根目录）
  - 可选列：enre_json / ENRE路径，不填则按 {base_enre}/{项目名}/{项目名}-report-enre.json 推导
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
import os
import sys

_DEV_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.normpath(os.path.join(_DEV_EVAL_DIR, "..", ".."))
for _p in (_SRC_ROOT, _DEV_EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

from base_rag_gen import generate_completions
from llm_clients import BackendName
from utils.excel_project_list import (
    normalize_excel_project_columns,
    resolve_enre_json_cell,
)
from utils.task_recall import clear_enre_elements, load_enre_elements


EXCEL_PATH = str(_package_path('src/generation/dev_eval/project_to_run/0311_5projects.xlsx'))
SOURCE_CODE_DIR = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', ''))
MODEL_NAME = "deepseek-v3.2"
MODEL_BACKEND_CHOICE = "openai"
# RAG data source; supported values include "bm25", "unixcoder", "feature", and "feature+bm25"
RAG_DATA_SOURCE = "bm25"
# Diagnostic file name corresponding to the RAG source, such as unixcoder -> diagnostic_unixcoder_code.jsonl
DEFAULT_DIAGNOSTIC_FILENAME = "diagnostic_bm25_code.jsonl"
COMPLETION_FILENAME = "bm25_rag_completion.jsonl"  # It must correspond to the RAG source.

DEFAULT_BASE_SEARCH_OUT = str(_package_path('results/retrieval/deveval/featlens_and_rag'))
DEFAULT_BASE_COMPLETION_OUT = str(_package_path('results/devEvalCompletionOut'))
DEFAULT_BASE_ENRE = str(_package_path('results/retrieval/deveval/featlens_and_rag'))
SUBFOLDER = "0303_full"


def run_one_project(
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    base_completion_out: str,
    source_code_dir: str,
    diagnostic_filename: str,
    rag_data_source: str,
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    timeout_s: float,
    max_tasks: int | None,
    sleep_s: float,
    debug_log_override: str = "",
) -> None:
    """对单个项目：清空 ENRE，再调用 generate_completions（内部会 load_methods_info + load_enre_elements）。"""
    filtered_path = os.path.join(base_search_out, project_name, "filtered.jsonl")
    methods_csv = os.path.join(base_search_out, project_name, "methods.csv")
    diagnostic_jsonl = os.path.join(base_search_out, project_name, diagnostic_filename)
    output_jsonl = os.path.join(base_completion_out, project_name, SUBFOLDER, COMPLETION_FILENAME)

    if not os.path.exists(filtered_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_path}", file=sys.stderr)
        return
    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}", file=sys.stderr)
        return
    if not os.path.exists(diagnostic_jsonl):
        print(f"[skip] {project_name}: diagnostic not found at {diagnostic_jsonl}", file=sys.stderr)
        return

    clear_enre_elements()
    # generate_completions internally calls load_methods_info(methods_csv) and load_enre_elements(enre_json)
    debug_log = debug_log_override.strip() or None
    if not debug_log:
        debug_log = os.path.splitext(output_jsonl)[0] + "_debug.log"

    print(f"[run] {project_name} rag_data_source={rag_data_source}", file=sys.stderr)
    generate_completions(
        filtered_path=filtered_path,
        source_code_dir=source_code_dir,
        methods_csv=methods_csv,
        enre_json=enre_json,
        diagnostic_jsonl=diagnostic_jsonl,
        rag_data_source=rag_data_source,
        output_jsonl=output_jsonl,
        debug_log_path_override=debug_log,
        backend=backend,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        max_tasks=max_tasks,
        sleep_s=sleep_s,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Batch run base_rag_gen from Excel project list")
    p.add_argument("--excel_path", default=EXCEL_PATH, help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径")
    p.add_argument("--base_search_out", default=DEFAULT_BASE_SEARCH_OUT, help="filtered/methods/diagnostic 所在根目录")
    p.add_argument("--base_completion_out", default=DEFAULT_BASE_COMPLETION_OUT, help="Completion 输出根目录")
    p.add_argument("--base_enre", default=DEFAULT_BASE_ENRE, help="ENRE 根目录（未填 enre_json 时推导用）")
    p.add_argument("--diagnostic_filename", default=DEFAULT_DIAGNOSTIC_FILENAME,
                   help="诊断 jsonl 文件名，如 diagnostic_unixcoder_code.jsonl")
    p.add_argument("--rag_data_source", default=RAG_DATA_SOURCE,
                   choices=["bm25", "unixcoder", "feature", "feature+bm25"])
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = normalize_excel_project_columns(df)

    if "project_name" not in df.columns or "project_root" not in df.columns:
        print("Error: Excel 需包含「项目名称」与「项目根目录」列（或 project_name / project_root）", file=sys.stderr)
        sys.exit(1)

    for _, row in df.iterrows():
        project_name = str(row["project_name"]).strip()
        project_root = str(row["project_root"]).strip()
        if not project_name or not project_root:
            continue
        enre_json = resolve_enre_json_cell(row, project_name, args.base_enre)

        run_one_project(
            project_name=project_name,
            project_root=project_root,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
            base_completion_out=args.base_completion_out,
            source_code_dir=args.source_code_dir,
            diagnostic_filename=args.diagnostic_filename,
            rag_data_source=args.rag_data_source,
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
