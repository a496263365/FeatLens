
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import pandas as pd

from expand_and_rank_graph_exclude_TM_last import run_project

_SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
from utils.excel_project_list import (
    normalize_excel_project_columns,
    resolve_enre_json_cell,
)
from utils.project_rel_from_root import ProjectRelMode, project_rel_from_project_root


EXCEL_PATH = str(_package_path('data/evocodebench/projects_5.xlsx'))
SOURCE_CODE_DIR = str(_external_path('FEATLENS_EVOCODEBENCH_SOURCE_ROOT', 'external/evocodebench/source_code', ''))
# Use two path segments (e.g., System/mrjob for DevEval) or one segment (e.g., litdata for EvoCodeBench)
PROJECT_REL_MODE: ProjectRelMode = "one_segment"
DEFAULT_BASE_SEARCH_OUT = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_BASE_ENRE = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
# Which embedding backend to use for personalization scores.
#   - "unixcoder": default, UniXcoder-based embeddings
#   - "bge-code": use BAAI/bge-code-v1 via sentence-transformers
EMBEDDING_BACKEND_KIND = "bge-code"


def run_one_project(
    *,
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    source_code_dir: str,
    embedding_backend_kind: str,
    project_rel_mode: ProjectRelMode,
) -> None:
    project_base = os.path.join(base_search_out, project_name)
    methods_csv = os.path.join(project_base, "methods.csv")
    filtered_path = os.path.join(project_base, "filtered.jsonl")
    output_graph_path = os.path.join(project_base, "graph_results_***all")

    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return
    if not os.path.exists(filtered_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_path}")
        return
    if not os.path.exists(output_graph_path):
        print(f"[skip] {project_name}: graph dir not found at {output_graph_path}")
        return

    project_rel = project_rel_from_project_root(
        project_root, source_code_dir, mode=project_rel_mode
    )
    project_path = os.path.join(source_code_dir, project_rel)

    print(f"[run] {project_name} PROJECT_PATH={project_rel} (expand+rank)")
    run_project(
        methods_csv=methods_csv,
        enre_json=enre_json,
        filtered_path=filtered_path,
        output_graph_path=output_graph_path,
        project_path=project_path,
        embedding_backend_kind=embedding_backend_kind,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch expand and rank graphs for multiple projects from Excel project list"
    )
    p.add_argument(
        "--excel_path",
        default=EXCEL_PATH,
        help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径",
    )
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="methods/filtered/graph_results_***all 等所在根目录",
    )
    p.add_argument(
        "--base_enre",
        default=DEFAULT_BASE_ENRE,
        help="ENRE 根目录（未填 enre_json 时推导用）",
    )
    p.add_argument(
        "--source_code_dir",
        default=SOURCE_CODE_DIR,
        help="DevEval 源代码根目录（用于从 project_root 截取 PROJECT_PATH）",
    )
    p.add_argument(
        "--project_rel_mode",
        choices=["two_segments", "one_segment"],
        default=None,
        help="相对 source_code_dir 截取几段；默认用脚本顶部 PROJECT_REL_MODE",
    )
    p.add_argument(
        "--embedding_backend_kind",
        default=EMBEDDING_BACKEND_KIND,
        choices=["unixcoder", "bge-code"],
    )
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()
    project_rel_mode: ProjectRelMode = (
        args.project_rel_mode if args.project_rel_mode is not None else PROJECT_REL_MODE
    )

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = normalize_excel_project_columns(df)

    if "project_name" not in df.columns or "project_root" not in df.columns:
        raise SystemExit(
            "Excel 需包含「项目名称」与「项目根目录」列（或 project_name / project_root）"
        )

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
            source_code_dir=args.source_code_dir,
            embedding_backend_kind=args.embedding_backend_kind,
            project_rel_mode=project_rel_mode,
        )


if __name__ == "__main__":
    main()

