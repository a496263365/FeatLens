
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
from typing import Any, Dict, List

_SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import pandas as pd

from build_graph import build_graph, METHODS_CSV, ENRE_JSON, FILTERED_PATH, DIAGNOSTIC_JSONL, OUTPUT_GRAPH_PATH
from utils.excel_project_list import (
    normalize_excel_project_columns,
    resolve_enre_json_cell,
)


EXCEL_PATH = str(_package_path('data/evocodebench/projects_5.xlsx'))
DEFAULT_BASE_SEARCH_OUT = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_BASE_ENRE = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_OUTPUT_CSV = str(_package_path('results/retrieval/evocodebench/featlens_and_rag/build_graph_batch_metrics.csv'))


def run_one_project(
    project_name: str,
    enre_json: str,
    base_search_out: str,
) -> Dict[str, Any]:
    project_base = os.path.join(base_search_out, project_name)
    methods_csv = os.path.join(project_base, "methods.csv")
    filtered_path = os.path.join(project_base, "filtered.jsonl")
    diagnostic_jsonl = os.path.join(project_base, "diagnostic_***feature.jsonl")
    output_graph_path = os.path.join(project_base, "graph_results_***all")

    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return {}
    if not os.path.exists(filtered_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_path}")
        return {}
    if not os.path.exists(diagnostic_jsonl):
        print(f"[skip] {project_name}: diagnostic feature jsonl not found at {diagnostic_jsonl}")
        return {}

    print(f"[run] {project_name} (build_graph)")
    metrics = build_graph(
        methods_csv=methods_csv,
        enre_json=enre_json,
        filtered_path=filtered_path,
        diagnostic_jsonl=diagnostic_jsonl,
        output_graph_path=output_graph_path,
    )
    metrics["project_name"] = project_name
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch build graphs for multiple projects from Excel project list"
    )
    p.add_argument(
        "--excel_path",
        default=EXCEL_PATH,
        help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径",
    )
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="methods/filtered/diagnostic 等所在根目录",
    )
    p.add_argument(
        "--base_enre",
        default=DEFAULT_BASE_ENRE,
        help="ENRE 根目录（未填 enre_json 时推导用）",
    )
    p.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help="批量 graph 构建结果（逐项目 + 总体）输出 CSV 路径",
    )
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = normalize_excel_project_columns(df)

    if "project_name" not in df.columns:
        raise SystemExit(
            "Excel 需包含「项目名称」列（或 project_name），用于定位 base_search_out 下的子目录"
        )

    all_rows: List[Dict[str, Any]] = []
    per_project_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        project_name = str(row["project_name"]).strip()
        if not project_name:
            continue
        enre_json = resolve_enre_json_cell(row, project_name, args.base_enre)

        metrics = run_one_project(
            project_name=project_name,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
        )
        if not metrics:
            continue

        per_project_rows.append(
            {
                "project_name": project_name,
                "total_tasks": metrics.get("total_tasks", 0),
                "same_file_nonzero_tasks": metrics.get("same_file_nonzero_tasks", 0),
                "same_feature_nonzero_tasks": metrics.get("same_feature_nonzero_tasks", 0),
                "similar_method_nonzero_tasks": metrics.get("similar_method_nonzero_tasks", 0),
            }
        )

    all_rows.extend(per_project_rows)

    # Compute overall aggregate results
    if per_project_rows:
        total_tasks_sum = sum(r["total_tasks"] for r in per_project_rows)
        same_file_sum = sum(r["same_file_nonzero_tasks"] for r in per_project_rows)
        same_feature_sum = sum(r["same_feature_nonzero_tasks"] for r in per_project_rows)
        similar_method_sum = sum(r["similar_method_nonzero_tasks"] for r in per_project_rows)

        all_rows.append(
            {
                "project_name": "ALL",
                "total_tasks": total_tasks_sum,
                "same_file_nonzero_tasks": same_file_sum,
                "same_feature_nonzero_tasks": same_feature_sum,
                "similar_method_nonzero_tasks": similar_method_sum,
            }
        )

    if all_rows:
        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved batch graph metrics to {args.output_csv}")
    else:
        print("No metrics produced; nothing to save.")


if __name__ == "__main__":
    main()

