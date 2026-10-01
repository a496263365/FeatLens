
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

from bm25_search import analyze_project
from utils.excel_project_list import (
    normalize_excel_project_columns,
    resolve_enre_json_cell,
)
from utils.project_rel_from_root import ProjectRelMode, project_rel_from_project_root


EXCEL_PATH = str(_package_path('data/evocodebench/projects_5.xlsx'))
SOURCE_CODE_DIR = str(_external_path('FEATLENS_EVOCODEBENCH_SOURCE_ROOT', 'external/evocodebench/source_code', ''))
# Match project_path in data.jsonl: two segments (e.g., System/mrjob for DevEval) or one segment (e.g., litdata for EvoCodeBench)
PROJECT_REL_MODE: ProjectRelMode = "one_segment"
DEFAULT_BASE_SEARCH_OUT = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_BASE_ENRE = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_OUTPUT_CSV = str(_package_path('results/retrieval/evocodebench/featlens_and_rag/bm25_batch_metrics.csv'))
DATA_JSONL = str(_package_path('data/evocodebench/tasks_all_123.jsonl'))


def run_one_project(
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    source_code_dir: str,
    data_jsonl: str,
    *,
    project_rel_mode: ProjectRelMode,
) -> Dict[str, Any]:
    project_dir = project_rel_from_project_root(
        project_root, source_code_dir, mode=project_rel_mode
    )

    project_base = os.path.join(base_search_out, project_name)
    methods_csv = os.path.join(project_base, "methods.csv")
    methods_desc_csv = os.path.join(project_base, "methods_with_desc.csv")
    filtered_path = os.path.join(project_base, "filtered.jsonl")
    refined_queries_cache_path = os.path.join(project_base, "refined_queries.json")

    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return {}
    if not os.path.exists(methods_desc_csv):
        print(f"[skip] {project_name}: methods_with_desc.csv not found at {methods_desc_csv}")
        return {}

    print(f"[run] {project_name} PROJECT_DIR={project_dir}")
    metrics = analyze_project(
        project_dir=project_dir,
        methods_csv=methods_csv,
        methods_desc_csv=methods_desc_csv,
        filtered_path=filtered_path,
        refined_queries_cache_path=refined_queries_cache_path,
        enre_json=enre_json,
        data_jsonl=data_jsonl,
    )
    metrics["project_name"] = project_name
    return metrics


def _append_rows_for_type(
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    project_name: str,
    metric_type: str,
) -> None:
    type_metrics = metrics.get(metric_type, {})
    num_examples = metrics.get("num_examples", 0)
    for k, v in type_metrics.items():
        rows.append(
            {
                "project_name": project_name,
                "project_dir": metrics.get("project_dir", ""),
                "metric_type": metric_type,
                "k": int(k),
                "P": v.get("P", 0.0),
                "R": v.get("R", 0.0),
                "F1": v.get("F1", 0.0),
                "match": v.get("match", 0),
                "pred": v.get("pred", 0),
                "gt": v.get("top_gt", 0),
                "tasks": num_examples,
            }
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Batch run BM25 search experiments from Excel project list")
    p.add_argument("--excel_path", default=EXCEL_PATH, help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径")
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="methods/methods_with_desc/filtered 等所在根目录",
    )
    p.add_argument(
        "--base_enre",
        default=DEFAULT_BASE_ENRE,
        help="ENRE 根目录（未填 enre_json 时推导用）",
    )
    p.add_argument(
        "--source_code_dir",
        default=SOURCE_CODE_DIR,
        help="DevEval 源代码根目录（用于从 project_root 截取 PROJECT_DIR）",
    )
    p.add_argument(
        "--project_rel_mode",
        choices=["two_segments", "one_segment"],
        default=None,
        help="相对 source_code_dir 截取几段：two_segments=两段，one_segment=一段；默认用脚本顶部 PROJECT_REL_MODE",
    )
    p.add_argument(
        "--data_jsonl",
        default=DATA_JSONL,
        help="DevEval 完整数据集 jsonl 路径",
    )
    p.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help="批量实验结果（逐项目 + 总体）输出 CSV 路径",
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

    all_rows: List[Dict[str, Any]] = []
    per_project_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        project_name = str(row["project_name"]).strip()
        project_root = str(row["project_root"]).strip()
        if not project_name or not project_root:
            continue
        enre_json = resolve_enre_json_cell(row, project_name, args.base_enre)

        metrics = run_one_project(
            project_name=project_name,
            project_root=project_root,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
            source_code_dir=args.source_code_dir,
            data_jsonl=args.data_jsonl,
            project_rel_mode=project_rel_mode,
        )
        if not metrics:
            continue

        _append_rows_for_type(per_project_rows, metrics, project_name, "signature")
        _append_rows_for_type(per_project_rows, metrics, project_name, "code")
        _append_rows_for_type(per_project_rows, metrics, project_name, "desc")

    all_rows.extend(per_project_rows)

    # Compute aggregate metrics weighted by each project's number of tasks
    if per_project_rows:
        import collections

        grouped: Dict[tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in per_project_rows:
            key = (r["metric_type"], int(r["k"]))
            grouped[key].append(r)

        for (metric_type, k), rows in grouped.items():
            total_tasks = sum(r["tasks"] for r in rows)
            if total_tasks <= 0:
                continue
            P_total = sum(r["P"] * r["tasks"] for r in rows) / total_tasks
            R_total = sum(r["R"] * r["tasks"] for r in rows) / total_tasks
            F1_total = sum(r["F1"] * r["tasks"] for r in rows) / total_tasks
            match_total = sum(r["match"] for r in rows)
            pred_total = sum(r["pred"] for r in rows)
            gt_total = sum(r["gt"] for r in rows)
            all_rows.append(
                {
                    "project_name": "ALL",
                    "project_dir": "",
                    "metric_type": metric_type,
                    "k": k,
                    "P": P_total,
                    "R": R_total,
                    "F1": F1_total,
                    "match": match_total,
                    "pred": pred_total,
                    "gt": gt_total,
                    "tasks": total_tasks,
                }
            )

    if all_rows:
        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved batch BM25 metrics to {args.output_csv}")
    else:
        print("No metrics produced; nothing to save.")


if __name__ == "__main__":
    main()

