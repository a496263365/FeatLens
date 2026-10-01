
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
from typing import Any, Dict, List, Optional

_SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import pandas as pd

from unixcoder_based_search import (
    evaluate_retrieval_with_unixcoder,
)
from utils.excel_project_list import (
    normalize_excel_project_columns,
    resolve_enre_json_cell,
)


EXCEL_PATH = str(_package_path('data/evocodebench/projects_5.xlsx'))
DEFAULT_BASE_SEARCH_OUT = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_BASE_ENRE = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_OUTPUT_CSV = str(_package_path('results/retrieval/evocodebench/featlens_and_rag/unixcoder_batch_metrics.csv'))


def run_one_project(
    project_name: str,
    enre_json: str,
    base_search_out: str,
    model_path: Optional[str],
    batch_size: int,
) -> Dict[str, Any]:
    project_base = os.path.join(base_search_out, project_name)
    methods_csv = os.path.join(project_base, "methods.csv")
    filtered_file = os.path.join(project_base, "filtered.jsonl")
    refined_queries_cache_path = os.path.join(project_base, "refined_queries.json")

    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return {}
    if not os.path.exists(filtered_file):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_file}")
        return {}

    print(f"[run] {project_name} (unixcoder)")
    metrics = evaluate_retrieval_with_unixcoder(
        methods_csv=methods_csv,
        filtered_file=filtered_file,
        enre_json=enre_json,
        refined_queries_cache_path=refined_queries_cache_path,
        model_path=model_path,
        batch_size=batch_size,
    )
    metrics["project_name"] = project_name
    return metrics


def _append_rows(
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    project_name: str,
) -> None:
    type_metrics = metrics.get("unixcoder", {})
    num_examples = metrics.get("num_examples", 0)
    for k, v in type_metrics.items():
        rows.append(
            {
                "project_name": project_name,
                "metric_type": "unixcoder",
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
    p = argparse.ArgumentParser(
        description="Batch run UniXcoder-based search experiments from Excel project list"
    )
    p.add_argument(
        "--excel_path",
        default=EXCEL_PATH,
        help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径",
    )
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="methods/filtered 等所在根目录",
    )
    p.add_argument(
        "--base_enre",
        default=DEFAULT_BASE_ENRE,
        help="ENRE 根目录（未填 enre_json 时推导用）",
    )
    p.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help="批量实验结果（逐项目 + 总体）输出 CSV 路径",
    )
    p.add_argument(
        "--model_path",
        default="",
        help="UniXcoder 模型路径或名称（留空则使用默认 microsoft/unixcoder-base）",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="UniXcoder 编码 batch size",
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

    model_path = args.model_path.strip() or None

    for _, row in df.iterrows():
        project_name = str(row["project_name"]).strip()
        if not project_name:
            continue
        enre_json = resolve_enre_json_cell(row, project_name, args.base_enre)

        metrics = run_one_project(
            project_name=project_name,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
            model_path=model_path,
            batch_size=args.batch_size,
        )
        if not metrics:
            continue

        _append_rows(per_project_rows, metrics, project_name)

    all_rows.extend(per_project_rows)

    # Compute aggregate metrics weighted by each project's number of tasks
    if per_project_rows:
        import collections

        grouped: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in per_project_rows:
            key = int(r["k"])
            grouped[key].append(r)

        for k, rows_for_k in grouped.items():
            total_tasks = sum(r["tasks"] for r in rows_for_k)
            if total_tasks <= 0:
                continue
            P_total = sum(r["P"] * r["tasks"] for r in rows_for_k) / total_tasks
            R_total = sum(r["R"] * r["tasks"] for r in rows_for_k) / total_tasks
            F1_total = sum(r["F1"] * r["tasks"] for r in rows_for_k) / total_tasks
            match_total = sum(r["match"] for r in rows_for_k)
            pred_total = sum(r["pred"] for r in rows_for_k)
            gt_total = sum(r["gt"] for r in rows_for_k)
            all_rows.append(
                {
                    "project_name": "ALL",
                    "metric_type": "unixcoder",
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
        print(f"Saved batch UniXcoder metrics to {args.output_csv}")
    else:
        print("No metrics produced; nothing to save.")


if __name__ == "__main__":
    main()

