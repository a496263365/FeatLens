"""
批量对 Excel 中的多个项目运行 compare_graph_recall，保留每项目的 CSV 与 log，
并汇总各项目及总体的详细指标到一张 CSV。
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
import re
import sys
from typing import Any, Dict, List, Optional

_SRC_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import pandas as pd

from compare_graph_recall import run_compare_recall
from utils.excel_project_list import (
    normalize_excel_project_columns,
    resolve_enre_json_cell,
)

EXCEL_PATH = str(_package_path('data/evocodebench/projects_5.xlsx'))
SOURCE_CODE_DIR = str(_external_path('FEATLENS_EVOCODEBENCH_SOURCE_ROOT', 'external/evocodebench/source_code', ''))
DEFAULT_BASE_SEARCH_OUT = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_BASE_ENRE = str(_package_path('results/retrieval/evocodebench/featlens_and_rag'))
DEFAULT_OUTPUT_CSV = str(_package_path('results/retrieval/evocodebench/featlens_and_rag/batch_compare_graph_recall_summary.csv'))
DEBUG = True


def _k_from_rank_dir(name: str) -> Optional[int]:
    """从 rank 子目录名解析 k，如 PageRank-10-subgraph -> 10。"""
    m = re.search(r"(?:PageRank[-_]?)?(\d+)", name, re.I)
    return int(m.group(1)) if m else None


def _rank_key(name: str) -> str:
    k = _k_from_rank_dir(name)
    return f"rank_{k}" if k is not None else name.replace("-", "_").replace(" ", "_")


def _r_p_f1(match: int, pred: int, gt: int) -> tuple:
    r = (match / gt) if gt > 0 else 0.0
    p = (match / pred) if pred > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return r, p, f1


def run_one_project(
    project_name: str,
    enre_json: str,
    base_search_out: str,
) -> Optional[Dict[str, Any]]:
    project_base = os.path.join(base_search_out, project_name)
    graph_results_dir = os.path.join(project_base, "graph_results_***all")
    filtered_jsonl_path = os.path.join(project_base, "filtered.jsonl")
    output_report_file = os.path.join(project_base, "compare_graph_recall_report.csv")
    debug_log_file = os.path.join(project_base, "compare_graph_recall.debug.log")

    if not os.path.isdir(graph_results_dir):
        print(f"[skip] {project_name}: graph_results_***all not found at {graph_results_dir}")
        return None
    if not os.path.isfile(filtered_jsonl_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_jsonl_path}")
        return None

    print(f"[run] {project_name} (compare_graph_recall)")
    stats = run_compare_recall(
        graph_results_dir=graph_results_dir,
        filtered_jsonl_path=filtered_jsonl_path,
        output_report_file=output_report_file,
        enre_json=enre_json,
        debug=DEBUG,
        debug_log_file=debug_log_file,
    )
    stats["project_name"] = project_name
    return stats


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch run compare_graph_recall for multiple projects from Excel; output per-project CSV/log and one summary CSV."
    )
    p.add_argument("--excel_path", default=EXCEL_PATH, help="Excel 路径，含 project_name，可选 enre_json")
    p.add_argument("--base_search_out", default=DEFAULT_BASE_SEARCH_OUT, help="各项目目录所在根目录")
    p.add_argument("--base_enre", default=DEFAULT_BASE_ENRE, help="ENRE 根目录（未填 enre_json 时推导）")
    p.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="汇总指标 CSV 路径")
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = normalize_excel_project_columns(df)

    if "project_name" not in df.columns:
        raise SystemExit("Excel 需包含「项目名称」或 project_name 列")

    all_project_stats: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        project_name = str(row["project_name"]).strip()
        if not project_name:
            continue
        enre_json = resolve_enre_json_cell(row, project_name, args.base_enre)

        st = run_one_project(
            project_name=project_name,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
        )
        if st is None:
            continue
        all_project_stats.append(st)

    if not all_project_stats:
        print("No project stats collected; nothing to write.")
        return

    # Find every rank subdirectory, sort by k, and derive the shared rank-column prefixes
    rank_names_sorted: List[str] = []
    seen_ks: set = set()
    for st in all_project_stats:
        for name in st.get("rank_improved_decreased", {}).keys():
            k = _k_from_rank_dir(name)
            if k is not None and k not in seen_ks:
                seen_ks.add(k)
    for k in sorted(seen_ks):
        for st in all_project_stats:
            for name in st.get("rank_improved_decreased", {}).keys():
                if _k_from_rank_dir(name) == k:
                    rank_names_sorted.append(name)
                    break
            else:
                continue
            break
    # If a project has no rank subdirectory, use the rank name from the first project
    if not rank_names_sorted and all_project_stats:
        rd = all_project_stats[0].get("rank_pred_by_dir", {})
        rank_names_sorted = sorted(rd.keys(), key=lambda x: (_k_from_rank_dir(x) or 0, x))

    def row_from_stats(st: Dict[str, Any], is_overall: bool = False) -> Dict[str, Any]:
        total_tasks = st.get("total_tasks", 0)
        sum_gt = st.get("sum_gt", 0)
        ori_match = st.get("ori_match", 0)
        ori_pred = st.get("ori_pred", 0)
        mid_match = st.get("mid_match", 0)
        mid_pred = st.get("mid_pred", 0)
        ori_r, ori_p, ori_f1 = _r_p_f1(ori_match, ori_pred, sum_gt)
        mid_r, mid_p, mid_f1 = _r_p_f1(mid_match, mid_pred, sum_gt)

        out = {
            "project_name": st.get("project_name", ""),
            "total_tasks": total_tasks,
            "mid_improved": st.get("mid_improved", 0),
            "mid_decreased": st.get("mid_decreased", 0),
            "sum_gt": sum_gt,
            "ori_match": ori_match,
            "ori_pred": ori_pred,
            "ori_R": round(ori_r, 6),
            "ori_P": round(ori_p, 6),
            "ori_F1": round(ori_f1, 6),
            "mid_match": mid_match,
            "mid_pred": mid_pred,
            "mid_R": round(mid_r, 6),
            "mid_P": round(mid_p, 6),
            "mid_F1": round(mid_f1, 6),
        }
        for name in rank_names_sorted:
            key = _rank_key(name)
            improved, decreased = st.get("rank_improved_decreased", {}).get(name, (0, 0))
            rmatch = st.get("rank_match_by_dir", {}).get(name, 0)
            rpred = st.get("rank_pred_by_dir", {}).get(name, 0)
            rr, rp, rf1 = _r_p_f1(rmatch, rpred, sum_gt)
            out[f"{key}_improved"] = improved
            out[f"{key}_decreased"] = decreased
            out[f"{key}_match"] = rmatch
            out[f"{key}_pred"] = rpred
            out[f"{key}_R"] = round(rr, 6)
            out[f"{key}_P"] = round(rp, 6)
            out[f"{key}_F1"] = round(rf1, 6)
        return out

    rows: List[Dict[str, Any]] = []
    for st in all_project_stats:
        rows.append(row_from_stats(st))

    # Overall rows sum task counts and improved/decreased counts; R/P/F1 are weighted averages by task count
    total_tasks_sum = sum(s["total_tasks"] for s in all_project_stats)
    sum_gt_all = sum(s["sum_gt"] for s in all_project_stats)
    ori_match_all = sum(s["ori_match"] for s in all_project_stats)
    ori_pred_all = sum(s["ori_pred"] for s in all_project_stats)
    mid_match_all = sum(s["mid_match"] for s in all_project_stats)
    mid_pred_all = sum(s["mid_pred"] for s in all_project_stats)

    mid_improved_all = sum(s["mid_improved"] for s in all_project_stats)
    mid_decreased_all = sum(s["mid_decreased"] for s in all_project_stats)

    # R/P weighted by task count; overall F1 is computed from the weighted P and R
    def weighted_r_p(stats_list: List[Dict], key_r: str, key_p: str) -> tuple:
        tw = 0
        rw = 0.0
        pw = 0.0
        for s in stats_list:
            t = s.get("total_tasks", 0)
            if t <= 0:
                continue
            row = row_from_stats(s)
            tw += t
            rw += row.get(key_r, 0) * t
            pw += row.get(key_p, 0) * t
        if tw <= 0:
            return 0.0, 0.0
        return rw / tw, pw / tw

    ori_R_all, ori_P_all = weighted_r_p(all_project_stats, "ori_R", "ori_P")

    overall = {
        "project_name": "ALL",
        "total_tasks": total_tasks_sum,
        "mid_improved": mid_improved_all,
        "mid_decreased": mid_decreased_all,
        "sum_gt": sum_gt_all,
        "ori_match": ori_match_all,
        "ori_pred": ori_pred_all,
        "ori_R": round(ori_R_all, 6),
        "ori_P": round(ori_P_all, 6),
        "ori_F1": round(_safe_f1(ori_P_all, ori_R_all), 6),
        "mid_match": mid_match_all,
        "mid_pred": mid_pred_all,
    }
    mid_R_all, mid_P_all = weighted_r_p(all_project_stats, "mid_R", "mid_P")
    overall["mid_R"] = round(mid_R_all, 6)
    overall["mid_P"] = round(mid_P_all, 6)
    overall["mid_F1"] = round(_safe_f1(mid_P_all, mid_R_all), 6)

    for name in rank_names_sorted:
        key = _rank_key(name)
        improved_all = sum(s.get("rank_improved_decreased", {}).get(name, (0, 0))[0] for s in all_project_stats)
        decreased_all = sum(s.get("rank_improved_decreased", {}).get(name, (0, 0))[1] for s in all_project_stats)
        rmatch_all = sum(s.get("rank_match_by_dir", {}).get(name, 0) for s in all_project_stats)
        rpred_all = sum(s.get("rank_pred_by_dir", {}).get(name, 0) for s in all_project_stats)
        rr, rp = weighted_r_p(all_project_stats, f"{key}_R", f"{key}_P")
        overall[f"{key}_improved"] = improved_all
        overall[f"{key}_decreased"] = decreased_all
        overall[f"{key}_match"] = rmatch_all
        overall[f"{key}_pred"] = rpred_all
        overall[f"{key}_R"] = round(rr, 6)
        overall[f"{key}_P"] = round(rp, 6)
        overall[f"{key}_F1"] = round(_safe_f1(rp, rr), 6)

    rows.append(overall)

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved batch compare_graph_recall summary to {args.output_csv}")


def _safe_f1(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2 * p * r / (p + r)


if __name__ == "__main__":
    main()
