
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

import os
import glob
import re
import json
import sys
import networkx as nx
import pandas as pd
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Default path arguments, provided only as command-line conveniences for direct execution of this script;
# During batch experiments, pass these paths through run_compare_recall(...).
GRAPH_RESULTS_DIR = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/graphs'))
FILTERED_JSONL_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
OUTPUT_REPORT_FILE = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/303_expand_graph_match_comparison_report.csv'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))

DEBUG = True
DEBUG_LOG_FILE = os.path.join(os.path.dirname(OUTPUT_REPORT_FILE), "compare_graph_recall.debug.log")

variables_enre = set()  # Variable type: count a hit when the retrieved code uses this variable
unresolved_attribute_enre = set()  # In ENRE, this type usually denotes a self.xxx attribute of a class; count a hit when the retrieved code contains self.xxx
module_enre = set()  # Module, effectively a Python file; standalone module names may appear in dependencies, and retrieving any element from that module counts as a hit
package_enre = set()  # Package: handled similarly to a module


def clear_enre_elements() -> None:
    """清空 ENRE 元素集合。批量跑多项目时，每切换项目前调用。"""
    variables_enre.clear()
    unresolved_attribute_enre.clear()
    module_enre.clear()
    package_enre.clear()


def load_enre_elements(json_path: str) -> None:
    if not os.path.exists(json_path):
        print(f"Warning: ENRE JSON file not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading ENRE JSON: {e}")
        return

    variables = data.get('variables', [])
    for var in variables:
        if var.get('category') == 'Variable':
            qname = var.get('qualifiedName')
            if qname:
                variables_enre.add(qname)
        elif var.get('category') == 'Unresolved Attribute':
            # The File field must exist and contain '/'; otherwise the symbol may be an attribute of an external library
            if 'File' in var and '/' in var.get('File'):
                qname = var.get('qualifiedName')
                if qname:
                    unresolved_attribute_enre.add(qname)
        elif var.get('category') == 'Module':
            qname = var.get('qualifiedName')
            if qname:
                module_enre.add(qname)
        elif var.get('category') == 'Package':
            qname = var.get('qualifiedName')
            if qname:
                package_enre.add(qname)


def _normalize_symbol(s: str) -> str:
    if "(" in s:
        return s.split("(", 1)[0]
    return s

def _normalize_sig_like_rank(sig: str) -> str:
    s = "" if sig is None else str(sig)
    base = s.split("(", 1)[0] if "(" in s else s
    return base.lstrip(".")

def _get_target_method_sig_from_gml(gml_path: str) -> str:
    if not gml_path or not os.path.exists(gml_path):
        return ""
    try:
        G = nx.read_gml(gml_path)
    except Exception:
        return ""
    return _normalize_sig_like_rank(G.graph.get("target_method", ""))

def _exclude_target_method_from_context(context_code_list, target_method_sig: str):
    if not target_method_sig:
        return context_code_list
    excluded = 0
    filtered = []
    for x in context_code_list:
        ms = str(x.get("method_signature", ""))
        node_sig = str(x.get("node_sig", ""))
        if _normalize_sig_like_rank(ms) == target_method_sig or _normalize_sig_like_rank(node_sig) == target_method_sig:
            excluded += 1
            continue
        filtered.append(x)
    if DEBUG and excluded:
        print(f"Excluded target_method from mid recall: target_method={target_method_sig} excluded_items={excluded}")
    return filtered

def compute_task_recall(dependency, searched_context_code_list):
    dep = dependency or []
    dep_set = {x for x in dep}

    # Matches here can only be functions or classes
    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    # Special case: convert retrieved elements such as xx.__init__.yy to xx.yy because ENRE and DevEval use different representations
    retrieved_set = {x.replace(".__init__", "") for x in retrieved_set}

    dep_total = len(dep_set)

    hit_set = dep_set & retrieved_set
    hit = len(hit_set) if dep_total > 0 else 0

    # Special case
    for x in dep:
        if x in variables_enre:
            # For a variable, check whether it appears in any retrieved code
            var_name = x.split('.')[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code", "")
                if var_name in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in unresolved_attribute_enre:
            attr_name = x.split('.')[-1]  # Attribute name
            class_name = '.'.join(x.split('.')[:-1])  # The class name is everything before the attribute name
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                code_detail = context_code.get("method_code", "")
                # If the retrieved code is actually inside class_name and contains the self.xxx attribute, count it as a hit
                if sig.startswith(f"{class_name}.") and f"self.{attr_name}" in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in module_enre:
            # For a module, check whether its name is a prefix of the signature of any context code
            module_name = x
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                if sig.startswith(module_name):
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in package_enre:
            # For a package, check whether its name is a prefix of the signature of any context code, similarly to Module handling
            package_name = x
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                if sig.startswith(package_name):
                    hit_set.add(x)
                    hit += 1
                    break
    
    if DEBUG:
        print(f"retrieved_set: {retrieved_set}")
        print(f"hit_set: {hit_set}")

    recall = (hit / dep_total) if dep_total > 0 else None
    return {
        "dependency_total": dep_total,
        "dependency_hit": hit,
        "recall": recall,
    }

def load_ground_truth(task_id: str, filtered_jsonl_path: str) -> List[str]:
    """
    Loads ground truth dependency symbols for a specific task ID from the JSONL file.
    Returns a list of symbols.
    """
    dep = []
    if not os.path.exists(filtered_jsonl_path):
        print(f"Warning: Filtered JSONL file not found at {filtered_jsonl_path}")
        return dep

    try:
        target_line_num = int(task_id)
        current_line_num = 0
        with open(filtered_jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                current_line_num += 1
                
                # Check if this is the target line (1-based index)
                if current_line_num == target_line_num:
                    data = json.loads(line)
                    dependency = data.get('dependency', {})
                    dep.extend(dependency.get('intra_class', []))
                    dep.extend(dependency.get('intra_file', []))
                    dep.extend(dependency.get('cross_file', []))
                    break
    except ValueError:
        print(f"Error: Invalid task_id '{task_id}' - must be an integer.")
    except Exception as e:
        print(f"Error reading filtered JSONL: {e}")
        
    return dep

def load_context_code_list_from_gml(gml_path):
    if not gml_path or not os.path.exists(gml_path):
        return [], 0

    try:
        G = nx.read_gml(gml_path)
    except Exception as e:
        print(f"Error reading {gml_path}: {e}")
        return [], 0

    context_code_list = []
    for node_id in G.nodes():
        node = G.nodes[node_id]
        if node.get('category') != 'Function':
            continue

        context_code_list.append({
            "sig": str(node.get("method_signature", "")),
            "node_sig": str(node.get("sig", "")),
            "method_signature": str(node.get("method_signature", "")),
            "method_code": str(node.get("method_code", "")),
        })

    return context_code_list, G.number_of_nodes()

def get_matched_count(gml_path, gt_sigs):
    """
    Reads a GML file and computes dependency hit count with variable-aware matching.
    """
    context_code_list, _pred = load_context_code_list_from_gml(gml_path)
    stats = compute_task_recall(list(gt_sigs), context_code_list)
    return stats.get("dependency_hit", 0)

def list_rank_subdirs(base_dir):
    if not base_dir or not os.path.isdir(base_dir):
        return []

    rank_dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        rank_dirs.append(name)
    return sorted(rank_dirs)

def sanitize_col(s):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")

def _redirect_stdout_stderr_to_file(log_file: str):
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return open(log_file, "w", encoding="utf-8", buffering=1)


def run_compare_recall(
    *,
    graph_results_dir: str,
    filtered_jsonl_path: str,
    output_report_file: str,
    enre_json: str,
    debug: bool = DEBUG,
    debug_log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对单个项目运行图召回率对比：读取 ori/mid/rank GML，与 ground truth 对比，写 CSV 和 log，并返回完整统计供批量汇总。
    """
    stdout0, stderr0 = sys.stdout, sys.stderr
    log_fp = None
    if debug:
        log_file = debug_log_file or os.path.join(os.path.dirname(output_report_file), "compare_graph_recall.debug.log")
        log_fp = _redirect_stdout_stderr_to_file(log_file)
        sys.stdout = log_fp
        sys.stderr = log_fp

    clear_enre_elements()
    load_enre_elements(enre_json)

    # 1. Group files by task_id
    # Pattern: task_{id}_{type}.gml
    base_files = glob.glob(os.path.join(graph_results_dir, "*.gml"))
    task_files = defaultdict(dict)

    for f in base_files:
        basename = os.path.basename(f)
        match = re.search(r'task_(\d+)_(\w+)\.gml', basename)
        if match:
            task_id = match.group(1)
            file_type = match.group(2)
            task_files[task_id][file_type] = f

    rank_subdirs = list_rank_subdirs(graph_results_dir)
    rank_files_by_dir = {name: {} for name in rank_subdirs}
    for name in rank_subdirs:
        rank_dir = os.path.join(graph_results_dir, name)
        for f in glob.glob(os.path.join(rank_dir, "*.gml")):
            basename = os.path.basename(f)
            match = re.search(r'task_(\d+)_rank\.gml', basename)
            if match:
                task_id = match.group(1)
                rank_files_by_dir[name][task_id] = f

    # 2. Analyze each task
    results = []
    sum_ori_pred = 0
    sum_mid_pred = 0
    sum_rank_pred_by_dir = {name: 0 for name in rank_subdirs}
    
    # Sort by task_id integer
    sorted_task_ids = sorted(task_files.keys(), key=lambda x: int(x))
    
    header_parts = [
        f"{'Task ID':<10}",
        f"{'GT Total':<10}",
        f"{'Ori Hit':<10}",
        f"{'Mid Hit':<10}",
    ]
    for name in rank_subdirs:
        header_parts.append(f"{name + ' Hit':<18}")
    header_parts.extend(
        [
            f"{'Ori Recall':<10}",
            f"{'Mid Recall':<10}",
        ]
    )
    for name in rank_subdirs:
        header_parts.append(f"{name + ' Recall':<18}")

    print(" | ".join(header_parts))
    print("-" * (len(" | ".join(header_parts))))

    for task_id in sorted_task_ids:
        types = task_files[task_id]
        
        # Load GT
        dep = load_ground_truth(task_id, filtered_jsonl_path)
        print(f"task_id: {task_id}, dep: {dep}")
        gt_total = len(set(dep))

        ori_ctx, ori_pred = load_context_code_list_from_gml(types.get('ori'))
        mid_ctx, mid_pred = load_context_code_list_from_gml(types.get('mid'))
        sum_ori_pred += ori_pred
        sum_mid_pred += mid_pred

        ori_stats = compute_task_recall(dep, ori_ctx)
        target_method_sig = _get_target_method_sig_from_gml(types.get('mid')) or _get_target_method_sig_from_gml(types.get('ori'))
        mid_stats = compute_task_recall(dep, _exclude_target_method_from_context(mid_ctx, target_method_sig))

        ori_hit = ori_stats.get("dependency_hit", 0)
        mid_hit = mid_stats.get("dependency_hit", 0)

        ori_recall = ori_stats.get("recall")
        mid_recall = mid_stats.get("recall")

        def fmt_recall(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "None"

        row_parts = [
            f"{task_id:<10}",
            f"{gt_total:<10}",
            f"{ori_hit:<10}",
            f"{mid_hit:<10}",
        ]
        rank_stats_by_dir = {}
        for name in rank_subdirs:
            rank_path = rank_files_by_dir.get(name, {}).get(task_id)
            rank_ctx, rank_pred = load_context_code_list_from_gml(rank_path)
            sum_rank_pred_by_dir[name] += rank_pred
            rank_stats = compute_task_recall(dep, rank_ctx)
            rank_stats_by_dir[name] = rank_stats
            rank_hit = rank_stats.get("dependency_hit", 0)
            rank_recall = rank_stats.get("recall")
            row_parts.append(f"{rank_hit:<18}")
        row_parts.extend(
            [
                f"{fmt_recall(ori_recall):<10}",
                f"{fmt_recall(mid_recall):<10}",
            ]
        )
        for name in rank_subdirs:
            rank_recall = rank_stats_by_dir.get(name, {}).get("recall")
            row_parts.append(f"{fmt_recall(rank_recall):<18}")

        print(" | ".join(row_parts))
        
        row = {
            'task_id': task_id,
            'dependency_total': gt_total,
            'ori_hit': ori_hit,
            'mid_hit': mid_hit,
            'ori_recall': ori_recall,
            'mid_recall': mid_recall,
        }
        for name in rank_subdirs:
            col = sanitize_col(name)
            stats = rank_stats_by_dir.get(name, {})
            row[f"rank_hit__{col}"] = stats.get("dependency_hit", 0)
            row[f"rank_recall__{col}"] = stats.get("recall")
        results.append(row)

    # 3. Save to CSV
    df = pd.DataFrame(results)
    ordered_cols = [
        "task_id",
        "dependency_total",
        "ori_hit",
        "mid_hit",
    ]
    ordered_cols.extend([f"rank_hit__{sanitize_col(name)}" for name in rank_subdirs])
    ordered_cols.extend(
        [
            "ori_recall",
            "mid_recall",
        ]
    )
    ordered_cols.extend([f"rank_recall__{sanitize_col(name)}" for name in rank_subdirs])
    existing_cols = [c for c in ordered_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]
    df.to_csv(output_report_file, index=False)
    print(f"\nReport saved to: {output_report_file}")

    # 4. Statistics Summary
    total_tasks = len(results)
    if total_tasks > 0:
        def is_valid_recall(x):
            return isinstance(x, (int, float))

        # Mid Stats
        mid_improved = sum(
            1 for r in results
            if is_valid_recall(r.get('mid_recall')) and is_valid_recall(r.get('ori_recall')) and r['mid_recall'] > r['ori_recall']
        )
        mid_decreased = sum(
            1 for r in results
            if is_valid_recall(r.get('mid_recall')) and is_valid_recall(r.get('ori_recall')) and r['mid_recall'] < r['ori_recall']
        )

        # Totals
        sum_gt = sum(r['dependency_total'] for r in results)
        sum_ori = sum(r['ori_hit'] for r in results)
        sum_mid = sum(r['mid_hit'] for r in results)
        
        # Calculate percentages
        def calc_pct(count, total):
            return (count / total * 100) if total > 0 else 0
            
        print("\n" + "="*60)
        print("STATISTICS SUMMARY")
        print("="*60)
        
        print(f"Total Tasks: {total_tasks}")
        print("-" * 30)
        
        print(f"Mid Method:")
        print(f"  Improved:  {mid_improved:3d} tasks ({calc_pct(mid_improved, total_tasks):6.2f}%)")
        print(f"  Decreased: {mid_decreased:3d} tasks ({calc_pct(mid_decreased, total_tasks):6.2f}%)")
        
        rank_improved_decreased = {}
        for name in rank_subdirs:
            col = sanitize_col(name)
            improved = sum(
                1 for r in results
                if is_valid_recall(r.get(f"rank_recall__{col}")) and is_valid_recall(r.get('ori_recall')) and r[f"rank_recall__{col}"] > r['ori_recall']
            )
            decreased = sum(
                1 for r in results
                if is_valid_recall(r.get(f"rank_recall__{col}")) and is_valid_recall(r.get('ori_recall')) and r[f"rank_recall__{col}"] < r['ori_recall']
            )
            rank_improved_decreased[name] = (improved, decreased)
            print("-" * 30)
            print(f"Rank Method ({name}):")
            print(f"  Improved:  {improved:3d} tasks ({calc_pct(improved, total_tasks):6.2f}%)")
            print(f"  Decreased: {decreased:3d} tasks ({calc_pct(decreased, total_tasks):6.2f}%)")
        
        print("="*60)
        ratio_parts = [f"{mid_improved}/{mid_decreased}"]
        ratio_parts.extend([f"{rank_improved_decreased.get(name, (0, 0))[0]}/{rank_improved_decreased.get(name, (0, 0))[1]}" for name in rank_subdirs])
        print(" ".join(ratio_parts))
        print("="*60)
        print("RECALL STATISTICS")
        print("-" * 30)
        
        recall_ori = calc_pct(sum_ori, sum_gt)
        recall_mid = calc_pct(sum_mid, sum_gt)
        
        print(f"Total Ground Truth: {sum_gt}")
        print(f"Total Ori Matches:  {sum_ori} / Preds: {sum_ori_pred} (Recall: {recall_ori:.2f}%)，{recall_ori:.2f}%({sum_ori}/{sum_ori_pred})")
        print(f"Total Mid Matches:  {sum_mid} / Preds: {sum_mid_pred} (Recall: {recall_mid:.2f}%)，{recall_mid:.2f}%({sum_mid}/{sum_mid_pred})")
        for name in rank_subdirs:
            col = sanitize_col(name)
            sum_rank = sum(r.get(f"rank_hit__{col}", 0) for r in results)
            recall_rank = calc_pct(sum_rank, sum_gt)
            sum_rank_pred = sum_rank_pred_by_dir.get(name, 0)
            print(f"Total Rank Matches ({name}): {sum_rank} / Preds: {sum_rank_pred} (Recall: {recall_rank:.2f}%)，{recall_rank:.2f}%({sum_rank}/{sum_rank_pred})")
        
        print("-" * 30)
        print(f"Mid Recall Growth:  {recall_mid - recall_ori:+.2f}%")
        for name in rank_subdirs:
            col = sanitize_col(name)
            sum_rank = sum(r.get(f"rank_hit__{col}", 0) for r in results)
            recall_rank = calc_pct(sum_rank, sum_gt)
            print(f"Rank Recall Growth ({name}): {recall_rank - recall_ori:+.2f}%")
        
        print("="*60)
        print("AVERAGE MATCH STATISTICS")
        print("-" * 30)
        
        avg_ori = sum_ori / total_tasks
        avg_mid = sum_mid / total_tasks
        
        print(f"Average Ori Matches:  {avg_ori:.2f}")
        print(f"Average Mid Matches:  {avg_mid:.2f}")
        for name in rank_subdirs:
            col = sanitize_col(name)
            sum_rank = sum(r.get(f"rank_hit__{col}", 0) for r in results)
            avg_rank = sum_rank / total_tasks
            print(f"Average Rank Matches ({name}): {avg_rank:.2f}")
        print("="*60)

    if log_fp is not None:
        log_fp.flush()
        sys.stdout = stdout0
        sys.stderr = stderr0
        log_fp.close()

    # Build return stats for batch summary
    stats: Dict[str, Any] = {
        "total_tasks": len(results),
        "results": results,
        "sum_gt": 0,
        "ori_match": 0,
        "ori_pred": 0,
        "mid_match": 0,
        "mid_pred": 0,
        "mid_improved": 0,
        "mid_decreased": 0,
        "rank_improved_decreased": {},
        "rank_match_by_dir": {},
        "rank_pred_by_dir": dict(sum_rank_pred_by_dir) if results else {},
    }
    if results:
        stats["sum_gt"] = sum(r["dependency_total"] for r in results)
        stats["ori_match"] = sum(r["ori_hit"] for r in results)
        stats["mid_match"] = sum(r["mid_hit"] for r in results)
        stats["ori_pred"] = sum_ori_pred
        stats["mid_pred"] = sum_mid_pred
        stats["rank_pred_by_dir"] = dict(sum_rank_pred_by_dir)

        def is_valid_recall(x):
            return isinstance(x, (int, float))

        stats["mid_improved"] = sum(
            1 for r in results
            if is_valid_recall(r.get("mid_recall")) and is_valid_recall(r.get("ori_recall")) and r["mid_recall"] > r["ori_recall"]
        )
        stats["mid_decreased"] = sum(
            1 for r in results
            if is_valid_recall(r.get("mid_recall")) and is_valid_recall(r.get("ori_recall")) and r["mid_recall"] < r["ori_recall"]
        )
        for name in rank_subdirs:
            col = sanitize_col(name)
            improved = sum(
                1 for r in results
                if is_valid_recall(r.get(f"rank_recall__{col}")) and is_valid_recall(r.get("ori_recall")) and r[f"rank_recall__{col}"] > r["ori_recall"]
            )
            decreased = sum(
                1 for r in results
                if is_valid_recall(r.get(f"rank_recall__{col}")) and is_valid_recall(r.get("ori_recall")) and r[f"rank_recall__{col}"] < r["ori_recall"]
            )
            stats["rank_improved_decreased"][name] = (improved, decreased)
            stats["rank_match_by_dir"][name] = sum(r.get(f"rank_hit__{col}", 0) for r in results)

    return stats


def main() -> None:
    run_compare_recall(
        graph_results_dir=GRAPH_RESULTS_DIR,
        filtered_jsonl_path=FILTERED_JSONL_PATH,
        output_report_file=OUTPUT_REPORT_FILE,
        enre_json=ENRE_JSON,
        debug=DEBUG,
        debug_log_file=DEBUG_LOG_FILE,
    )


if __name__ == "__main__":
    main()
