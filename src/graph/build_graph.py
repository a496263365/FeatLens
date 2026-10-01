
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

import pandas as pd
import json
import networkx as nx
# import matplotlib.pyplot as plt # Not essential for just building/saving, but good if visualization needed
import os
from collections import defaultdict
from typing import Dict, Any


# Default path arguments, provided only as command-line conveniences for direct execution of this script;
# During batch experiments, pass these paths through function arguments.
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
DIAGNOSTIC_JSONL = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/diagnostic_***feature.jsonl'))
OUTPUT_GRAPH_PATH = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/graphs'))

REMOVE_FIRST_DOT_PREFIX = False
PREFIX = "mrjob"  # When prefix removal is enabled, specify the project name as the prefix here
# Controls how many similar methods are selected for later class call-chain prediction bonuses
SIMILAR_TOPK = 3


def _load_methods_info(methods_csv: str):
    print("Loading METHODS_CSV...")
    df_methods = pd.read_csv(methods_csv)
    method_sig_to_info = {}
    method_clean_sig_to_info = {}
    for _, row in df_methods.iterrows():
        sig = str(row["method_signature"])
        method_sig_to_info[sig] = row.to_dict()
        clean_sig = sig.split("(", 1)[0] if "(" in sig else sig
        method_clean_sig_to_info[clean_sig] = row.to_dict()
    print(f"Loaded {len(method_sig_to_info)} methods from CSV.")
    return method_sig_to_info, method_clean_sig_to_info


def _load_enre_graph(enre_json: str):
    print("Loading ENRE_JSON and processing variables...")
    with open(enre_json, "r") as f:
        enre_data = json.load(f)

    variables = enre_data.get("variables", [])
    cells = enre_data.get("cells", [])

    valid_nodes = {}  # id -> node_data
    id_to_qname = {}  # id -> qualifiedName
    qname_to_id = {}  # qualifiedName -> id

    for var in variables:
        cat = var.get("category", "")
        # Ignore entries whose category is Unresolved or starts with Unresolved
        if not cat.startswith("Un"):
            vid = var["id"]
            qname = var["qualifiedName"]
            valid_nodes[vid] = var
            id_to_qname[vid] = qname
            qname_to_id[qname] = vid

    print(f"Filtered {len(valid_nodes)} valid nodes from ENRE_JSON.")

    print("Processing relations in ENRE_JSON...")
    edges = []
    seen_edges = set()
    for cell in cells:
        src = cell.get("src")
        dest = cell.get("dest")
        values = cell.get("values", {})
        kind = values.get("kind")

        # Add all valid edges to edges
        if src in valid_nodes and dest in valid_nodes:
            edge_tuple = (src, dest, kind)
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                edges.append((src, dest, kind))

    print(f"Loaded {len(edges)} valid edges.")

    print("Graph construction logic needs adjacency list...")
    adj = defaultdict(list)
    for src, dest, kind in edges:
        adj[src].append((dest, kind))

    return valid_nodes, id_to_qname, qname_to_id, adj


def _maybe_insert_init_in_target_method(target_method):
    if not target_method:
        return target_method
    base = target_method.split("(", 1)[0] if "(" in target_method else target_method
    if ".__init__." in base:
        return base
    parts = [p for p in base.split(".") if p]
    if len(parts) < 2:
        return base
    insert_at = None
    for i, part in enumerate(parts):
        if any(c.isupper() for c in part):
            insert_at = i
            break
    if insert_at is None:
        insert_at = len(parts) - 1
    insert_at = max(1, insert_at)
    parts.insert(insert_at, "__init__")
    return ".".join(parts)

def build_graph(
    *,
    methods_csv: str,
    enre_json: str,
    filtered_path: str,
    diagnostic_jsonl: str,
    output_graph_path: str,
) -> Dict[str, Any]:
    method_sig_to_info, method_clean_sig_to_info = _load_methods_info(methods_csv)
    valid_nodes, id_to_qname, qname_to_id, adj = _load_enre_graph(enre_json)

    print("Step 1: Loading FILTERED_PATH for task info...")
    tasks_info = []
    # Load all tasks for one project, one task per line
    with open(filtered_path, "r") as f:
        for line in f:
            if line.strip():
                tasks_info.append(json.loads(line))
    print(f"Loaded {len(tasks_info)} tasks from FILTERED_PATH.")

    print("Step 2 Loading DIAGNOSTIC_JSONL and Building Graph...")
    diag_records = []
    # Load previous search results, including match indicators
    with open(diagnostic_jsonl, "r") as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    os.makedirs(output_graph_path, exist_ok=True)

    total_tasks = len(diag_records)
    tasks_same_file_nonzero = 0
    tasks_same_feature_nonzero = 0
    tasks_similar_method_nonzero = 0
        
    for i, rec in enumerate(diag_records):
        example_id = rec.get('example_id', i)
        target_method = tasks_info[i].get("namespace") or ""
        if REMOVE_FIRST_DOT_PREFIX:
            target_method = PREFIX + "." + target_method
        target_feature_other_methods = rec.get("target_feature_other_methods") or []
        target_feature_other_methods_set = {
            (x.split("(", 1)[0] if "(" in str(x) else str(x))
            for x in target_feature_other_methods
            if x
        }

        similar_methods = rec.get("similar_methods")
        if not isinstance(similar_methods, dict):
            similar_methods = {}
        similar_method_names = similar_methods.get(f"top{SIMILAR_TOPK}") or []
        similar_method_names_set = {
            (x.split("(", 1)[0] if "(" in str(x) else str(x))
            for x in similar_method_names
            if x
        }

        target_file = ""
        if target_method in method_clean_sig_to_info:
            target_file = str(method_clean_sig_to_info[target_method].get("func_file", ""))
        if not target_file:
            target_method_with_init = _maybe_insert_init_in_target_method(target_method)
            if target_method_with_init != target_method and target_method_with_init in method_clean_sig_to_info:
                target_file = str(method_clean_sig_to_info[target_method_with_init].get("func_file", ""))
        print(f"[task] example_id={example_id} target_method={target_method} target_file={target_file}", flush=True)

        preds = []
        try:
            # preds = rec["feature"]["top3"]["predictions"]
            preds = rec["feature"]["top3"]["predictions"]# These are the predictions for the specified idea
            filtered_preds = [
                p
                for p in preds
                if (p["method"] if "(" not in p["method"] else p["method"].split("(", 1)[0])
                != target_method
            ]
            if len(filtered_preds) != len(preds):
                print(
                    f"Attention! {len(preds) - len(filtered_preds)} items filtered out for task {example_id} (method == {target_method})",
                    flush=True,
                )
            preds = filtered_preds
        except KeyError:
            print(f"Skipping task {example_id}: missing feature/top3/predictions", flush=True)
            continue

        pred_signatures = [p["method"] for p in preds]

        G = nx.DiGraph()
        G.graph["target_method"] = target_method
        G.graph["target_file"] = target_file

        relevant_ids = set()
        for sig in pred_signatures:
            clean_sig = sig.split("(", 1)[0] if "(" in sig else sig

            if clean_sig in qname_to_id:
                eid = qname_to_id[clean_sig]
                relevant_ids.add(eid)
                node_info = valid_nodes[eid]

                attrs = {
                    'sig': clean_sig,
                    'category': node_info['category'],
                    'is_SameFile': False,
                    'is_SameFeature': False,
                    'is_SimilarMethod': False,
                }

                if sig in method_sig_to_info:
                    csv_info = method_sig_to_info[sig]
                    attrs['method_signature'] = str(csv_info.get('method_signature', ''))
                    attrs['func_file'] = str(csv_info.get('func_file', ''))
                    attrs['method_code'] = str(csv_info.get('method_code', ''))
                elif clean_sig in method_sig_to_info:
                    csv_info = method_sig_to_info[clean_sig]
                    attrs['method_signature'] = str(csv_info.get('method_signature', ''))
                    attrs['func_file'] = str(csv_info.get('func_file', ''))
                    attrs['method_code'] = str(csv_info.get('method_code', ''))
                else:
                    print(f"Warning: No method info found for {sig}", flush=True)

                G.add_node(eid, **attrs)

        for u in relevant_ids:
            if u in adj:
                for v, kind in adj[u]:
                    if v in relevant_ids:
                        G.add_edge(u, v, type=kind)

        same_file_count = 0
        same_feature_count = 0
        similar_method_count = 0
        if target_file:
            for _, attrs in G.nodes(data=True):
                if str(attrs.get("func_file", "")) == target_file:
                    attrs["is_SameFile"] = True
                    same_file_count += 1
        if target_feature_other_methods_set:
            for _, attrs in G.nodes(data=True):
                if str(attrs.get("sig", "")) in target_feature_other_methods_set:
                    attrs["is_SameFeature"] = True
                    same_feature_count += 1
        if similar_method_names_set:
            for _, attrs in G.nodes(data=True):
                if str(attrs.get("sig", "")) in similar_method_names_set:
                    attrs["is_SimilarMethod"] = True
                    similar_method_count += 1
        print(
            f"[task] example_id={example_id} same_file={same_file_count} same_feature={same_feature_count} similar_method={similar_method_count}",
            flush=True,
        )
        if same_file_count != 0:
            tasks_same_file_nonzero += 1
        if same_feature_count != 0:
            tasks_same_feature_nonzero += 1
        if similar_method_count != 0:
            tasks_similar_method_nonzero += 1

        out_file = os.path.join(output_graph_path, f"task_{example_id}_ori.gml")
        nx.write_gml(G, out_file)
        print(f"Saved graph for task {example_id} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print(
        f"[summary] total_tasks={total_tasks} same_file_nonzero_tasks={tasks_same_file_nonzero} same_feature_nonzero_tasks={tasks_same_feature_nonzero} similar_method_nonzero_tasks={tasks_similar_method_nonzero}",
        flush=True,
    )

    return {
        "total_tasks": total_tasks,
        "same_file_nonzero_tasks": tasks_same_file_nonzero,
        "same_feature_nonzero_tasks": tasks_same_feature_nonzero,
        "similar_method_nonzero_tasks": tasks_similar_method_nonzero,
    }


if __name__ == "__main__":
    build_graph(
        methods_csv=METHODS_CSV,
        enre_json=ENRE_JSON,
        filtered_path=FILTERED_PATH,
        diagnostic_jsonl=DIAGNOSTIC_JSONL,
        output_graph_path=OUTPUT_GRAPH_PATH,
    )
