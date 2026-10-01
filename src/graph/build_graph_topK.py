
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


METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/methods.csv'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/report-enre.json'))
FILTERED_PATH = str(_package_path('data/deveval/projects/Database/alembic/filtered.jsonl')) 
DIAGNOSTIC_JSONL = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/diagnostic_feature.jsonl'))
# OUTPUT_GRAPH_PATH = "<package_root>/results/devEvalSearchOut/Internet_boto/0115/graph_results"
OUTPUT_GRAPH_PATH = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/graphs'))
TOP_KS = [5, 10, 15, 20]

REMOVE_FIRST_DOT_PREFIX = False
PREFIX = "alembic"  # When prefix removal is enabled, specify the project name as the prefix here


# Load all previously processed method information
print("Loading METHODS_CSV...")
df_methods = pd.read_csv(METHODS_CSV)
method_sig_to_info = {}
for index, row in df_methods.iterrows():
    # Ensure we handle potential NaN values or convert to string if needed
    sig = str(row['method_signature'])
    method_sig_to_info[sig] = row.to_dict()

print(f"Loaded {len(method_sig_to_info)} methods from CSV.")


# Load a previously generated enre_report.json and process code elements and their relationships
print("Loading ENRE_JSON and processing variables...")
with open(ENRE_JSON, 'r') as f:
    enre_data = json.load(f)

variables = enre_data.get('variables', [])
cells = enre_data.get('cells', [])

valid_nodes = {} # id -> node_data
id_to_qname = {} # id -> qualifiedName
qname_to_id = {} # qualifiedName -> id

for var in variables:
    cat = var.get('category', '')
    # Ignore entries whose category is Unresolved or starts with Unresolved
    if not cat.startswith('Un'):
        vid = var['id']
        qname = var['qualifiedName']
        
        valid_nodes[vid] = var
        id_to_qname[vid] = qname
        qname_to_id[qname] = vid
        
print(f"Filtered {len(valid_nodes)} valid nodes from ENRE_JSON.")


print("Processing relations in ENRE_JSON...")
edges = []
seen_edges = set()
for cell in cells:
    src = cell.get('src')
    dest = cell.get('dest')
    values = cell.get('values', {})
    kind = values.get('kind')
    
    # Add all valid edges to edges
    if src in valid_nodes and dest in valid_nodes:
        edge_tuple = (src, dest, kind)
        if edge_tuple not in seen_edges:
            seen_edges.add(edge_tuple)
            edges.append((src, dest, kind))
        
print(f"Loaded {len(edges)} valid edges.")


print("Graph construction logic needs adjacency list...")
# Build adjacency for whole graph
adj = defaultdict(list)
for src, dest, kind in edges:
    adj[src].append((dest, kind))


def build_graph():
    print("Step 1: Loading FILTERED_PATH for task info...")
    tasks_info = []
    # Load all tasks for one project, one task per line
    with open(FILTERED_PATH, 'r') as f:
        for line in f:
            if line.strip():
                tasks_info.append(json.loads(line))
    print(f"Loaded {len(tasks_info)} tasks from FILTERED_PATH.")

    print("Step 2 Loading DIAGNOSTIC_JSONL and Building Graph...")
    diag_records = []
    # Load previous search results, including match indicators
    with open(DIAGNOSTIC_JSONL, 'r') as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    os.makedirs(OUTPUT_GRAPH_PATH, exist_ok=True)
        
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

        target_file = ""
        if target_method in method_sig_to_info:
            target_file = str(method_sig_to_info[target_method].get("func_file", ""))
        print(f"[task] example_id={example_id} target_method={target_method_clean} target_file={target_file}", flush=True)

        top_ks_raw = TOP_KS if isinstance(TOP_KS, (list, tuple, set)) else [TOP_KS]
        top_ks = []
        for k in top_ks_raw:
            try:
                k_int = int(k)
            except Exception:
                continue
            if k_int > 0:
                top_ks.append(k_int)
        top_ks = list(dict.fromkeys(top_ks))
        if not top_ks:
            top_ks = [10]

        for K in top_ks:
            preds = []
            try:
                # preds = rec["hybrid"]["recall_top7_clusters"][f"rank_top{K}"]["predictions"]
                preds = rec["feature"]["top3"]["predictions"]
                filtered_preds = [p for p in preds if (p['method'] if '(' not in p['method'] else p['method'].split('(')[0]) != target_method]
                if len(filtered_preds) != len(preds):
                    print(f"Attention! {len(preds) - len(filtered_preds)} items filtered out for task {example_id} (method == {target_method})")
                preds = filtered_preds
            except KeyError:
                print(f"Skipping task {example_id} for top-{K}: missing hybrid/recall_top7_clusters/rank_top{K}/predictions")
                continue

            pred_signatures = [p['method'] for p in preds]

            G = nx.DiGraph()

            relevant_ids = set()
            for sig in pred_signatures:
                if '(' in sig:
                    clean_sig = sig.split('(')[0]
                else:
                    clean_sig = sig

                if clean_sig in qname_to_id:
                    eid = qname_to_id[clean_sig]
                    relevant_ids.add(eid)
                    node_info = valid_nodes[eid]

                    attrs = {
                        'sig': clean_sig,
                        'category': node_info['category'],
                        'is_SameFile': False,
                        'is_SameFeature': False,
                    }

                    if sig in method_sig_to_info:
                        csv_info = method_sig_to_info[sig]
                        attrs['method_signature'] = str(csv_info.get('method_signature', ''))
                        attrs['func_file'] = str(csv_info.get('func_file', ''))
                        attrs['method_code'] = str(csv_info.get('method_code', ''))
                    else:
                        print(f"Warning: No method info found for {sig}")

                    G.add_node(eid, **attrs)

            for u in relevant_ids:
                if u in adj:
                    for v, kind in adj[u]:
                        if v in relevant_ids:
                            G.add_edge(u, v, type=kind)

            same_file_count = 0
            same_feature_count = 0
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
            print(
                f"[task] example_id={example_id} topK={K} same_file={same_file_count} same_feature={same_feature_count}",
                flush=True,
            )

            top_k_dir = os.path.join(OUTPUT_GRAPH_PATH, f"top-{K}-subgraph")
            os.makedirs(top_k_dir, exist_ok=True)
            out_file = os.path.join(top_k_dir, f"task_{example_id}_ori.gml")
            nx.write_gml(G, out_file)
            print(f"Saved graph for task {example_id} top-{K} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if __name__ == "__main__":
    build_graph()
