
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
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

import sys
import json
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch

# Ensure CodeContextSearch/src is on sys.path for namespace-package imports (graph.*, utils.*, etc.)
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from graph.embedding_backends import create_embedding_backend
from graph.class_code_extract import *

# Default path arguments, provided only as command-line conveniences for direct execution of this script;
# During batch experiments, pass these paths through function arguments.
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
OUTPUT_GRAPH_PATH = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/graphs'))
PROJECT_PATH = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', 'System/mrjob')) #mrjob
# PROJECT_PATH = "<package_root>/data/Datasets/Source_Code/Internet/boto" #boto
# PROJECT_PATH = "<package_root>/data/Datasets/Source_Code/Database/alembic"  #alembic
#PROJECT_PATH = "<package_root>/data/Datasets/Source_Code/Security/diffprivlib" #diffprivlib
# PROJECT_PATH = "<package_root>/data/Datasets/Source_Code/Multimedia/modipy" #modipy
TOP_KS = [10, 15, 20]

ENABLE_EXTRA_EXPANDED_NODE_BONUS = True

# Which embedding backend to use for personalization scores.
#   - "unixcoder": default, UniXcoder-based embeddings
#   - "bge-code": use BAAI/bge-code-v1 via sentence-transformers
EMBEDDING_BACKEND_KIND = "bge-code"

# Set True to print class skeleton extraction preview (file, class, length, first lines)
DEBUG_CLASS_SKELETON = False

def load_methods_csv(csv_path):
    print("Loading METHODS_CSV...")
    df = pd.read_csv(csv_path)
    # method_signature -> info
    mapping = {}
    clean_sig = ""
    for _, row in df.iterrows():
        sig = str(row['method_signature'])
        if '(' in sig:
            clean_sig = sig.split('(')[0]
        else:
            clean_sig = sig
        mapping[clean_sig] = row.to_dict()
    print(f"Loaded {len(mapping)} methods from CSV.")
    return mapping

def load_enre_json(json_path):
    print("Loading ENRE_JSON...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    variables = data.get('variables', [])
    cells = data.get('cells', [])
    
    valid_nodes = {} # id -> data
    qname_to_id = {} # id -> qualifiedName
    id_to_qname = {} # qualifiedName -> id
    
    for var in variables:
        if not var.get('category', '').startswith('Un'):
            vid = var['id']
            qname = var['qualifiedName']
            valid_nodes[vid] = var
            qname_to_id[qname] = vid
            id_to_qname[vid] = qname
            
    # Build adjacency
    adj = defaultdict(list) # src -> [(dest, kind)]
    reverse_adj = defaultdict(list) # dest -> [(src, kind)]
    seen_edges = set()

    for cell in cells:
        src = cell.get('src')
        dest = cell.get('dest')
        kind = cell.get('values', {}).get('kind')
        if src in valid_nodes and dest in valid_nodes:
            edge_tuple = (src, dest, kind)
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                adj[src].append((dest, kind))
                reverse_adj[dest].append((src, kind))
    
    print(f"Loaded {len(valid_nodes)} valid nodes and {len(seen_edges)} valid edges from ENRE_JSON.")
    return valid_nodes, qname_to_id, id_to_qname, adj, reverse_adj


def expand_graph(G, valid_nodes, adj, reverse_adj, method_map, id_to_qname, project_path: str):
    for node in G.nodes():
        if 'src_type' not in G.nodes[node]:
            G.nodes[node]['src_type'] = 'original'

    def preferred_id_type():
        for n in G.nodes():
            return str if isinstance(n, str) else int
        return str

    preferred_type = preferred_id_type()

    def node_key_for_enre_id(enre_id):
        if enre_id in G.nodes:
            return enre_id
        sid = str(enre_id)
        if sid in G.nodes:
            return sid
        return sid if preferred_type is str else enre_id

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def get_method_info(qname):
        if not qname:
            return {}
        if qname in method_map:
            return method_map[qname]
        base = qname.split('(')[0]
        if base in method_map:
            return method_map[base]
        short = base.split('.')[-1]
        return method_map.get(short, {})

    nodes_to_add = []
    edges_to_add = []
    added_node_keys = set()

    def ensure_function_node(enre_id):
        key = node_key_for_enre_id(enre_id)
        if key in G.nodes or key in added_node_keys:
            return key
        qname = id_to_qname.get(enre_id, "")
        method_info = get_method_info(qname)
        nodes_to_add.append((key, {
            'sig': qname,
            'category': 'Function',
            'src_type': 'expand',
            'method_signature': method_info.get('method_signature', ''),
            'func_file': method_info.get('func_file', ''),
            'method_code': method_info.get('method_code', '')
        }))
        added_node_keys.add(key)
        return key

    original_function_enre_ids = []
    for node_id, node_data in list(G.nodes(data=True)):
        if node_data.get('src_type') != 'original' or node_data.get('category') != 'Function':
            continue
        enre_id = safe_int(node_id)
        if enre_id is None or enre_id not in valid_nodes:
            continue
        original_function_enre_ids.append(enre_id)

    for enre_id in original_function_enre_ids:
        src_key = node_key_for_enre_id(enre_id)

        if enre_id in adj:
            for dest_id, kind in adj[enre_id]:
                if kind not in ['Call', 'Use', 'Contain']:
                    continue
                if dest_id not in valid_nodes or valid_nodes[dest_id].get('category') != 'Function':
                    continue
                dest_key = ensure_function_node(dest_id)
                edges_to_add.append((src_key, dest_key, {'kind': kind}))

        if enre_id in reverse_adj:
            for caller_id, kind in reverse_adj[enre_id]:
                if kind != 'Call':
                    continue
                if caller_id not in valid_nodes or valid_nodes[caller_id].get('category') != 'Function':
                    continue
                caller_key = ensure_function_node(caller_id)
                edges_to_add.append((caller_key, src_key, {'kind': kind}))

    for n, attr in nodes_to_add:
        if n not in G:
            G.add_node(n, **attr)

    for u, v, attr in edges_to_add:
        if not G.has_edge(u, v):
            G.add_edge(u, v, **attr)

    all_function_enre_ids = set()
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('category') != 'Function':
            continue
        enre_id = safe_int(node_id)
        if enre_id is None or enre_id not in valid_nodes:
            continue
        all_function_enre_ids.add(enre_id)

    class_nodes_to_add = []
    class_edges_to_add = []
    class_added_keys = set()

    def ensure_class_node_in_second_pass(enre_id):
        key = node_key_for_enre_id(enre_id)
        if key in G.nodes or key in class_added_keys:
            return key
        src_node_info = valid_nodes[enre_id]
        qname = id_to_qname.get(enre_id, "")
        file_path = src_node_info.get('File', "")
        class_code = get_class_code_default(project_path, file_path, qname) if file_path else ""
        if DEBUG_CLASS_SKELETON:
            if class_code:
                preview_lines = class_code.strip().split("\n")[:20]
                preview = "\n".join(preview_lines)
                print(f"[class_skeleton] file={file_path} class={qname} len={len(class_code)} chars")
                print(f"  --- preview ---\n{preview}\n  ---")
            else:
                print(f"[class_skeleton] file={file_path} class={qname} -> empty (file missing or class not found)")
        class_nodes_to_add.append((key, {
            'sig': qname,
            'category': 'Class',
            'src_type': 'expand',
            'method_signature': qname,
            'func_file': file_path,
            'method_code': class_code
        }))
        class_added_keys.add(key)
        return key

    for func_enre_id in all_function_enre_ids:
        func_key = node_key_for_enre_id(func_enre_id)
        if func_enre_id not in reverse_adj:
            continue
        for class_id, kind in reverse_adj[func_enre_id]:
            if kind != 'Define':
                continue
            if class_id not in valid_nodes or valid_nodes[class_id].get('category') != 'Class':
                continue
            class_key = ensure_class_node_in_second_pass(class_id)
            class_edges_to_add.append((class_key, func_key, {'kind': kind}))

    for n, attr in class_nodes_to_add:
        if n not in G:
            G.add_node(n, **attr)

    for u, v, attr in class_edges_to_add:
        if not G.has_edge(u, v):
            G.add_edge(u, v, **attr)

    return G

def get_graph_input_dirs(base_dir):
    try:
        children = [os.path.join(base_dir, name) for name in os.listdir(base_dir)]
    except FileNotFoundError:
        return []

    def has_ori_gml_files(dir_path):
        try:
            for name in os.listdir(dir_path):
                if name.endswith("_ori.gml"):
                    return True
        except Exception:
            return False
        return False

    subdirs = [p for p in children if os.path.isdir(p)]
    subdirs_with_ori = [p for p in subdirs if has_ori_gml_files(p)]
    if subdirs_with_ori:
        return sorted(subdirs_with_ori)

    if has_ori_gml_files(base_dir):
        return [base_dir]

    return sorted(subdirs) if subdirs else [base_dir]

def process_graph_dir(
    graph_dir,
    tasks,
    embedder,
    method_map,
    valid_nodes,
    id_to_qname,
    adj,
    reverse_adj,
    project_path: str,
):
    def is_truthy(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes", "y"}
        return False

    def normalize_sig(sig):
        s = "" if sig is None else str(sig)
        base = s.split("(", 1)[0] if "(" in s else s
        return base.lstrip(".")

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return None

    for i, task in enumerate(tasks):
        task_id = task.get('example_id', i + 1)
        query = task.get('requirement', {}).get('Functionality', '') + ' ' + task.get('requirement', {}).get('Arguments', '')

        dependency = task.get('dependency', {})
        gt_methods = []
        gt_methods.extend(dependency.get('intra_class', []))
        gt_methods.extend(dependency.get('intra_file', []))
        gt_methods.extend(dependency.get('cross_file', []))

        gml_filename = f"task_{task_id}_ori.gml"
        gml_path = os.path.join(graph_dir, gml_filename)

        if not os.path.exists(gml_path):
            print(f"Graph file not found for task {task_id}: {gml_path}")
            continue

        try:
            G = nx.read_gml(gml_path)
        except Exception as e:
            print(f"Error reading GML for task {task_id}: {e}")
            continue

        target = normalize_sig(task.get('namespace') or G.graph.get('target_method', ''))
        def exclude_target(graph):
            graph.remove_nodes_from([
                n for n, attrs in graph.nodes(data=True)
                if target and (
                    normalize_sig(attrs.get('sig') or attrs.get('method_signature')) == target
                    or normalize_sig(attrs.get('sig') or attrs.get('method_signature')).startswith(target + '.')
                )
            ])

        exclude_target(G)
        original_node_ids = set(G.nodes())

        print(f"Processing Task {task_id}: Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

        expanded_G = expand_graph(G, valid_nodes, adj, reverse_adj, method_map, id_to_qname, project_path)
        # Exclusion after PageRank is too late: target code and its edges would
        # already affect embeddings, restart scores and graph propagation.
        exclude_target(expanded_G)
        print(f"  -> Expanded: {len(expanded_G.nodes)} nodes and {len(expanded_G.edges)} edges.")

        mid_gml_filename = f"task_{task_id}_mid.gml"
        mid_gml_path = os.path.join(graph_dir, mid_gml_filename)
        nx.write_gml(expanded_G, mid_gml_path)
        print(f"  -> Saved expanded graph to {mid_gml_path}")

        # if i < 3:
        #     print(f"  -> Debug: First 3 expanded nodes for task {task_id}:")
        #     for n in list(expanded_G.nodes(data=True))[:10]:
        #         print(f"    - Node ID: {n[0]}, Attributes: {n[1]}")

        # Encode query and graph node code with the selected embedding backend.
        nl_emb = embedder.encode_query(query)

        node_list = list(expanded_G.nodes(data=True))
        node_codes = []
        node_ids = []

        for n, d in node_list:
            code = ""
            if 'method_code' in d:
                code = d['method_code']

            node_codes.append(code if code else "")
            node_ids.append(n)

        if node_codes:
            code_embs = embedder.encode_code(node_codes, batch_size=16)

            if code_embs.shape[0] > 0:
                sims = (
                    torch.mm(nl_emb, code_embs.t())
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                sims = np.zeros(len(node_codes))
        else:
            sims = np.zeros(len(node_codes))

        personalization = {}
        for idx, n in enumerate(node_ids):
            score = max(0.0, float(sims[idx]))
            personalization[n] = score

        boosted_nodes = 0
        boosted_total_delta = 0.0
        if ENABLE_EXTRA_EXPANDED_NODE_BONUS:
            for n in original_node_ids:
                attrs = expanded_G.nodes.get(n, {})
                inc = 0.0
                if is_truthy(attrs.get("is_SameFeature")):
                    inc += 0.1
                if is_truthy(attrs.get("is_SameFile")):
                    inc += 0.1
                if is_truthy(attrs.get("is_SimilarMethod")):
                    inc += 0.1
                if inc <= 0:
                    continue

                neighbors = set()
                try:
                    neighbors.update(expanded_G.successors(n))
                except Exception:
                    pass
                try:
                    neighbors.update(expanded_G.predecessors(n))
                except Exception:
                    pass

                for nb in neighbors:
                    if nb in original_node_ids:
                        continue
                    personalization[nb] = float(personalization.get(nb, 0.0)) + inc
                    boosted_nodes += 1
                    boosted_total_delta += inc

        if boosted_nodes > 0:
            print(
                f"  -> Personalization boosted: edges_to_expand_nodes={boosted_nodes} total_delta={boosted_total_delta:.4f}",
                flush=True,
            )

        total_score = sum(personalization.values())
        if total_score == 0:
            print("???")
            personalization = {n: 1.0/len(node_ids) for n in node_ids} if node_ids else {}

        try:
            ppr_scores = nx.pagerank(expanded_G, alpha=0.85, personalization=personalization)
        except Exception as e:
            print(f"  -> PageRank failed: {e}. Using similarity scores directly.")
            ppr_scores = personalization

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
            top_ks = [15]

        target_method_sig = normalize_sig(expanded_G.graph.get("target_method", G.graph.get("target_method", "")))
        target_method_enre_ids = set()
        if target_method_sig:
            for _eid, _qname in id_to_qname.items():
                if normalize_sig(_qname) == target_method_sig:
                    target_method_enre_ids.add(_eid)

        excluded_nodes = set()
        if target_method_sig or target_method_enre_ids:
            for node_id, node_data in expanded_G.nodes(data=True):
                enre_id = safe_int(node_id)
                if enre_id is not None and enre_id in target_method_enre_ids:
                    excluded_nodes.add(node_id)
                    continue
                if target_method_sig and normalize_sig(node_data.get("sig", "")) == target_method_sig:
                    excluded_nodes.add(node_id)

        if excluded_nodes:
            print(
                f"  -> Excluding target_method from saved subgraphs: target_method={target_method_sig} excluded_nodes={len(excluded_nodes)}",
                flush=True,
            )

        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        rank_gml_filename = f"task_{task_id}_rank.gml"

        for K in top_ks:
            top_k_nodes = [n for n, _ in sorted_nodes if n not in excluded_nodes][:K]

            subgraph = expanded_G.subgraph(top_k_nodes).copy()

            for node in subgraph.nodes():
                if node in ppr_scores:
                    subgraph.nodes[node]['score'] = ppr_scores[node]

            top_k_dir = os.path.join(graph_dir, f"PageRank-{K}-subgraph")
            os.makedirs(top_k_dir, exist_ok=True)
            rank_gml_path = os.path.join(top_k_dir, rank_gml_filename)
            nx.write_gml(subgraph, rank_gml_path)
            print(f"  -> Saved top-{K} subgraph to {rank_gml_path}")

def main():
    run_project(
        methods_csv=METHODS_CSV,
        enre_json=ENRE_JSON,
        filtered_path=FILTERED_PATH,
        output_graph_path=OUTPUT_GRAPH_PATH,
        project_path=PROJECT_PATH,
        top_ks=TOP_KS,
        embedding_backend_kind=EMBEDDING_BACKEND_KIND,
        enable_extra_expanded_node_bonus=ENABLE_EXTRA_EXPANDED_NODE_BONUS,
        debug_class_skeleton=DEBUG_CLASS_SKELETON,
    )


def run_project(
    *,
    methods_csv: str,
    enre_json: str,
    filtered_path: str,
    output_graph_path: str,
    project_path: str,
    top_ks: list[int] | None = None,
    embedding_backend_kind: str = EMBEDDING_BACKEND_KIND,
    enable_extra_expanded_node_bonus: bool = ENABLE_EXTRA_EXPANDED_NODE_BONUS,
    debug_class_skeleton: bool = DEBUG_CLASS_SKELETON,
) -> None:
    """
    对单个项目进行图扩展与排序（读取 build_graph 产出的 *_ori.gml），并输出 PageRank 子图 gml。
    """
    global TOP_KS, ENABLE_EXTRA_EXPANDED_NODE_BONUS, DEBUG_CLASS_SKELETON
    if top_ks is not None:
        TOP_KS = top_ks
    ENABLE_EXTRA_EXPANDED_NODE_BONUS = enable_extra_expanded_node_bonus
    DEBUG_CLASS_SKELETON = debug_class_skeleton

    embedder = create_embedding_backend(kind=embedding_backend_kind)
    os.makedirs(output_graph_path, exist_ok=True)

    method_map = load_methods_csv(methods_csv)
    valid_nodes, qname_to_id, id_to_qname, adj, reverse_adj = load_enre_json(enre_json)

    tasks = []
    print("Loading Tasks...")
    with open(filtered_path, "r") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks.")

    graph_dirs = get_graph_input_dirs(output_graph_path)
    for graph_dir in graph_dirs:
        if not os.path.isdir(graph_dir):
            continue
        print(f"Processing graph directory: {graph_dir}")

        process_graph_dir(
            graph_dir=graph_dir,
            tasks=tasks,
            embedder=embedder,
            method_map=method_map,
            valid_nodes=valid_nodes,
            id_to_qname=id_to_qname,
            adj=adj,
            reverse_adj=reverse_adj,
            project_path=project_path,
        )

if __name__ == "__main__":
    main()
