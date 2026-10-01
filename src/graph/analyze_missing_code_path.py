
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
import json
import pandas as pd
import networkx as nx
from collections import defaultdict, deque


# METHODS_CSV = "<package_root>/results/repoSummaryOut/mrjob/1:3/methods.csv"
# ENRE_JSON = "<package_root>/results/repoSummaryOut/mrjob/1:3/mrjob-report-enre.json"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/mrjob/1:3/filtered.jsonl"
# OUTPUT_GRAPH_PATH = "<package_root>/results/devEvalSearchOut/System_mrjob/0108/graph_results"

METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
OUTPUT_GRAPH_PATH = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/graphs'))

REMOVE_FIRST_DOT_PREFIX = False
PREFIX = "mrjob"  # When prefix removal is enabled, specify the project name as the prefix here

def load_methods_csv(csv_path):
    df = pd.read_csv(csv_path)
    # method_signature -> info
    mapping = {}
    for _, row in df.iterrows():
        mapping[row['method_signature']] = row.to_dict()
    return mapping

def _keep_enre_node(category: str) -> bool:
    if not category:
        return False
    if category.startswith("Un"):
        return False
    if category in {"Package", "Module"}:
        return False
    return True

def load_enre_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    variables = data.get('variables', [])
    cells = data.get('cells', [])
    
    valid_nodes = {} # id -> data
    qname_to_id = {} # id -> qualifiedName
    id_to_qname = {} # qualifiedName -> id
    
    for var in variables:
        cat = var.get('category', '')
        if not _keep_enre_node(cat):
            continue
        vid = str(var['id'])
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
        if src is None or dest is None:
            continue
        src = str(src)
        dest = str(dest)
        if src in valid_nodes and dest in valid_nodes:
            edge_tuple = (src, dest, kind)
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                adj[src].append((dest, kind))
                reverse_adj[dest].append((src, kind))
            
    return valid_nodes, qname_to_id, id_to_qname, adj, reverse_adj

def normalize_signature(sig, remove_prefix=False):    
    clean = sig.split('(')[0]
    if remove_prefix and '.' in sig:
        # Remove first segment
        parts = sig.split('.')
        if len(parts) > 1:
            clean = '.'.join(parts[1:])
    return clean

def _normalize_qname_for_enre_lookup(qname: str) -> str:
    if not qname:
        return qname
    qname = normalize_signature(qname, remove_prefix=False)
    parts = qname.split(".")
    if len(parts) >= 2 and parts[0] == parts[1]:
        qname = ".".join([parts[0]] + parts[2:])
    return qname

def resolve_enre_id(missing_qname: str, qname_to_id: dict) -> str | None:
    if not missing_qname:
        return None

    missing_qname = _normalize_qname_for_enre_lookup(missing_qname)

    if missing_qname in qname_to_id:
        return qname_to_id[missing_qname]

    if PREFIX and not missing_qname.startswith(PREFIX + "."):
        prefixed = PREFIX + "." + missing_qname
        prefixed = _normalize_qname_for_enre_lookup(prefixed)
        if prefixed in qname_to_id:
            return qname_to_id[prefixed]

    if PREFIX and missing_qname.startswith(PREFIX + "."):
        unprefixed = missing_qname[len(PREFIX) + 1:]
        unprefixed = _normalize_qname_for_enre_lookup(unprefixed)
        if unprefixed in qname_to_id:
            return qname_to_id[unprefixed]

    best_eid = None
    best_extra_segments = None
    for qname, eid in qname_to_id.items():
        if qname == missing_qname or qname.endswith("." + missing_qname):
            extra = len(qname.split(".")) - len(missing_qname.split("."))
            if best_extra_segments is None or extra < best_extra_segments:
                best_extra_segments = extra
                best_eid = eid
    return best_eid

def build_enre_digraph(valid_nodes: dict, adj: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, node_data in valid_nodes.items():
        G.add_node(nid, qualifiedName=node_data.get("qualifiedName"), category=node_data.get("category"))

    for src, outs in adj.items():
        for dest, kind in outs:
            data = G.get_edge_data(src, dest, default=None)
            if data is None:
                G.add_edge(src, dest, kinds=(kind,) if kind is not None else tuple())
            else:
                kinds = set(data.get("kinds") or ())
                if kind is not None:
                    kinds.add(kind)
                G[src][dest]["kinds"] = tuple(sorted(kinds))
    return G

def build_undirected_adj_from_digraph(G: nx.DiGraph) -> dict:
    undirected_adj = defaultdict(list)
    for u, v in G.edges():
        undirected_adj[u].append(v)
        undirected_adj[v].append(u)
    return undirected_adj

def multi_source_shortest_path(undirected_adj: dict, sources: set, target: str) -> list[str] | None:
    if target in sources:
        return [target]

    visited = set()
    parent = {}
    q = deque()

    for s in sources:
        visited.add(s)
        parent[s] = None
        q.append(s)

    while q:
        cur = q.popleft()
        for nxt in undirected_adj.get(cur, ()):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = cur
            if nxt == target:
                q.clear()
                break
            q.append(nxt)

    if target not in parent:
        return None

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def path_to_pattern(path: list[str], enre_digraph: nx.DiGraph) -> str:
    if not path or len(path) == 1:
        return ""

    tokens = []
    for u, v in zip(path, path[1:]):
        u_cat = enre_digraph.nodes[u].get("category") if u in enre_digraph.nodes else None
        v_cat = enre_digraph.nodes[v].get("category") if v in enre_digraph.nodes else None
        u_cat = u_cat or "Unknown"
        v_cat = v_cat or "Unknown"

        uv = enre_digraph.get_edge_data(u, v)
        vu = enre_digraph.get_edge_data(v, u)

        has_uv = uv is not None
        has_vu = vu is not None
        if has_uv and has_vu:
            arrow_left, arrow_right = "<->", "<->"
        elif has_uv:
            arrow_left, arrow_right = "-(", ")->"
        elif has_vu:
            arrow_left, arrow_right = "<-(", ")-"
        else:
            arrow_left, arrow_right = "-(", ")?"

        kinds = set()
        if uv:
            kinds.update(uv.get("kinds") or ())
        if vu:
            kinds.update(vu.get("kinds") or ())
        kind_str = "|".join(sorted(kinds)) if kinds else "Unknown"
        if arrow_left == "<->":
            tokens.append(f"{u_cat} <->({kind_str})<-> {v_cat}")
        else:
            tokens.append(f"{u_cat} {arrow_left}{kind_str}{arrow_right} {v_cat}")

    return " / ".join(tokens)

def main():
    print("Loading Data...")
    method_map = load_methods_csv(METHODS_CSV)
    enre_nodes, qname_to_id, id_to_qname, adj, reverse_adj = load_enre_json(ENRE_JSON)
    enre_digraph = build_enre_digraph(enre_nodes, adj)
    undirected_adj = build_undirected_adj_from_digraph(enre_digraph)
    
    tasks = []
    with open(FILTERED_PATH, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
                
    print(f"Loaded {len(tasks)} tasks.")

    
    # Count missing code elements and the shortest-path patterns that can recover them; keys are normalized path forms and values are counts, with frequent patterns summarized afterward
    missing_path_pattern_to_cnt = {}
    missing_shortest_len_to_cnt = {}
    ground_truth_count = 0
    match_count = 0
    missing_in_enre_count = 0
    missing_unreachable_count = 0
    missing_reachable_count = 0


    
    # Iterate tasks
    for i, task in enumerate(tasks):
        task_id = task.get('example_id', i + 1)
        
        # Ground Truth
        dependency = task.get('dependency', {})
        gt_methods = []
        gt_methods.extend(dependency.get('intra_class', []))
        gt_methods.extend(dependency.get('intra_file', []))
        gt_methods.extend(dependency.get('cross_file', []))

        ground_truth_count += len(gt_methods)
        
        # Load Graph
        gml_filename = f"task_{task_id}_ori.gml"
        gml_path = os.path.join(OUTPUT_GRAPH_PATH, gml_filename)
        if not os.path.exists(gml_path):            
            print(f"Graph file not found for task {task_id}: {gml_path}")
            continue
        
        try:
            G = nx.read_gml(gml_path)
        except Exception as e:
            print(f"Error reading GML for task {task_id}: {e}")
            continue
            
        # Get graph nodes' signatures
        graph_sigs = set()
        node_id_map = {} # normalized_sig -> graph_node_id (ENRE ID)
        
        for node_id, data in G.nodes(data=True):
            sig = data.get('sig')
            # Normalize
            norm_sig = normalize_signature(sig, remove_prefix=REMOVE_FIRST_DOT_PREFIX)
            graph_sigs.add(norm_sig)
            
            # We assume the node ID in NetworkX IS the ENRE ID (as I implemented in build_graph)
            # G.nodes() yields the ID.
            node_id_map[norm_sig] = node_id
            
        # Compare GT with Graph
        missing_methods = []
        for gt in gt_methods:            
            if gt not in graph_sigs:
                missing_methods.append(gt)
                
        print(f"Task {task_id}: {len(missing_methods)} missing methods out of {len(gt_methods)} GT.")
        match_count += len(gt_methods) - len(missing_methods)
        
        if not missing_methods:
            continue

        # Obtain the ENRE IDs of preliminary search results
        source_node_ids = {str(nid) for nid in G.nodes()}
        source_node_ids = {nid for nid in source_node_ids if nid in enre_digraph}
            
        # Analyze missing methods
        for missing in missing_methods:
            missing_lookup = normalize_signature(missing, remove_prefix=False)
            if REMOVE_FIRST_DOT_PREFIX:
                missing_lookup = PREFIX + '.' + missing_lookup

            if missing in qname_to_id:
                found_eid = qname_to_id[missing]
            else:
                print(f"  [Missing] {missing} -> Could not find in ENRE.")
                missing_in_enre_count += 1
                continue

            path = multi_source_shortest_path(undirected_adj, source_node_ids, found_eid)
            if path is None:
                print(f"  [Missing] {missing} -> ENRE ID {found_eid} ({id_to_qname.get(found_eid, '')}) unreachable from searched nodes.")
                missing_unreachable_count += 1
                continue

            pattern = path_to_pattern(path, enre_digraph)
            length = max(len(path) - 1, 0)
            missing_shortest_len_to_cnt[length] = missing_shortest_len_to_cnt.get(length, 0) + 1
            missing_path_pattern_to_cnt[pattern] = missing_path_pattern_to_cnt.get(pattern, 0) + 1
            missing_reachable_count += 1
            print(f"  [Missing] {missing} -> ENRE ID {found_eid} shortest_len={length} pattern={pattern}")
                

    print(f"Total ground truth methods: {ground_truth_count}")
    print(f"Matched methods: {match_count}")
    print(f"Missing methods not found in ENRE: {missing_in_enre_count}")
    print(f"Missing methods unreachable from searched nodes: {missing_unreachable_count}")
    print(f"Missing methods reachable via shortest path: {missing_reachable_count}")

    print("Shortest path length distribution:")
    for k in sorted(missing_shortest_len_to_cnt.keys()):
        print(f"  {k}: {missing_shortest_len_to_cnt[k]}")

    print("Top path patterns:")
    for pattern, cnt in sorted(missing_path_pattern_to_cnt.items(), key=lambda x: x[1], reverse=True)[:50]:
        print(f"  {cnt}: {pattern}")

if __name__ == "__main__":
    main()
