
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
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import json
import pandas as pd
from typing import Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from utils.query_refine import refine_query

PROJECT_PATH = "System/mrjob"
# PROJECT_PATH = "Internet/boto"
# PROJECT_PATH ="Database/alembic"

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/methods.csv" 
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/Filited/mrjob/filtered.jsonl" 
# refined_queries_cache_path= '<package_root>/results/repoSummaryOut/Filited/mrjob/refined_queries.json'
# ENRE_JSON = "<package_root>/results/repoSummaryOut/Filited/mrjob/mrjob-report-enre.json"

FEATURE_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/features.csv')) 
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv')) 
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl')) 
refined_queries_cache_path = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/refined_queries.json'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/methods.csv" 
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/Filited/mrjob/filtered.jsonl" 
# refined_queries_cache_path = '<package_root>/results/repoSummaryOut/Filited/mrjob/refined_queries.json' 
# ENRE_JSON = "<package_root>/results/repoSummaryOut/Filited/mrjob/mrjob-report-enre.json"
# Path to the DevEval case JSON file, not to the dataset repository itself
DATA_JSONL = str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl'))

# Whether method names should be normalized; for example, convert mrjob.mrjob.xx in the generated CSV to mrjob.xx for evaluation
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False

# Recall parameters
# RECALL_CLUSTER_K = 5  # Use top 5 clusters for recall
CLUSTER_KS = [5, 6, 7]
SIG_KS = [15, 20]

DEBUG_ON = False
DEBUG_EXAMPLE_ID = 32
DEBUG_CLUSTER_K = 5
DEBUG_SIG_K = 15

variables_enre = set()  # Variable type: count a hit when the retrieved code uses this variable
unresolved_attribute_enre = set()  # In ENRE, this type usually denotes a self.xxx attribute of a class; count a hit when the retrieved code contains self.xxx
module_enre = set()  # Module, effectively a Python file; standalone module names may appear in dependencies, and retrieving any element from that module counts as a hit
package_enre = set()  # Package: handled similarly to a module


# Read the ENRE result file, focusing on Variable, Unresolved Attribute, Module, and Package entries
def load_enre_elements(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: ENRE JSON file not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading ENRE JSON: {e}")
        return

    variables = data.get("variables", [])
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


def compute_task_recall(
    dependency: Optional[list[str]],
    searched_context_code_list: list[Dict[str, Any]],
) -> Dict[str, Any]:
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

    recall = (hit / dep_total) if dep_total > 0 else None
    return {
        "dependency_total": dep_total,
        "dependency_hit": hit,
        "recall": recall,
    }


def is_method_hit(method_signature: str, method_code: str, deps_list: list[str]) -> bool:
	norm_sig = _normalize_symbol(method_signature).replace(".__init__", "")
	for x in deps_list:
		if x == norm_sig:
			return True
		if x in variables_enre:
			var_name = x.split(".")[-1]
			if var_name and var_name in (method_code or ""):
				return True
		if x in unresolved_attribute_enre:
			attr_name = x.split('.')[-1]
			class_name = '.'.join(x.split('.')[:-1])
			if norm_sig.startswith(f"{class_name}.") and f"self.{attr_name}" in (method_code or ""):
				return True
		if x in module_enre:
			if norm_sig.startswith(x):
				return True
		if x in package_enre:
			if norm_sig.startswith(x):
				return True
	return False


def analyze_project(project_path):
    # Step 1: Filter by project_path
    print("Filtering data...")
    with open(DATA_JSONL, 'r') as infile, open(FILTERED_PATH, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == project_path:
                outfile.write(line)
    
    # Step 2: Load Features and Clusters
    print("Loading features...")
    df = pd.read_csv(
        FEATURE_CSV,
        dtype=str,
        keep_default_na=False,
        quoting=0,
        engine="python",
        on_bad_lines="skip"
    )
    clusters = df.groupby('id')['desc'].first().reset_index()
    
    # Extract all method_name values from clusters
    method_names = df['method_name'].str.split('(').str[0].unique().tolist()

    if NEED_METHOD_NAME_NORM:
        # Normalize method_name by removing arguments and retaining the part after the first dot, e.g., mrjob.hadoop.main -> hadoop.main
        base_names = df['method_name'].astype(str).str.split('(').str[0]
        df['method_name_norm'] = base_names.str.split('.', n=1).str[1].fillna(base_names)
        method_names = df['method_name_norm'].unique().tolist()
    print(f"Loaded {len(method_names)} unique method names from features.")
    print(method_names[:10])

    load_enre_elements(ENRE_JSON)

    print("Encoding clusters...")
    # # model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model_path = _model_location('FEATLENS_MINILM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

    if os.path.exists(model_path):
        print(f"找到本地模型: {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"未找到本地模型，尝试下载...")
        # If the resource is unavailable locally, try downloading it
        model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())

    # Step 3: Load Methods for BM25
    print("Loading methods for BM25...")
    methods_df = pd.read_csv(METHODS_CSV, dtype=str).fillna('')
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

    def tokenize_signature(sig: str):
        part = sig.split('.')[-1]
        part = part.replace('_', ' ')
        part = re.sub(r'[\(\),.:]', ' ', part)
        toks = re.findall(r'\w+', part.lower())
        return toks

    def tokenize_code(code: str):
        toks = re.findall(r'\w+', code.lower())
        return toks

    # Build BM25 on Code
    print("Building BM25 index on code...")
    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    methods_corpus_strings = methods_df['method_signature'].tolist()
    bm25_code = BM25Okapi(code_docs)
    method_sig_to_code = dict(
        zip(
            methods_df["method_signature"].astype(str).tolist(),
            methods_df["method_code"].astype(str).tolist(),
        )
    )

    # Build Mapping: Feature Method Name -> Indices in methods_df
    # This is needed to map the "Recalled" methods from Feature Search to the BM25 corpus
    print("Building mapping from Feature names to Method indices...")
    feature_name_to_indices = {}
    
    # Pre-compute valid indices for each feature method name
    # Note: This might be slow if done naively. We assume method_name is a substring of method_signature.
    
    # Optimization: iterate over methods_df once
    # For each method in methods_df, check which feature names it matches.
    # To avoid O(N*M), we can try to match by exact string if possible, or use the containment assumption.
    # Given the previous code used `any(dep in m for m in methods_k)`, where dep is from feature names.
    # We need to support the same logic.
    
    # Let's try to do it:
    # 1. Get all unique feature method names (un-normalized for matching with raw data if needed, or normalized?)
    # The 'deps' in ground truth are filtered by `method_names` (which are normalized if NEED_METHOD_NAME_NORM is True).
    # But `methods_from_top_k_clusters` returns `df['method_name']` which is NOT normalized in the dataframe itself (unless we updated it).
    # In original code: `df['method_name_norm']` was created but `methods_from_top_k_clusters` uses `df["method_name"]`.
    # So the recall returns un-normalized names.
    
    raw_feature_names = df['method_name'].unique().tolist()
    
    # We will map RAW feature names to method indices.
    # Because `methods_from_top_k_clusters` returns raw names.
    
    # To speed up: 
    # many raw_feature_names might be substrings of methods_corpus_strings.
    # Let's just iterate. For 10k methods and 2k features, 20M checks is fine in Python (~seconds).
    
    for idx, sig in enumerate(methods_corpus_strings):
        for fname in raw_feature_names:
            if fname in sig:
                if fname not in feature_name_to_indices:
                    feature_name_to_indices[fname] = []
                feature_name_to_indices[fname].append(idx)
                
    print(f"Mapping built for {len(feature_name_to_indices)} feature names.")

    # Load or initialize the query cache
    if os.path.exists(refined_queries_cache_path):
        with open(refined_queries_cache_path, 'r') as f:
            refined_queries_cache = json.load(f)
    else:
        refined_queries_cache = {}
        with open(refined_queries_cache_path, 'w') as f:
            json.dump(refined_queries_cache, f, indent=2)

    # Helper functions
    def methods_from_top_k_clusters(similarities, k):
        idx = np.argsort(similarities)[-k:][::-1]
        ids = clusters.iloc[idx]["id"].tolist()
        methods = []
        for cid in ids:
            methods.extend(df[df["id"] == cid]["method_name"].tolist())
        return methods
       

    def safe_div(a, b):
        return (a / b) if b != 0 else 0

    # Evaluation
    print("Starting Hybrid Search Evaluation...")
    with open(FILTERED_PATH, 'r') as f:
        lines = f.readlines()

    hybrid_records = []
    
    # Metrics containers
    # metrics structure: metrics[cluster_k][sig_k] = {"match": 0, "pred": 0}
    metrics = {}
    for ck in CLUSTER_KS:
        metrics[ck] = {}
        for sk in SIG_KS:
            metrics[ck][sk] = {"match": 0, "pred": 0}
            
    top_gt = 0
    example_counter = 0

    for line in tqdm(lines):
        example_counter += 1
        data = json.loads(line.strip())
        deps = []
        deps.extend(data['dependency']['intra_class'])
        deps.extend(data['dependency']['intra_file'])
        deps.extend(data['dependency']['cross_file'])
        
        #deps = [dep for dep in deps if (dep in method_names) or (dep in variables_enre)]
        top_gt += len(deps)

        # Feature Search Embedding
        original_query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']

        if USE_REFINED_QUERY:
            if original_query in refined_queries_cache:
                query = refined_queries_cache[original_query]
            else:
                modelname = "deepseek-v3.2"
                query = refine_query(original_query, modelname)
                refined_queries_cache[original_query] = query
                # Save immediately
                with open(refined_queries_cache_path, 'w') as f:
                    json.dump(refined_queries_cache, f, indent=2)
        else:
            query = original_query

        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]
        
        q_tokens = re.findall(r'\w+', query.lower())
        all_bm25_scores = bm25_code.get_scores(q_tokens)
        
        # Prepare record for this example
        record = {
            "example_id": example_counter,
            "query": query,
            "ground_truth": deps,
            "hybrid": {}
        }

        # Task, top_k cluster, and top_k method targeted by DEBUG
        # A run may test multiple cluster and method counts; DEBUG currently supports detailed output for one task and configuration
        debug_hit = (
            DEBUG_ON and 
            (DEBUG_EXAMPLE_ID is not None and example_counter == DEBUG_EXAMPLE_ID)
            and (DEBUG_CLUSTER_K is None or DEBUG_CLUSTER_K in CLUSTER_KS)
            and (DEBUG_SIG_K is None or DEBUG_SIG_K in SIG_KS)
        )

        # Iterate over different Cluster Recall Ks
        for ck in CLUSTER_KS:
            # 1. Feature Search Recall (Top ck clusters)
            recalled_raw_names = methods_from_top_k_clusters(similarities, ck)
            
            # Map to BM25 indices
            candidate_indices = set()
            for rname in recalled_raw_names:
                if rname in feature_name_to_indices:
                    candidate_indices.update(feature_name_to_indices[rname])
            
            candidate_indices = list(candidate_indices)
            
            # 2. BM25 Re-ranking on Code
            final_methods = []
            scored_candidates = []
            if candidate_indices:
                for idx in candidate_indices:
                    scored_candidates.append((all_bm25_scores[idx], methods_corpus_strings[idx]))
                
                # Sort by score desc
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Extract method signatures
                final_methods = [x[1] for x in scored_candidates]
            
            if debug_hit and ck == DEBUG_CLUSTER_K:
                print("\n" + "=" * 100)
                print(f"[DEBUG] example_id={example_counter} ck={ck} sk={DEBUG_SIG_K}")
                print(f"[DEBUG] query: {query}")
                print(f"[DEBUG] ground_truth({len(deps)}):")
                for d in deps:
                    print(f"  - {d}")

                top_cluster_idx = np.argsort(similarities)[-ck:][::-1]
                print(f"[DEBUG] feature_top{ck}_clusters:")
                for rank, cidx in enumerate(top_cluster_idx, start=1):
                    cid = str(clusters.iloc[cidx]["id"])
                    cdesc = str(clusters.iloc[cidx]["desc"])
                    sim = float(similarities[cidx])
                    methods_in_cluster = df[df["id"] == cid]["method_name"].tolist()
                    print(f"  [{rank}] cluster_id={cid} sim={sim:.6f}")
                    print(f"      feature_desc: {cdesc}")
                    print(f"      methods({len(methods_in_cluster)}):")
                    for m in methods_in_cluster:
                        print(f"        - {m}")

                unique_recalled = len(set(recalled_raw_names))
                print(f"[DEBUG] recalled_raw_method_names: total={len(recalled_raw_names)} unique={unique_recalled}")
                print(f"[DEBUG] bm25_candidates_in_corpus: {len(candidate_indices)}")

            # 3. Calculate metrics for different Signature Top-Ks (SIG_KS)
            record["hybrid"][f"recall_top{ck}_clusters"] = {}
            
            for sk in SIG_KS:
                mk = final_methods[:sk]
                
                searched_context_code_list = [
                    {"sig": _normalize_symbol(m), "method_signature": m, "method_code": method_sig_to_code.get(m, "")}
                    for m in mk
                ]
                recall_info = compute_task_recall(deps, searched_context_code_list)
                num_match = int(recall_info["dependency_hit"])
                num_pred = len(mk)
                
                metrics[ck][sk]["pred"] += num_pred
                metrics[ck][sk]["match"] += num_match
                
                # Per-example metrics
                num_gt_ex = int(recall_info["dependency_total"])
                p = safe_div(num_match, num_pred)
                r = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = safe_div(2 * p * r, p + r)
                
                record["hybrid"][f"recall_top{ck}_clusters"][f"rank_top{sk}"] = {
                    "metrics": {"P": p, "R": r, "F1": f1, "pred": num_pred, "match": num_match, "gt": num_gt_ex},
                    "predictions": [
                        {
                            "method": m,
                            "match": is_method_hit(m, method_sig_to_code.get(m, ""), deps)
                        }
                        for m in mk
                    ]
                }

                if debug_hit and ck == DEBUG_CLUSTER_K and sk == DEBUG_SIG_K:
                    print(f"[DEBUG] bm25_rerank_top{sk}:")
                    for rank, (score, sig) in enumerate(scored_candidates[:sk], start=1):
                        is_match = _normalize_symbol(sig) in deps
                        print(f"  [{rank:02d}] score={float(score):.6f} match={is_match} method={sig}")
                    print("=" * 100)
        
        hybrid_records.append(record)

    # Output results
    out_dir = os.path.dirname(FILTERED_PATH)
    hybrid_path = os.path.join(out_dir, "diagnostic_hybrid_feature_BM25Code.jsonl")
    with open(hybrid_path, "w", encoding="utf-8") as fo:
        for rec in hybrid_records:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nHybrid Search Results (Feature Recall + BM25 Code Re-rank):")
    print("Format: Recall Top-K Clusters -> Rank Top-K Results")
    print("="*60)
    
    for ck in CLUSTER_KS:
        print(f"--- Feature Search Recall: Top {ck} Clusters ---")
        for sk in SIG_KS:
            m = metrics[ck][sk]["match"]
            p = metrics[ck][sk]["pred"]
            
            precision = (safe_div(m, p)) * 100
            recall = (safe_div(m, top_gt)) * 100
            f1 = (2 * safe_div(m, p) * safe_div(m, top_gt) / (safe_div(m, p) + safe_div(m, top_gt))) * 100 if (safe_div(m, p) + safe_div(m, top_gt)) > 0 else 0
            
            print(f"  Rank Top {sk}:")
            print(f"    Match: {m}, Pred: {p}")
            print(f"    P={precision:.2f}%, R={recall:.2f}%, F1={f1:.2f}%")
        print("-" * 60)

    print(f"Hybrid analysis completed. Results saved to {hybrid_path}")

if __name__ == "__main__":
    analyze_project(PROJECT_PATH)
