
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
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
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

# PROJECT_PATH = "Internet/boto"
# PROJECT_PATH = "System/mrjob"
PROJECT_PATH ="Database/alembic"

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/mrjob/1:7.6/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/mrjob/1:7.6/methods.csv" 
# METHODS_DESC_CSV = "<package_root>/results/repoSummaryOut/mrjob/1:7.6/methods_with_desc.csv"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/mrjob/1:7.6/filtered.jsonl"
# refined_queries_cache_path = '<package_root>/results/repoSummaryOut/mrjob/1:3/refined_queries.json' 

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/boto/1:5/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/boto/1:5/methods.csv" 
# METHODS_DESC_CSV = "<package_root>/results/repoSummaryOut/boto/1:5/methods_with_desc.csv"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/boto/1:5/filtered.jsonl" 
# refined_queries_cache_path = '<package_root>/results/repoSummaryOut/boto/1:5/refined_queries.json'

FEATURE_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/features.csv')) 
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/methods.csv')) 
METHODS_DESC_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/methods_with_desc.csv'))
FILTERED_PATH = str(_package_path('data/deveval/projects/Database/alembic/filtered.jsonl')) 
refined_queries_cache_path = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/refined_queries.json')) 
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/Database/alembic/report-enre.json'))

# Path to the DevEval case JSON file, not to the dataset repository itself
DATA_JSONL = str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl'))

# Whether method names should be normalized; for example, convert mrjob.mrjob.xx in the generated CSV to mrjob.xx for evaluation
NEED_METHOD_NAME_NORM = True
USE_REFINED_QUERY = True

# Recall parameters
# RECALL_CLUSTER_K = 5  # Use top 5 clusters for recall
CLUSTER_KS = [1, 3, 5]
SIG_KS = [5, 10, 15]

variables_enre = set()


def load_enre_json(json_path: str) -> None:
    with open(json_path, "r") as f:
        data = json.load(f)

    variables = data.get("variables", [])
    for var in variables:
        if not var.get("category", "").startswith("Un") and var.get("category") == "Variable":
            qname = var.get("qualifiedName")
            if qname:
                variables_enre.add(qname)


def _normalize_symbol(s: str) -> str:
    if "(" in s:
        return s.split("(", 1)[0]
    return s


def compute_task_recall(
    dependency: Optional[list[str]],
    searched_context_code_list: list[Dict[str, Any]],
) -> Dict[str, Any]:
    dep = dependency or []
    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    dep_total = len(dep)
    hit = 0

    for x in dep:
        if x in retrieved_set:
            hit += 1
            continue
        if x in variables_enre:
            var_name = x.split(".")[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code") or ""
                if var_name in code_detail:
                    hit += 1
                    break

    recall = (hit / dep_total) if dep_total > 0 else None
    return {
        "dependency_total": dep_total,
        "dependency_hit": hit,
        "recall": recall,
    }


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

    load_enre_json(ENRE_JSON)

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

    # Step 3: Load Methods for BM25 (Description)
    print("Loading methods with description for BM25...")
    methods_desc_df = pd.read_csv(METHODS_DESC_CSV, dtype=str).fillna('')
    if 'func_desc' not in methods_desc_df.columns:
        raise ValueError("methods_with_desc.csv must contain 'func_desc' column")

    methods_df = pd.read_csv(METHODS_CSV, dtype=str).fillna('')
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")
    method_sig_to_code = dict(
        zip(
            methods_df["method_signature"].astype(str).tolist(),
            methods_df["method_code"].astype(str).tolist(),
        )
    )

    def tokenize_text(text: str):
        toks = re.findall(r'\w+', text.lower())
        return toks

    # Build BM25 on Description
    print("Building BM25 index on description...")
    desc_docs = [tokenize_text(d) for d in methods_desc_df['func_desc'].tolist()]
    methods_desc_corpus_strings = methods_desc_df['func_fullName'].tolist()
    bm25_desc = BM25Okapi(desc_docs)

    # Build Mapping: Feature Method Name -> Indices in methods_desc_df
    print("Building mapping from Feature names to Method indices...")
    feature_name_to_indices = {}
    
    raw_feature_names = df['method_name'].unique().tolist()
    
    # We will map RAW feature names to method indices.
    # Because `methods_from_top_k_clusters` returns raw names.
    
    for idx, sig in enumerate(methods_desc_corpus_strings):
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
    print("Starting Hybrid Search Evaluation (Feature + BM25 Desc)...")
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
        
        # Filter deps by the normalized method names (as in original code)
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
        all_bm25_scores = bm25_desc.get_scores(q_tokens)
        
        # Prepare record for this example
        record = {
            "example_id": example_counter,
            "query": query,
            "ground_truth": deps,
            "hybrid": {}
        }

        # Iterate over different Cluster Recall Ks
        for ck in CLUSTER_KS:
            # 1. Feature Search Recall (Top ck clusters)
            recalled_raw_names = methods_from_top_k_clusters(similarities, ck)
            
            # Map to BM25 indices
            candidate_indices = []
            for rname in recalled_raw_names:
                if rname in feature_name_to_indices:
                    candidate_indices.extend(feature_name_to_indices[rname])
            
            # 2. BM25 Re-ranking on Description
            final_methods = []
            if candidate_indices:
                scored_candidates = []
                for idx in candidate_indices:
                    scored_candidates.append((all_bm25_scores[idx], methods_desc_corpus_strings[idx]))
                
                # Sort by score desc
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Extract method signatures
                final_methods = [x[1] for x in scored_candidates]
            
            # 3. Calculate metrics for different Signature Top-Ks (SIG_KS)
            record["hybrid"][f"recall_top{ck}_clusters"] = {}
            
            for sk in SIG_KS:
                mk = final_methods[:sk]
                
                searched_context_code_list = [
                    {"method_signature": m, "method_code": method_sig_to_code.get(m, "")}
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
                    "predictions": [{"method": m, "match": (_normalize_symbol(m) in deps)} for m in mk]
                }
        
        hybrid_records.append(record)

    # Output results
    out_dir = os.path.dirname(FILTERED_PATH)
    hybrid_path = os.path.join(out_dir, "diagnostic_hybrid_desc.jsonl")
    with open(hybrid_path, "w", encoding="utf-8") as fo:
        for rec in hybrid_records:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nHybrid Search Results (Feature Recall + BM25 Desc Re-rank):")
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
