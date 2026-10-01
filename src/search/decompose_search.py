
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
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi
from utils.query_decompose import decompose_query

# PROJECT_PATH = "Internet/boto"
PROJECT_PATH = "System/mrjob"
# PROJECT_PATH ="Database/alembic"


FEATURE_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/features.csv')) 
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv')) 
METHODS_DESC_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods_with_desc.csv'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl')) 
decomposed_queries_cache_path = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/decomposed_queries.json')) 

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/boto/boto_testAug/1122_codet5/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/boto/boto_testAug/1122_codet5/methods.csv" 
# METHODS_DESC_CSV = "<package_root>/results/repoSummaryOut/boto/boto_testAug/1122_codet5/methods_with_desc.csv"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/boto/boto_testAug/1122_codet5/filtered.jsonl"
# decomposed_queries_cache_path = '<package_root>/results/repoSummaryOut/boto/boto_testAug/1122_codet5/decomposed_queries.json' 

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/alembic/0.1_0.85_40/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/alembic/0.1_0.85_40/methods.csv" 
# METHODS_DESC_CSV = "<package_root>/results/repoSummaryOut/alembic/0.1_0.85_40/methods_with_desc.csv"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/alembic/0.1_0.85_40/filtered.jsonl" 
# decomposed_queries_cache_path = '<package_root>/results/repoSummaryOut/alembic/0.1_0.85_40/decomposed_queries.json' 

DATA_JSONL = str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl'))

NEED_METHOD_NAME_NORM = True

def analyze_project(project_path):
    # Step 1: Filter by project_path (Assuming this is already done or we reuse existing file)
    if not os.path.exists(FILTERED_PATH):
        with open(DATA_JSONL, 'r') as infile, open(FILTERED_PATH, 'w') as outfile:
            for line in infile:
                data = json.loads(line.strip())
                if data.get('project_path') == project_path:
                    outfile.write(line)
    
    # Load methods for normalization reference
    df = pd.read_csv(
        FEATURE_CSV,
        dtype=str,
        keep_default_na=False,
        quoting=0,
        engine="python",
        on_bad_lines="skip"
    )
    method_names = df['method_name'].str.split('(').str[0].unique().tolist()

    if NEED_METHOD_NAME_NORM:
        base_names = df['method_name'].astype(str).str.split('(').str[0]
        df['method_name_norm'] = base_names.str.split('.', n=1).str[1].fillna(base_names)
        method_names = df['method_name_norm'].unique().tolist()
    print(method_names[:30])

    # load methods with description corpus
    methods_desc_df = pd.read_csv(METHODS_DESC_CSV, dtype=str).fillna('')
    if 'func_desc' not in methods_desc_df.columns:
        raise ValueError("methods_with_desc.csv must contain 'func_desc' column")

    def tokenize_text(text: str):
        toks = re.findall(r'\w+', text.lower())
        return toks

    desc_docs = [tokenize_text(d) for d in methods_desc_df['func_desc'].tolist()]
    # methods_desc_corpus_strings should match what we want to return (e.g. func_fullName)
    methods_desc_corpus_strings = methods_desc_df['func_fullName'].tolist()

    bm25_desc = BM25Okapi(desc_docs)

    # Load or initialize the query cache
    if os.path.exists(decomposed_queries_cache_path):
        with open(decomposed_queries_cache_path, 'r') as f:
            decomposed_queries_cache = json.load(f)
    else:
        decomposed_queries_cache = {}
        with open(decomposed_queries_cache_path, 'w') as f:
            json.dump(decomposed_queries_cache, f, indent=2)

    with open(FILTERED_PATH, 'r') as f:
        example_counter = 0
        desc_records = []
        top_gt = 0

        # Different k values for sub-query retrieval
        sub_query_ks = [1, 2, 3]
        # Metrics storage: We want to measure performance for each 'k' used in sub-queries
        # Note: The final set size depends on number of sub-queries * k
        desc_metrics = {k: {"match": 0, "pred": 0} for k in sub_query_ks}

        def bm25_topk_strings(order, k, corpus_strings):
            idx = order[-k:][::-1]
            return [corpus_strings[i] for i in idx]

        for line in f:
            example_counter += 1
            data = json.loads(line.strip())
            deps = []
            deps.extend(data['dependency']['intra_class'])
            deps.extend(data['dependency']['intra_file'])
            deps.extend(data['dependency']['cross_file'])
            deps = [dep for dep in deps if dep in method_names]
            top_gt += len(deps)

            original_query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']

            if original_query in decomposed_queries_cache:
                sub_queries = decomposed_queries_cache[original_query]
                print("found in cache")
            else:
                modelname = "deepseek-v3.2"
                sub_queries = decompose_query(original_query, modelname)
                decomposed_queries_cache[original_query] = sub_queries
                with open(decomposed_queries_cache_path, 'w') as f:
                    json.dump(decomposed_queries_cache, f, indent=2)
            
            print("sub queries: ", sub_queries)

            # Perform search for each sub-query and aggregate
            for k in sub_query_ks:
                all_found_methods = set()
                for sq in sub_queries:
                    q_tokens = re.findall(r'\w+', sq.lower())
                    desc_scores = bm25_desc.get_scores(q_tokens)
                    desc_order = np.argsort(desc_scores)
                    # Retrieve top-k for this sub-query
                    found = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                    all_found_methods.update(found)
                
                # Convert set back to list for metrics
                m = list(all_found_methods)
                desc_metrics[k]["pred"] += len(m)
                desc_metrics[k]["match"] += sum(1 for dep in deps if any(dep in s for s in m))

            # Record detailed results (optional, mimicking original structure)
            desc_record = {
                "example_id": example_counter,
                "query": original_query,
                "sub_queries": sub_queries,
                "ground_truth": deps,
                "bm25_decomposed_desc": {}
            }
            
            for k in sub_query_ks:
                all_found_methods = set()
                for sq in sub_queries:
                    q_tokens = re.findall(r'\w+', sq.lower())
                    desc_scores = bm25_desc.get_scores(q_tokens)
                    desc_order = np.argsort(desc_scores)
                    found = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                    all_found_methods.update(found)
                
                mk = list(all_found_methods)
                num_pred = len(mk)
                num_match = sum(1 for dep in deps if any(dep in m for m in mk))
                num_gt = len(deps)
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = (num_match / num_gt) if num_gt > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                desc_record["bm25_decomposed_desc"][f"sub_top{k}"] = {
                    "metrics": {
                        "P": precision,
                        "R": recall,
                        "F1": f1,
                        "pred": num_pred,
                        "match": num_match,
                        "gt": num_gt
                    },
                    "predictions": [{"method": m, "match": any(dep in m for dep in deps)} for m in mk]
                }
            desc_records.append(desc_record)

        out_dir = os.path.dirname(FILTERED_PATH)
        desc_path = os.path.join(out_dir, "diagnostic_bm25_decomposed_desc.jsonl")
        with open(desc_path, "w", encoding="utf-8") as do:
            for rec in desc_records:
                do.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Save cache again to be sure
        with open(decomposed_queries_cache_path, 'w') as f:
            json.dump(decomposed_queries_cache, f, indent=4)

        def safe_div(a, b):
            return (a / b) if b != 0 else 0

        # ---- print BM25 decomposed desc-based metrics ----
        print("BM25 (decomposed desc) results:")
        for k in sub_query_ks:
            m = desc_metrics[k]["match"]
            p = desc_metrics[k]["pred"]
            print(f"Sub-Query Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={(2* safe_div(m, p) * safe_div(m, top_gt) / (safe_div(m, p) + safe_div(m, top_gt)))*100:.2f}%")
        print("--------------------------------")
    
    print(f"Analysis completed for {project_path}")

if __name__ == "__main__":
    analyze_project(PROJECT_PATH)
