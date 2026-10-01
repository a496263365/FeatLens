
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
from utils.query_refine import refine_query

# PROJECT_PATH = "System/mrjob"
# PROJECT_PATH = "Internet/boto"
# PROJECT_PATH ="Database/alembic"
# PROJECT_PATH = "Multimedia/Mopidy"
# PROJECT_PATH = "Security/diffprivlib"
# PROJECT_PATH = "Security/diffprivlib"

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/methods.csv" 
# METHODS_DESC_CSV = "<package_root>/results/repoSummaryOut/Filited/mrjob/methods_with_desc.csv"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/Filited/mrjob/filtered.jsonl" 
# refined_queries_cache_path = '<package_root>/results/repoSummaryOut/Filited/mrjob/refined_queries.json' 

# FEATURE_CSV = "<package_root>/results/repoSummaryOut/Filited/boto/features.csv" 
# METHODS_CSV = "<package_root>/results/repoSummaryOut/Filited/boto/methods.csv" 
# METHODS_DESC_CSV = "<package_root>/results/repoSummaryOut/Filited/boto/methods_with_desc.csv"
# FILTERED_PATH = "<package_root>/results/repoSummaryOut/Filited/boto/filtered.jsonl"
# refined_queries_cache_path = '<package_root>/results/repoSummaryOut/Filited/boto/refined_queries.json' 
# ENRE_JSON = "<package_root>/results/repoSummaryOut/Filited/boto/boto-report-enre.json"

FEATURE_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/features.csv')) 
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv')) 
METHODS_DESC_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods_with_desc.csv'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl')) 
refined_queries_cache_path = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/refined_queries.json')) 
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
# Path to the DevEval case JSON file, not to the dataset repository itself
# DATA_JSONL = "<package_root>/data/Datasets/data_have_dependency_cross_file.jsonl"
# Path to the complete dataset JSONL
DATA_JSONL = str(_package_path('scripts/metrics/deveval/data.jsonl'))


# Whether method names should be normalized; for example, convert mrjob.mrjob.xx in the generated CSV to mrjob.xx for evaluation
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False
TOP_KS = [1, 2, 3]# Controls the number of similar_methods in BM25 and feature-search methods
CLUSTER_KS = [1, 3, 5]# Controls the feature-search method


variables_enre = set()  # Variable type: count a hit when the retrieved code uses this variable
unresolved_attribute_enre = set()  # In ENRE, this type usually denotes a self.xxx attribute of a class; count a hit when the retrieved code contains self.xxx
module_enre = set()  # Module, effectively a Python file; standalone module names may appear in dependencies, and retrieving any element from that module counts as a hit
package_enre = set()  # Package: handled similarly to a module


def load_enre_elements(json_path):
    """读取enre的解析结果文件，重点读取Variable, Unresolved Attribute, Module, Package类型"""
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


def _maybe_insert_init_in_target_method(target_method: str) -> str:
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

def analyze_project(project_path):
    # Create output directory
    # output_dir = project_path
    # os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Filter by project_path
    with open(DATA_JSONL, 'r') as infile, open(FILTERED_PATH, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == project_path:
                outfile.write(line)
    
    # Step 2: Find similar clusters
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
    #print(method_names[:30])

    method_norm_to_feature_id = {}
    for fid, m in zip(df["id"].astype(str).tolist(), df["method_name"].astype(str).tolist()):
        m_norm = _normalize_symbol(m)
        if m_norm and m_norm not in method_norm_to_feature_id:
            method_norm_to_feature_id[m_norm] = fid

    # #model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # Check whether the path exists
    model_path = _model_location('FEATLENS_MINILM_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

    if os.path.exists(model_path):
        print(f"找到本地模型: {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"未找到本地模型，尝试下载...")
        # If the resource is unavailable locally, try downloading it
        model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())

    # load methods corpus
    methods_df = pd.read_csv(METHODS_CSV, dtype=str).fillna('')
    # ensure columns exist
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

    # load methods with description corpus
    methods_desc_df = pd.read_csv(METHODS_DESC_CSV, dtype=str).fillna('')
    if 'func_desc' not in methods_desc_df.columns:
        raise ValueError("methods_with_desc.csv must contain 'func_desc' column")

    def tokenize_signature(sig: str):
        # keep part after last '.', remove punctuation, replace underscores with spaces, split into tokens
        part = sig.split('.')[-1]
        part = part.replace('_', ' ')
        part = re.sub(r'[\(\),.:]', ' ', part)
        toks = re.findall(r'\w+', part.lower())
        return toks

    def tokenize_code(code: str):
        toks = re.findall(r'\w+', code.lower())
        return toks

    def tokenize_text(text: str):
        toks = re.findall(r'\w+', text.lower())
        return toks

    signature_docs = [tokenize_signature(s) for s in methods_df['method_signature'].tolist()]
    print(signature_docs[:10])
    print("=======================================")
    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    print(code_docs[:10])
    print("=======================================")
    desc_docs = [tokenize_text(d) for d in methods_desc_df['func_desc'].tolist()]
    print(desc_docs[:10])
    # store the original method strings to return as predicted items
    methods_corpus_strings = methods_df['method_signature'].tolist()
    # methods_desc_df['func_fullName'] and methods_df['method_signature'] are normally identical, but keep the former separate to handle unexpected differences
    methods_desc_corpus_strings = methods_desc_df['func_fullName'].tolist()

    bm25_sig = BM25Okapi(signature_docs)
    bm25_code = BM25Okapi(code_docs)
    bm25_desc = BM25Okapi(desc_docs)

    load_enre_elements(ENRE_JSON)

    method_sig_to_code = dict(
        zip(
            methods_df["method_signature"].astype(str).tolist(),
            methods_df["method_code"].astype(str).tolist(),
        )
    )
    normalized_sig_to_signature = {}
    for sig in method_sig_to_code.keys():
        norm = _normalize_symbol(sig)
        if norm not in normalized_sig_to_signature:
            normalized_sig_to_signature[norm] = sig

    def resolve_method_signature(method_str: str) -> str:
        if method_str in method_sig_to_code:
            return method_str
        norm = _normalize_symbol(method_str)
        if norm in normalized_sig_to_signature:
            return normalized_sig_to_signature[norm]
        return method_str

    def build_context_code_list(method_strs: list[str]) -> list[Dict[str, Any]]:
        ctx = []
        for m in method_strs:
            sig = resolve_method_signature(m)
            ctx.append(
                {
                    "sig": _normalize_symbol(sig),
                    "method_signature": sig,
                    "method_code": method_sig_to_code.get(sig, ""),
                }
            )
        return ctx

    def filter_out_target_method(method_strs: list[str], target_method_str: str) -> list[str]:
        target_norm = _normalize_symbol(target_method_str)
        return [m for m in method_strs if _normalize_symbol(m) != target_norm]

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

    # Load or initialize the query cache
    if os.path.exists(refined_queries_cache_path):
        with open(refined_queries_cache_path, 'r') as f:
            refined_queries_cache = json.load(f)
    else:
        refined_queries_cache = {}
        with open(refined_queries_cache_path, 'w') as f:
            json.dump(refined_queries_cache, f, indent=2)

    with open(FILTERED_PATH, 'r') as f:
        example_counter = 0
        feature_records = []
        sig_records = []
        code_records = []
        desc_records = []
        top_gt = 0

        cluster_metrics = {k: {"match": 0, "pred": 0} for k in CLUSTER_KS}

        sig_ks = [5, 10, 15 ,20]
        sig_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        code_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}
        desc_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        def methods_from_top_k_clusters(similarities, k):
            idx = np.argsort(similarities)[-k:][::-1]
            ids = clusters.iloc[idx]["id"].tolist()
            methods = []
            for cid in ids:
                methods.extend(df[df["id"] == cid]["method_name"].tolist())
            return methods

        def bm25_topk_strings(order, k, corpus_strings):
            idx = order[-k:][::-1]
            return [corpus_strings[i] for i in idx]

        for line in f:
            example_counter += 1
            data = json.loads(line.strip())
            deps = []
            target_method = data.get("namespace") or ""
            target_method_norm = _normalize_symbol(target_method)
            target_feature_id = method_norm_to_feature_id.get(target_method_norm)
            if target_feature_id is None:
                target_method_with_init = _maybe_insert_init_in_target_method(target_method_norm)
                if target_method_with_init != target_method_norm:
                    target_feature_id = method_norm_to_feature_id.get(target_method_with_init)
            target_feature_other_methods = []
            if target_feature_id is not None:
                target_feature_methods = (
                    df.loc[df["id"].astype(str) == str(target_feature_id), "method_name"]
                    .astype(str)
                    .tolist()
                )
                target_feature_other_methods = [
                    _normalize_symbol(m) for m in target_feature_methods if _normalize_symbol(m) != target_method_norm
                ]

            target_method_sig = resolve_method_signature(target_method)
            target_method_code = method_sig_to_code.get(target_method_sig, "")
            if not target_method_code:
                target_method_with_init = _maybe_insert_init_in_target_method(target_method_norm)
                if target_method_with_init != target_method_norm:
                    target_method_sig_with_init = resolve_method_signature(target_method_with_init)
                    target_method_code = method_sig_to_code.get(target_method_sig_with_init, "")
                    if target_method_code:
                        target_method_sig = target_method_sig_with_init
            similar_methods = {}
            target_code_tokens = tokenize_code(target_method_code) if target_method_code else []
            if target_code_tokens:
                target_code_scores = bm25_code.get_scores(target_code_tokens)
                target_code_order = np.argsort(target_code_scores)
                target_method_norm = _normalize_symbol(target_method)
                for k in TOP_KS:
                    k_int = int(k)
                    selected = []
                    seen = set()
                    for idx in target_code_order[::-1]:
                        m = methods_corpus_strings[int(idx)]
                        m_norm = _normalize_symbol(m)
                        if m_norm == target_method_norm:
                            continue
                        if m_norm in seen:
                            continue
                        selected.append(m_norm)
                        seen.add(m_norm)
                        if len(selected) >= k_int:
                            break
                    similar_methods[f"top{k_int}"] = selected
            else:
                for k in TOP_KS:
                    similar_methods[f"top{int(k)}"] = []
            # Extract ground-truth dependencies from the data; these are the correct answers for this search
            deps.extend(data['dependency']['intra_class'])
            deps.extend(data['dependency']['intra_file'])
            deps.extend(data['dependency']['cross_file'])
            # print("deps",deps)
            # Cleaning/filtering
            # deps = [dep for dep in deps if (dep in method_names) or (dep in variables_enre)]
            # print("deps after filter",deps)
            # input("please confirm the deps!")
            # Add the number of correct answers for this test case to the total
            top_gt += len(deps)

            # feature-based search
            original_query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
            #print("original query: ", original_query)

            if USE_REFINED_QUERY:
                if original_query in refined_queries_cache:
                    query = refined_queries_cache[original_query]
                    print("found in cache")
                else:
                    modelname = "deepseek-v3.2"
                    query = refine_query(original_query, modelname)
                    refined_queries_cache[original_query] = query
                    # Important: save immediately after appending.
                    with open(refined_queries_cache_path, 'w') as f:
                        json.dump(refined_queries_cache, f, indent=2)
                print("refined query: ", query)
            else:
                query = original_query
                print("Using original query: ", query)
            #input("please confirm the query!")
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]

            for k in CLUSTER_KS:
                methods_k = methods_from_top_k_clusters(similarities, k)
                methods_k = filter_out_target_method(methods_k, target_method)
                cluster_metrics[k]["pred"] += len(methods_k)
                cluster_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(methods_k))["dependency_hit"]

            # signature-based search using BM25
            q_tokens = re.findall(r'\w+', query.lower())
            sig_scores = bm25_sig.get_scores(q_tokens)
            sig_order = np.argsort(sig_scores)
            for k in sig_ks:
                m = bm25_topk_strings(sig_order, k, methods_corpus_strings)
                m = filter_out_target_method(m, target_method)
                sig_metrics[k]["pred"] += len(m)
                sig_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            # code-based search using BM25
            code_scores = bm25_code.get_scores(q_tokens)
            code_order = np.argsort(code_scores)
            for k in sig_ks:
                m = bm25_topk_strings(code_order, k, methods_corpus_strings)
                m = filter_out_target_method(m, target_method)
                code_metrics[k]["pred"] += len(m)
                code_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            # desc-based search using BM25
            desc_scores = bm25_desc.get_scores(q_tokens)
            desc_order = np.argsort(desc_scores)
            for k in sig_ks:
                m = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                m = filter_out_target_method(m, target_method)
                desc_metrics[k]["pred"] += len(m)
                desc_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            feature_record = {
                "example_id": example_counter,
                "query": query,
                "target_method": target_method,
                "similar_methods": similar_methods,
                "target_feature_id": target_feature_id,
                "target_feature_other_methods": target_feature_other_methods,
                "ground_truth": deps,
                "feature": {}
            }
            for k in CLUSTER_KS:
                mk = methods_from_top_k_clusters(similarities, k)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                feature_record["feature"][f"top{k}"] = {
                    "metrics": {
                        "P": precision,
                        "R": recall,
                        "F1": f1,
                        "pred": num_pred,
                        "match": num_match,
                        "gt": num_gt
                    },
                    "predictions": [
                        {
                            "method": m,
                            "match": is_method_hit(
                                resolve_method_signature(m),
                                method_sig_to_code.get(resolve_method_signature(m), ""),
                                deps,
                            ),
                        }
                        for m in mk
                    ]
                }
            feature_records.append(feature_record)

            sig_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_signature": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(sig_order, k,methods_corpus_strings)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                sig_record["bm25_signature"][f"top{k}"] = {
                    "metrics": {
                        "P": precision,
                        "R": recall,
                        "F1": f1,
                        "pred": num_pred,
                        "match": num_match,
                        "gt": num_gt
                    },
                    "predictions": [
                        {
                            "method": m,
                            "match": is_method_hit(m, method_sig_to_code.get(m, ""), deps),
                        }
                        for m in mk
                    ]
                }
            sig_records.append(sig_record)

            code_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_code": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(code_order, k, methods_corpus_strings)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                code_record["bm25_code"][f"top{k}"] = {
                    "metrics": {
                        "P": precision,
                        "R": recall,
                        "F1": f1,
                        "pred": num_pred,
                        "match": num_match,
                        "gt": num_gt
                    },
                    "predictions": [
                        {
                            "method": m,
                            "match": is_method_hit(m, method_sig_to_code.get(m, ""), deps),
                        }
                        for m in mk
                    ]
                }
            code_records.append(code_record)

            desc_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_desc": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                desc_record["bm25_desc"][f"top{k}"] = {
                    "metrics": {
                        "P": precision,
                        "R": recall,
                        "F1": f1,
                        "pred": num_pred,
                        "match": num_match,
                        "gt": num_gt
                    },
                    "predictions": [
                        {
                            "method": m,
                            "match": is_method_hit(
                                resolve_method_signature(m),
                                method_sig_to_code.get(resolve_method_signature(m), ""),
                                deps,
                            ),
                        }
                        for m in mk
                    ]
                }
            desc_records.append(desc_record)

        out_dir = os.path.dirname(FILTERED_PATH)
        feature_path = os.path.join(out_dir, "diagnostic_feature.jsonl")
        sig_path = os.path.join(out_dir, "diagnostic_bm25_signature.jsonl")
        code_path = os.path.join(out_dir, "diagnostic_bm25_code.jsonl")
        desc_path = os.path.join(out_dir, "diagnostic_bm25_desc.jsonl")
        with open(feature_path, "w", encoding="utf-8") as fo:
            for rec in feature_records:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(sig_path, "w", encoding="utf-8") as so:
            for rec in sig_records:
                so.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(code_path, "w", encoding="utf-8") as co:
            for rec in code_records:
                co.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(desc_path, "w", encoding="utf-8") as do:
            for rec in desc_records:
                do.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Save the updated cache
        with open(refined_queries_cache_path, 'w') as f:
            json.dump(refined_queries_cache, f, indent=4)

        # ===================== DIAGNOSTIC JSON CODE END =====================
        def safe_div(a, b):
            return (a / b) if b != 0 else 0

        for k in CLUSTER_KS:
            m = cluster_metrics[k]["match"]
            p = cluster_metrics[k]["pred"]
            print(f"Top {k} Match: {m}")
            print(f"Top {k} Pred: {p}")
            print(f"Top {k} P={(safe_div(m, p))*100:.2f}%")
            print(f"Top {k} R={(safe_div(m, top_gt))*100:.2f}%")
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top {k} F1={f1_val*100:.2f}%")
            print("--------------------------------")

        # ---- print BM25 signature-based metrics ----
        print("BM25 (signature) results:")
        for k in sig_ks:
            m = sig_metrics[k]["match"]
            p = sig_metrics[k]["pred"]
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
        print("--------------------------------")

        # ---- print BM25 code-based metrics ----
        print("BM25 (code) results:")
        for k in sig_ks:
            m = code_metrics[k]["match"]
            p = code_metrics[k]["pred"]
            #print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%")
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%，{(safe_div(m, top_gt))*100:.2f}%({m}/{p})")
        print("--------------------------------")

        # ---- print BM25 desc-based metrics ----
        print("BM25 (desc) results:")
        for k in sig_ks:
            m = desc_metrics[k]["match"]
            p = desc_metrics[k]["pred"]
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
        print("--------------------------------")
    
    print(f"Analysis completed for {project_path}")


if __name__ == "__main__":
    analyze_project(PROJECT_PATH)
