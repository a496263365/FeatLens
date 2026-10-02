
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

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
import json
import pandas as pd
from typing import Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi
from utils.query_refine import refine_query
from utils.enre_utils import (
    clear_enre_elements,
    load_enre_elements,
    _normalize_symbol,
    compute_task_recall,
    is_method_hit,
)

# Default paths and control parameters, provided only as conveniences for direct script execution;
# During batch experiments, pass these paths and hyperparameters through function arguments.
PROJECT_DIR = "System/mrjob"
FEATURE_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/features.csv'))
METHODS_CSV = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
REFINED_QUERIES_CACHE_PATH = (
    str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/refined_queries.json'))
)
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
# Path to the complete DevEval JSONL dataset
DATA_JSONL = str(_package_path('scripts/metrics/deveval/data.jsonl'))
TOP_SM = [1, 2, 3]  # Number of similar methods to select
CLUSTER_KS = [1, 3]

# Whether method names should be normalized; for example, convert mrjob.mrjob.xx in the generated CSV to mrjob.xx for evaluation
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False


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


def analyze_project(
    project_dir: str,
    *,
    feature_csv: str,
    methods_csv: str,
    filtered_path: str,
    refined_queries_cache_path: str,
    enre_json: str,
    data_jsonl: str,
    top_sm: list[int] = TOP_SM,
    cluster_ks: list[int] = CLUSTER_KS,
    use_refined_query: bool = USE_REFINED_QUERY,
) -> Dict[str, Any]:
    """
    对单个项目运行 feature-based 搜索并返回指标。
    所有路径参数显式传入，便于批量脚本复用。
    """
    # Step 1: Filter by project_dir
    with open(data_jsonl, "r") as infile, open(filtered_path, "w") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get("project_path") == project_dir:
                outfile.write(line)

    # Step 2: Find similar clusters
    df = pd.read_csv(
        feature_csv,
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
    methods_df = pd.read_csv(methods_csv, dtype=str).fillna("")
    # ensure columns exist
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

    clear_enre_elements()
    load_enre_elements(enre_json)

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

    def tokenize_code(code: str):
        toks = re.findall(r'\w+', code.lower())
        return toks

    code_docs = [tokenize_code(c) for c in methods_df["method_code"].tolist()]
    methods_corpus_strings = methods_df["method_signature"].tolist()
    bm25_code = BM25Okapi(code_docs)

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

    # Load or initialize the query cache
    if os.path.exists(refined_queries_cache_path):
        with open(refined_queries_cache_path, "r") as f:
            refined_queries_cache = json.load(f)
    else:
        refined_queries_cache = {}
        with open(refined_queries_cache_path, "w") as f:
            json.dump(refined_queries_cache, f, indent=2)

    with open(filtered_path, "r") as f:
        example_counter = 0
        feature_records = []
        top_gt = 0

        cluster_metrics = {k: {"match": 0, "pred": 0} for k in cluster_ks}

        def methods_from_top_k_clusters(similarities, k, forced_cluster_id=None):
            k_int = int(k)
            if k_int == 1 and (forced_cluster_id is None or str(forced_cluster_id).strip() == ""):
                return []
            ranked_idx = np.argsort(similarities)[::-1]
            ranked_ids = clusters.iloc[ranked_idx]["id"].astype(str).tolist()

            selected_ids = []
            if forced_cluster_id is not None:
                forced_id = str(forced_cluster_id)
                if forced_id in ranked_ids:
                    selected_ids.append(forced_id)

            for cid in ranked_ids:
                if cid in selected_ids:
                    continue
                selected_ids.append(cid)
                if len(selected_ids) >= k_int:
                    break

            methods = []
            for cid in selected_ids[:k_int]:
                methods.extend(
                    df.loc[df["id"].astype(str) == cid, "method_name"].astype(str).tolist()
                )
            return methods

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

            similar_methods = {}
            # Only observable task intent/name may initialize similarity. Never
            # use the held-out implementation, even to create a heuristic flag.
            target_code_tokens = tokenize_code(
                target_method_norm + ' ' + data['requirement']['Functionality']
                + ' ' + data['requirement']['Arguments']
            )
            if target_code_tokens:
                target_code_scores = bm25_code.get_scores(target_code_tokens)
                target_code_order = np.argsort(target_code_scores)
                target_method_norm = _normalize_symbol(target_method)
                for k in top_sm:
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
                for k in top_sm:
                    similar_methods[f"top{int(k)}"] = []
            # Extract ground-truth dependencies from the data; these are the correct answers for this search
            deps.extend(data["dependency"]["intra_class"])
            deps.extend(data["dependency"]["intra_file"])
            deps.extend(data["dependency"]["cross_file"])
            # print("deps",deps)
            # Cleaning/filtering
            # deps = [dep for dep in deps if (dep in method_names) or (dep in variables_enre)]
            # print("deps after filter",deps)
            # input("please confirm the deps!")
            # Add the number of correct answers for this test case to the total
            top_gt += len(deps)

            # feature-based search
            original_query = (
                data["requirement"]["Functionality"]
                + " "
                + data["requirement"]["Arguments"]
            )
            #print("original query: ", original_query)

            if use_refined_query:
                if original_query in refined_queries_cache:
                    query = refined_queries_cache[original_query]
                    # print("found in cache")
                else:
                    modelname = "deepseek-v3.2"
                    query = refine_query(original_query, modelname)
                    refined_queries_cache[original_query] = query
                    # Important: save immediately after appending.
                    with open(refined_queries_cache_path, "w") as f:
                        json.dump(refined_queries_cache, f, indent=2)
                # print("refined query: ", query)
            else:
                query = original_query
                # print("Using original query: ", query)
            #input("please confirm the query!")
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]

            for k in cluster_ks:
                methods_k = methods_from_top_k_clusters(similarities, k, target_feature_id)
                methods_k = filter_out_target_method(methods_k, target_method)
                cluster_metrics[k]["pred"] += len(methods_k)
                cluster_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(methods_k))["dependency_hit"]

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
            for k in cluster_ks:
                mk = methods_from_top_k_clusters(similarities, k, target_feature_id)
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

        out_dir = os.path.dirname(filtered_path)
        feature_path = os.path.join(out_dir, "diagnostic_***feature.jsonl")
        with open(feature_path, "w", encoding="utf-8") as fo:
            for rec in feature_records:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Save the updated cache
        with open(refined_queries_cache_path, "w") as f:
            json.dump(refined_queries_cache, f, indent=4)

        # ===================== DIAGNOSTIC JSON CODE END =====================
        def safe_div(a, b):
            return (a / b) if b != 0 else 0

        # Save aggregate metrics for batch statistics
        project_metrics: Dict[str, Any] = {
            "project_dir": project_dir,
            "num_examples": example_counter,
            "top_gt": top_gt,
            "feature": {},
        }

        for k in cluster_ks:
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
            project_metrics["feature"][k] = {
                "match": m,
                "pred": p,
                "top_gt": top_gt,
                "P": safe_div(m, p),
                "R": safe_div(m, top_gt),
                "F1": f1_val,
            }

    print(f"Analysis completed for {project_dir}")
    return project_metrics


if __name__ == "__main__":
    analyze_project(
        PROJECT_DIR,
        feature_csv=FEATURE_CSV,
        methods_csv=METHODS_CSV,
        filtered_path=FILTERED_PATH,
        refined_queries_cache_path=REFINED_QUERIES_CACHE_PATH,
        enre_json=ENRE_JSON,
        data_jsonl=DATA_JSONL,
        top_sm=TOP_SM,
        cluster_ks=CLUSTER_KS,
        use_refined_query=USE_REFINED_QUERY,
    )
