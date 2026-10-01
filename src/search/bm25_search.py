
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
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import json
import pandas as pd
from typing import Any, Dict, Optional
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

# Default path arguments, provided only as command-line conveniences for direct execution of this script;
# During batch experiments, pass these paths through function arguments.
PROJECT_DIR = "System/mrjob"
METHODS_CSV = (
    str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods.csv'))
)
METHODS_DESC_CSV = (
    str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/methods_with_desc.csv'))
)
FILTERED_PATH = (
    str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))
)
REFINED_QUERIES_CACHE_PATH = (
    str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/refined_queries.json'))
)
ENRE_JSON = (
    str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
)
# Path to the complete DevEval JSONL dataset
DATA_JSONL = str(_package_path('scripts/metrics/deveval/data.jsonl'))


# Whether method names should be normalized; for example, convert mrjob.mrjob.xx in the generated CSV to mrjob.xx for evaluation
NEED_METHOD_NAME_NORM = False
# Whether to use query refinement
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
    methods_csv: str,
    methods_desc_csv: str,
    filtered_path: str,
    refined_queries_cache_path: str,
    enre_json: str,
    data_jsonl: str,
    use_refined_query: bool = USE_REFINED_QUERY,
) -> Dict[str, Any]:
    """
    对单个项目运行 BM25 搜索并返回指标。

    所有路径参数显式传入，便于批量脚本复用。
    """
    # Step 1: Filter by project_dir
    with open(data_jsonl, "r") as infile, open(filtered_path, "w") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get("project_path") == project_dir:
                outfile.write(line)

    # load methods corpus
    methods_df = pd.read_csv(methods_csv, dtype=str).fillna("")
    # ensure columns exist
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

    # load methods with description corpus
    methods_desc_df = pd.read_csv(methods_desc_csv, dtype=str).fillna("")
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

    signature_docs = [
        tokenize_signature(s) for s in methods_df["method_signature"].tolist()
    ]
    # print(signature_docs[:10])
    # print("=======================================")
    code_docs = [tokenize_code(c) for c in methods_df["method_code"].tolist()]
    # print(code_docs[:10])
    # print("=======================================")
    desc_docs = [tokenize_text(d) for d in methods_desc_df["func_desc"].tolist()]
    # print(desc_docs[:10])
    # store the original method strings to return as predicted items
    methods_corpus_strings = methods_df["method_signature"].tolist()
    # methods_desc_df['func_fullName'] and methods_df['method_signature'] are normally identical, but keep the former separate to handle unexpected differences
    methods_desc_corpus_strings = methods_desc_df["func_fullName"].tolist()

    bm25_sig = BM25Okapi(signature_docs)
    bm25_code = BM25Okapi(code_docs)
    bm25_desc = BM25Okapi(desc_docs)

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
        sig_records = []
        code_records = []
        desc_records = []
        top_gt = 0

        sig_ks = [5, 10, 15, 20]
        sig_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        code_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}
        desc_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        def bm25_topk_strings(order, k, corpus_strings):
            idx = order[-k:][::-1]
            return [corpus_strings[i] for i in idx]

        for line in f:
            example_counter += 1
            data = json.loads(line.strip())
            deps = []
            target_method = data.get("namespace") or ""
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
                    with open(refined_queries_cache_path, 'w') as f:
                        json.dump(refined_queries_cache, f, indent=2)
                # print("refined query: ", query)
            else:
                query = original_query
                # print("Using original query: ", query)

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

            sig_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_signature": {},
            }
            for k in sig_ks:
                mk = bm25_topk_strings(sig_order, k, methods_corpus_strings)
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
                "bm25_code": {},
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
                "bm25_desc": {},
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

        out_dir = os.path.dirname(filtered_path)
        sig_path = os.path.join(out_dir, "diagnostic_bm25_signature.jsonl")
        code_path = os.path.join(out_dir, "diagnostic_bm25_code.jsonl")
        desc_path = os.path.join(out_dir, "diagnostic_bm25_desc.jsonl")
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
            "signature": {},
            "code": {},
            "desc": {},
        }

        # ---- print BM25 signature-based metrics ----
        print("BM25 (signature) results:")
        for k in sig_ks:
            m = sig_metrics[k]["match"]
            p = sig_metrics[k]["pred"]
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
            project_metrics["signature"][k] = {
                "match": m,
                "pred": p,
                "top_gt": top_gt,
                "P": safe_div(m, p),
                "R": safe_div(m, top_gt),
                "F1": f1_val,
            }
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
            project_metrics["code"][k] = {
                "match": m,
                "pred": p,
                "top_gt": top_gt,
                "P": safe_div(m, p),
                "R": safe_div(m, top_gt),
                "F1": f1_val,
            }
        print("--------------------------------")

        # ---- print BM25 desc-based metrics ----
        print("BM25 (desc) results:")
        for k in sig_ks:
            m = desc_metrics[k]["match"]
            p = desc_metrics[k]["pred"]
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
            project_metrics["desc"][k] = {
                "match": m,
                "pred": p,
                "top_gt": top_gt,
                "P": safe_div(m, p),
                "R": safe_div(m, top_gt),
                "F1": f1_val,
            }
        print("--------------------------------")

    print(f"Analysis completed for {project_dir}")
    return project_metrics


if __name__ == "__main__":
    analyze_project(
        PROJECT_DIR,
        methods_csv=METHODS_CSV,
        methods_desc_csv=METHODS_DESC_CSV,
        filtered_path=FILTERED_PATH,
        refined_queries_cache_path=REFINED_QUERIES_CACHE_PATH,
        enre_json=ENRE_JSON,
        data_jsonl=DATA_JSONL,
        use_refined_query=USE_REFINED_QUERY,
    )
