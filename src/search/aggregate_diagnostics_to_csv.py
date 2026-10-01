
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
import csv
import argparse

def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records

def aggregate(dir_path, out_csv):
    feature_path = os.path.join(dir_path, "diagnostic_feature.jsonl")
    sig_path = os.path.join(dir_path, "diagnostic_bm25_signature.jsonl")
    code_path = os.path.join(dir_path, "diagnostic_bm25_code.jsonl")

    feature_recs = read_jsonl(feature_path)
    sig_recs = read_jsonl(sig_path)
    code_recs = read_jsonl(code_path)

    examples = {}

    for r in feature_recs:
        ex = r.get("example_id")
        if ex is None:
            continue
        e = examples.setdefault(ex, {"feature_search": {}, "bm25_signature": {}, "bm25_code": {}})
        feat = r.get("feature", {})
        for topk in ["top1", "top3", "top5"]:
            m = feat.get(topk)
            if not m:
                continue
            e["feature_search"][topk] = m.get("metrics", {})

    for r in sig_recs:
        ex = r.get("example_id")
        if ex is None:
            continue
        e = examples.setdefault(ex, {"feature_search": {}, "bm25_signature": {}, "bm25_code": {}})
        sig = r.get("bm25_signature", {})
        for topk in ["top5", "top10", "top15"]:
            m = sig.get(topk)
            if not m:
                continue
            e["bm25_signature"][topk] = m.get("metrics", {})

    for r in code_recs:
        ex = r.get("example_id")
        if ex is None:
            continue
        e = examples.setdefault(ex, {"feature_search": {}, "bm25_signature": {}, "bm25_code": {}})
        cod = r.get("bm25_code", {})
        for topk in ["top5", "top10", "top15"]:
            m = cod.get(topk)
            if not m:
                continue
            e["bm25_code"][topk] = m.get("metrics", {})

    rows = []
    for ex in sorted(examples.keys()):
        e = examples[ex]
        for topk in ["top1", "top3", "top5"]:
            met = e["feature_search"].get(topk)
            if met is None:
                continue
            rows.append([ex, "feature_search", topk, met.get("P", 0), met.get("R", 0), met.get("F1", 0)])
        for topk in ["top5", "top10", "top15"]:
            met = e["bm25_signature"].get(topk)
            if met is None:
                continue
            rows.append([ex, "bm25_signature", topk, met.get("P", 0), met.get("R", 0), met.get("F1", 0)])
        for topk in ["top5", "top10", "top15"]:
            met = e["bm25_code"].get(topk)
            if met is None:
                continue
            rows.append([ex, "bm25_code", topk, met.get("P", 0), met.get("R", 0), met.get("F1", 0)])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "method", "topK", "P", "R", "F1"])
        for row in rows:
            w.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob')))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_csv = args.out or os.path.join(args.dir, "diagnostic_metrics_by_example.csv")
    aggregate(args.dir, out_csv)

if __name__ == "__main__":
    main()