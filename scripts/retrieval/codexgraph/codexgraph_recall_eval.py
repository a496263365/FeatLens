
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

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Make import robust after moving this script out of CodeContextSearch/src/search.
WORKSPACE_ROOT = _package_path()
SEARCH_SRC_DIR = _package_path('src/search')
if str(SEARCH_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SEARCH_SRC_DIR))

from utils.enre_utils import _normalize_symbol, compute_task_recall


DEFAULT_COMPLETION_JSONL = str(_package_path('results/retrieval/evocodebench/codexgraph/completions/deepseek_v3_2_q2_final.jsonl'))
DEFAULT_GT_JSONL = str(_package_path('data/evocodebench/tasks_all_123.jsonl'))
DEFAULT_OUTPUT_JSONL = str(_package_path('results/retrieval/evocodebench/codexgraph/dependency_recall_detail.jsonl'))


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def collect_deps(record: Dict[str, Any]) -> List[str]:
    dep = record.get("dependency", {}) or {}
    merged: List[str] = []
    merged.extend(dep.get("intra_class", []) or [])
    merged.extend(dep.get("intra_file", []) or [])
    merged.extend(dep.get("cross_file", []) or [])
    return [_normalize_symbol(str(x)).strip() for x in merged if str(x).strip()]


def parse_task_namespace(namespace: str, task_type: str) -> Tuple[str, Optional[str]]:
    parts = [p for p in namespace.split(".") if p]
    if not parts:
        return "", None

    if task_type == "method" and len(parts) >= 3:
        return ".".join(parts[:-2]), parts[-2]
    if len(parts) >= 2:
        return ".".join(parts[:-1]), None
    return "", None


def normalize_file_path_to_module(file_path: str) -> str:
    fp = str(file_path or "").strip().replace("\\", "/")
    fp = fp.lstrip("/")
    if not fp:
        return ""
    if fp.endswith(".py"):
        fp = fp[:-3]
    if fp.endswith("/__init__"):
        fp = fp[: -len("/__init__")]
    fp = fp.strip("/")
    return fp.replace("/", ".") if fp else ""


def infer_symbol_module(symbol: str, known_modules_sorted: List[str]) -> str:
    for module in known_modules_sorted:
        if symbol == module or symbol.startswith(module + "."):
            return module
    return ""


def build_symbol_index(
    gt_map: Dict[str, Dict[str, Any]],
) -> Tuple[Set[str], Dict[Tuple[str, str], Set[str]]]:
    known_modules: Set[str] = set()
    symbol_vocab: Set[str] = set()

    for namespace, meta in gt_map.items():
        module = meta.get("task_module", "")
        if module:
            known_modules.add(module)
        symbol_vocab.add(_normalize_symbol(namespace).replace(".__init__", ""))
        for dep in meta.get("deps", []):
            symbol_vocab.add(_normalize_symbol(dep).replace(".__init__", ""))

    known_modules_sorted = sorted(known_modules, key=len, reverse=True)

    module_tail_to_symbols: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for symbol in symbol_vocab:
        if not symbol:
            continue
        module = infer_symbol_module(symbol, known_modules_sorted)
        if not module:
            continue
        tail = symbol.split(".")[-1]
        module_tail_to_symbols[(module, tail)].add(symbol)

    return symbol_vocab, module_tail_to_symbols


def relative_depth(candidate: str, module: str) -> int:
    if module and candidate.startswith(module + "."):
        rel = candidate[len(module) + 1 :]
    else:
        rel = candidate
    return rel.count(".")


def choose_best_signature(
    entity: Dict[str, Any],
    task_meta: Dict[str, Any],
    module_tail_to_symbols: Dict[Tuple[str, str], Set[str]],
) -> str:
    entity_type = str(entity.get("entity_type", "")).upper().strip()
    entity_name = str(entity.get("entity_name", "")).strip()
    if not entity_name:
        return ""

    task_module = task_meta.get("task_module", "")
    task_class = task_meta.get("task_class")
    file_module = normalize_file_path_to_module(str(entity.get("file_path", "")))
    module = file_module or task_module

    idx_candidates: Set[str] = set()
    if module:
        idx_candidates |= module_tail_to_symbols.get((module, entity_name), set())
    if task_module and task_module != module:
        idx_candidates |= module_tail_to_symbols.get((task_module, entity_name), set())

    candidate_pool: Set[str] = set(idx_candidates)
    candidate_pool.add(entity_name)
    if module:
        candidate_pool.add(f"{module}.{entity_name}")
    if task_module:
        candidate_pool.add(f"{task_module}.{entity_name}")

    if entity_type == "METHOD" and task_class and task_module:
        candidate_pool.add(f"{task_module}.{task_class}.{entity_name}")
    if entity_type == "METHOD" and task_class and module:
        candidate_pool.add(f"{module}.{task_class}.{entity_name}")

    if entity_type == "MODULE":
        if "." in entity_name:
            candidate_pool.add(entity_name)
        if module:
            candidate_pool.add(module)

    def score(candidate: str) -> Tuple[int, int, str]:
        s = 0
        if candidate in idx_candidates:
            s += 100
        if module and candidate.startswith(module + "."):
            s += 20
        if task_module and candidate.startswith(task_module + "."):
            s += 10
        if task_class and candidate.endswith(f".{task_class}.{entity_name}"):
            s += 35

        depth = relative_depth(candidate, module or task_module)
        if entity_type == "METHOD":
            s += 15 if depth >= 1 else -10
        elif entity_type in {"FUNCTION", "CLASS", "MODULE"}:
            if depth == 0:
                s += 8
            elif depth > 1:
                s -= 2
        elif entity_type in {"FIELD", "GLOBAL_VARIABLE", "OTHER"}:
            s += 3 if depth >= 0 else 0

        return s, -len(candidate), candidate

    best = max(candidate_pool, key=score)
    return _normalize_symbol(best).strip()


def build_context_code_list(
    entities: List[Dict[str, Any]],
    top_k: int,
    task_meta: Dict[str, Any],
    module_tail_to_symbols: Dict[Tuple[str, str], Set[str]],
) -> List[Dict[str, Any]]:
    context_list: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for entity in entities[:top_k]:
        if not isinstance(entity, dict):
            continue
        sig = choose_best_signature(entity, task_meta, module_tail_to_symbols)
        if not sig:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        context_list.append(
            {
                "sig": sig,
                "method_signature": sig,
                "method_code": "",
            }
        )
    return context_list


def evaluate_recall(
    completion_jsonl: str,
    gt_jsonl: str,
    output_jsonl: Optional[str],
    top_ks: List[int],
) -> Dict[str, Any]:
    gt_map: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(gt_jsonl):
        namespace = str(rec.get("namespace", "")).strip()
        if not namespace:
            continue
        task_type = str(rec.get("type", "")).strip()
        task_module, task_class = parse_task_namespace(namespace, task_type)
        gt_map[namespace] = {
            "deps": collect_deps(rec),
            "type": task_type,
            "project_path": rec.get("project_path", ""),
            "task_module": task_module,
            "task_class": task_class,
        }

    _, module_tail_to_symbols = build_symbol_index(gt_map)

    agg_micro: Dict[int, Dict[str, int]] = {k: {"hit": 0, "total": 0} for k in top_ks}
    # Per-project micro recall aggregations keyed by project_path.
    project_agg_micro: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(
        lambda: {k: {"hit": 0, "total": 0} for k in top_ks}
    )
    project_eval_count: Dict[str, int] = defaultdict(int)

    detail_rows: List[Dict[str, Any]] = []
    total_rows = 0
    missing_ns = 0

    for row in read_jsonl(completion_jsonl):
        total_rows += 1
        namespace = str(row.get("namespace", "")).strip()
        if namespace not in gt_map:
            missing_ns += 1
            continue

        task_meta = gt_map[namespace]
        deps = task_meta["deps"]
        project_path = str(task_meta.get("project_path", "") or "").strip()
        entities = row.get("retrieved_entities") or []
        if not isinstance(entities, list):
            entities = []

        detail: Dict[str, Any] = {
            "namespace": namespace,
            "idx": row.get("idx"),
            "dependency_total": len(set(deps)),
            "retrieved_entities_count": len(entities),
            "metrics": {},
        }

        for k in top_ks:
            context_list = build_context_code_list(
                entities=entities,
                top_k=k,
                task_meta=task_meta,
                module_tail_to_symbols=module_tail_to_symbols,
            )
            recall_info = compute_task_recall(deps, context_list)
            dep_total = int(recall_info["dependency_total"])
            dep_hit = int(recall_info["dependency_hit"])
            recall_val = recall_info["recall"]

            agg_micro[k]["hit"] += dep_hit
            agg_micro[k]["total"] += dep_total
            if project_path:
                project_agg_micro[project_path][k]["hit"] += dep_hit
                project_agg_micro[project_path][k]["total"] += dep_total

            detail["metrics"][f"recall@{k}"] = {
                "hit": dep_hit,
                "total": dep_total,
                "recall": recall_val,
                "resolved_entities": [x["method_signature"] for x in context_list],
            }

        if project_path:
            project_eval_count[project_path] += 1
        detail_rows.append(detail)

    summary: Dict[str, Any] = {
        "completion_jsonl": completion_jsonl,
        "gt_jsonl": gt_jsonl,
        "num_rows": total_rows,
        "num_evaluated": len(detail_rows),
        "num_missing_namespace": missing_ns,
        "recall": {},
    }

    for k in top_ks:
        micro_total = agg_micro[k]["total"]
        micro_hit = agg_micro[k]["hit"]
        micro_recall = (micro_hit / micro_total) if micro_total > 0 else 0.0

        summary["recall"][f"@{k}"] = {
            "micro": {
                "hit": micro_hit,
                "total": micro_total,
                "recall": micro_recall,
            },
        }

    summary["project_recall"] = {}
    for project_path in sorted(project_agg_micro.keys()):
        summary["project_recall"][project_path] = {
            "num_evaluated": project_eval_count.get(project_path, 0),
            "recall": {},
        }
        for k in top_ks:
            p_total = project_agg_micro[project_path][k]["total"]
            p_hit = project_agg_micro[project_path][k]["hit"]
            p_recall = (p_hit / p_total) if p_total > 0 else 0.0
            summary["project_recall"][project_path]["recall"][f"@{k}"] = {
                "micro": {
                    "hit": p_hit,
                    "total": p_total,
                    "recall": p_recall,
                }
            }

    if output_jsonl:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for detail in detail_rows:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

        summary_path = out_path.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate codexgraph retrieved_entities recall@K against DevEval dependencies."
    )
    parser.add_argument(
        "--completion-jsonl",
        default=DEFAULT_COMPLETION_JSONL,
        help="Path to combined_completion.jsonl",
    )
    parser.add_argument(
        "--gt-jsonl",
        default=DEFAULT_GT_JSONL,
        help="Path to combined_filtered.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Output diagnostic jsonl path",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[10, 15, 20],
        help="Recall@K list, e.g. --top-k 10 15 20",
    )
    args = parser.parse_args()

    top_ks = sorted({k for k in args.top_k if k > 0})
    if not top_ks:
        raise ValueError("--top-k must contain at least one positive integer")

    summary = evaluate_recall(
        completion_jsonl=args.completion_jsonl,
        gt_jsonl=args.gt_jsonl,
        output_jsonl=args.output_jsonl or None,
        top_ks=top_ks,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
