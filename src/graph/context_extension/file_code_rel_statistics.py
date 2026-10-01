
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
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


PROJECT_DIR = str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', 'System/mrjob'))
ENRE_JSON = str(_package_path('results/retrieval/deveval/featlens_and_rag/System/mrjob/report-enre.json'))
FILTERED_PATH = str(_package_path('data/deveval/projects/System/mrjob/filtered.jsonl'))


@dataclass
class EnreIndex:
    nodes: Dict[str, Dict[str, Any]]
    adj: Dict[str, List[Tuple[str, Optional[str]]]]
    id_to_qname_norm: Dict[str, str]
    file_to_func_ids: Dict[str, List[str]]


def analyze_file_context_dependency(
    project_dir: str,
    file_path: str,
    method_to_complete: str,
    *,
    enre_json_path: str = ENRE_JSON,
    enre_index: Optional[EnreIndex] = None,
    top_k: int = 50,
    return_data: bool = False,
) -> Optional[Dict[str, Any]]:
    # First, find all code elements in the ENRE report variables whose File equals file_path and whose type is Function
    # These are all functions in the file, but method_to_complete must be excluded based on its qualifiedName
    # For each remaining function, compute the following statistics:
    # Find the function's ENRE ID, called src_id
    # Find all ENRE cells, which represent edges such as calls or uses, whose src is src_id
    # Record the dest IDs of these edges; they are all code elements used by the src function, whether in the same file or another file
    # method_to_complete is the full function name, such as mrjob.ssh.connect; find all code elements in dest that begin with this prefix
    # These code elements are internal to the function and are recorded, but they are not the focus later
    # The focus is other dest elements; record relationships such as src --CALL--> dest
    # When counting relations, distinguish destination types, since USE class and USE function differ
    # Apply the above logic to all functions in file_path except method_to_complete, and output the final statistics

    if enre_index is None:
        enre_index = load_enre_index(enre_json_path, project_dir)

    file_abs = _normalize_input_file_path(project_dir, file_path)
    method_prefix = method_to_complete

    func_ids = enre_index.file_to_func_ids.get(file_abs, [])
    analyzed_func_ids = [
        fid for fid in func_ids if enre_index.id_to_qname_norm.get(fid) != method_prefix
    ]

    relation_counter: Counter[Tuple[str, str]] = Counter()


    for src_id in analyzed_func_ids:
        internal_dest_ids: set[str] = set()
        external_dest_ids: set[str] = set()
        for dest_id, kind in enre_index.adj.get(src_id, []):
            dest_node = enre_index.nodes.get(dest_id)
            if dest_node is None:
                continue
            dest_qname_norm = enre_index.id_to_qname_norm.get(dest_id, "")
            if dest_qname_norm.startswith(method_prefix):
                internal_dest_ids.add(dest_id)
                continue

            external_dest_ids.add(dest_id)
            edge_kind = kind or "UNKNOWN"
            dest_category = dest_node.get("category") or "Unknown"

            relation_counter[(edge_kind, dest_category)] += 1

    def _sort_counter(counter: Counter, limit: int) -> List[Tuple[Any, int]]:
        return sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))[:limit]

    result: Dict[str, Any] = {
        "file_path": file_abs,
        "method_to_complete": method_to_complete,
        "functions_in_file": len(func_ids),
        "functions_analyzed": len(analyzed_func_ids),
        "top_relations": [
            {"kind": k, "dest_category": c, "count": cnt}
            for (k, c), cnt in _sort_counter(relation_counter, top_k)
        ],
    }

    if return_data:
        return result
    print(json.dumps(result, ensure_ascii=False))
    return None


def _keep_enre_node(var: dict) -> bool:
    if not var.get('category', ''):
        return False
    cat = var.get('category', '')
    if cat.startswith("Unknown") or cat.startswith("Ambiguous"):
        return False
    if cat == 'Unresolved Attribute':
        # The File field must exist and contain '/'; otherwise the symbol may be an attribute of an external library
        if ('File' not in var) or ('/' not in var.get('File', '')):
            return False
    return True

def _normalize_input_file_path(project_dir: str, file_path: str) -> str:
    file_path = file_path.strip()
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    return os.path.normpath(os.path.join(project_dir, file_path))


def _normalize_enre_file_path(project_dir: str, enre_file_path: Optional[str]) -> Optional[str]:
    if not enre_file_path:
        return None
    if os.path.isabs(enre_file_path):
        return os.path.normpath(enre_file_path)
    return os.path.normpath(os.path.join(project_dir, enre_file_path))


@lru_cache(maxsize=4)
def load_enre_index(enre_json_path: str, project_dir: str) -> EnreIndex:
    with open(enre_json_path, "r") as f:
        data = json.load(f)

    variables = data.get("variables", [])
    cells = data.get("cells", [])

    nodes: Dict[str, Dict[str, Any]] = {}
    id_to_qname_norm: Dict[str, str] = {}
    file_to_func_ids: DefaultDict[str, List[str]] = defaultdict(list)

    for var in variables:
        if not _keep_enre_node(var):
            continue
        vid = str(var.get("id"))
        qname = var.get("qualifiedName") or ""
        category = var.get("category") or ""
        file_abs = _normalize_enre_file_path(project_dir, var.get("File"))

        nodes[vid] = {"qualifiedName": qname, "category": category}
        id_to_qname_norm[vid] = qname

        if category == "Function" and file_abs is not None:
            file_to_func_ids[file_abs].append(vid)

    adj: DefaultDict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    seen_by_src: DefaultDict[str, set[Tuple[str, Optional[str]]]] = defaultdict(set)

    for cell in cells:
        src = cell.get("src")
        dest = cell.get("dest")
        if src is None or dest is None:
            continue
        src = str(src)
        dest = str(dest)
        if src not in nodes or dest not in nodes:
            continue
        kind = (cell.get("values") or {}).get("kind")
        pair = (dest, kind)
        if pair in seen_by_src[src]:
            continue
        seen_by_src[src].add(pair)
        adj[src].append(pair)

    return EnreIndex(
        nodes=nodes,
        adj=dict(adj),
        id_to_qname_norm=id_to_qname_norm,
        file_to_func_ids=dict(file_to_func_ids),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=PROJECT_DIR)
    parser.add_argument("--enre-json", default=ENRE_JSON)
    parser.add_argument("--top-k", type=int, default=100)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    one = subparsers.add_parser("one")
    one.add_argument("--file", required=True)
    one.add_argument("--method", required=True)

    batch = subparsers.add_parser("filtered")
    batch.add_argument("--filtered-jsonl", default=FILTERED_PATH)

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.cmd == "one":
            analyze_file_context_dependency(
                args.project_dir,
                args.file,
                args.method,
                enre_json_path=args.enre_json,
                top_k=args.top_k,
                return_data=False,
            )
            return 0

        if args.cmd == "filtered":
            enre_index = load_enre_index(args.enre_json, args.project_dir)
            with open(args.filtered_jsonl, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    task = json.loads(line)
                    method_to_complete = task.get("namespace") or ""
                    completion_path = task.get("completion_path") or ""
                    project_path = task.get("project_path") or ""
                    if project_path and completion_path.startswith(project_path):
                        completion_path = completion_path[len(project_path) :].lstrip("/\\")
                    example_id = task.get("example_id", i + 1)

                    res = analyze_file_context_dependency(
                        args.project_dir,
                        completion_path,
                        method_to_complete,
                        enre_json_path=args.enre_json,
                        enre_index=enre_index,
                        top_k=args.top_k,
                        return_data=True,
                    )
                    if res is None:
                        continue
                    res["example_id"] = example_id
                    print(json.dumps(res, ensure_ascii=False))
            return 0
    except BrokenPipeError:
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
