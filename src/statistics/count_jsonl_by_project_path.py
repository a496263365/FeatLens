
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
from collections import Counter
from typing import Dict, Iterable, Tuple


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def normalize_project_path(p: str) -> str:
    p = str(p).strip()
    if not p:
        return ""
    return os.path.normpath(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default=str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl')),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=None)
    args = parser.parse_args()

    jsonl_path = os.path.abspath(args.jsonl)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    counts: Counter[str] = Counter()
    missing_project_path = 0
    total_records = 0

    for _, obj in iter_jsonl(jsonl_path):
        total_records += 1
        project_path = obj.get("project_path")
        if project_path is None:
            missing_project_path += 1
            continue
        key = normalize_project_path(project_path)
        if not key:
            missing_project_path += 1
            continue
        counts[key] += 1

    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if args.top is not None and args.top > 0:
        items_to_print = items[: args.top]
    else:
        items_to_print = items

    print(f"jsonl={jsonl_path}")
    print(f"total_records={total_records}")
    print(f"projects={len(counts)}")
    print(f"missing_project_path={missing_project_path}")
    print("")
    for project_path, c in items_to_print:
        print(f"{c}\t{project_path}")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("project_path,count\n")
            for project_path, c in items:
                project_path_escaped = project_path.replace('"', '""')
                f.write(f"\"{project_path_escaped}\",{c}\n")
        print("")
        print(f"saved_csv={out_path}")


if __name__ == "__main__":
    main()

