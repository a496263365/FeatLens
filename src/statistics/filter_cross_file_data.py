"""
从 jsonl 中只保留 dependency.cross_file 非空的记录，输出到新 jsonl。
"""

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


DEFAULT_INPUT_JSONL = str(_package_path('data/deveval/tasks_all_1430.jsonl'))
DEFAULT_OUTPUT_JSONL = str(_package_path('data/deveval/tasks_with_dependencies_1146.jsonl'))

def main() -> None:
    p = argparse.ArgumentParser(description="Filter jsonl: keep only records with non-empty dependency.cross_file")
    p.add_argument("--input_jsonl", default=DEFAULT_INPUT_JSONL, help="输入 jsonl 路径")
    p.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="输出 jsonl 路径")
    args = p.parse_args()

    kept = 0
    total = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as f_in, open(
        args.output, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skip invalid json line: {e}", file=sys.stderr)
                continue
            dep = rec.get("dependency") or {}
            cross_file = dep.get("cross_file")
            if isinstance(cross_file, list) and len(cross_file) > 0:
                f_out.write(line + "\n")
                kept += 1

    print(f"Done: {kept}/{total} records kept -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
