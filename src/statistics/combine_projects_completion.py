"""
从 Excel 读取 project_name 列表，将各项目下指定子目录中的 jsonl 文件合并为一个大的 jsonl。

Excel 需包含 project_name 列（或「项目名称」）；路径规则：
  {BASE_COMPLETION_OUT}/{project_name}/{SUBFOLDER}/{JSONL_FILENAME}
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
import os
import sys

import pandas as pd

# Excel path; the project_name column identifies projects and is used to construct paths
DEFAULT_EXCEL = str(_external_path('FEATLENS_PROJECT_LIST_ROOT', 'external/project_lists', '0405_5projects_repograph.xlsx'))
# Base directory containing project folders, each with its own completion results
DEFAULT_BASE_COMPLETION_OUT = str(_package_path('results/generation/deveval/repograph/0405_repograph'))
# Optional nested subdirectory containing completion results; an empty value means the project folder directly
DEFAULT_SUBFOLDER = ""
# Completion file to merge for the selected generation strategy
DEFAULT_JSONL_FILENAME = "repograph_completion.jsonl"
# Path to the merged JSONL file
OUTPUT_COMBINED_JSONL = str(_package_path('results/generation/deveval/repograph/deepseek_v3_2_combined_completion.jsonl'))


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：项目名称 -> project_name"""
    col_map = {}
    for c in df.columns:
        c_str = str(c).strip()
        if c_str in ("项目名称", "project_name"):
            col_map[c] = "project_name"
    return df.rename(columns=col_map)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge per-project jsonl files into one, using project list from Excel")
    p.add_argument("--excel_path", default=DEFAULT_EXCEL, help="Excel 路径，需含 project_name 列")
    p.add_argument("--base_completion_out", default=DEFAULT_BASE_COMPLETION_OUT, help="Completion 输出根目录")
    p.add_argument("--subfolder", default=DEFAULT_SUBFOLDER, help="每个项目下的子目录，如 0303_full")
    p.add_argument("--jsonl_name", default=DEFAULT_JSONL_FILENAME, help="要合并的 jsonl 文件名，如 no_context_completion.jsonl")
    p.add_argument("--output", "-o", default=OUTPUT_COMBINED_JSONL, help="合并后的输出 jsonl 路径")
    p.add_argument("--sheet", type=int, default=0, help="Excel 工作表索引，默认 0")
    args = p.parse_args()

    if not os.path.isfile(args.excel_path):
        print(f"Error: Excel not found: {args.excel_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    df = _normalize_column_names(df)
    if "project_name" not in df.columns:
        print("Error: Excel 需包含「项目名称」或 project_name 列", file=sys.stderr)
        sys.exit(1)

    project_names = df["project_name"].astype(str).str.strip()
    project_names = project_names[project_names.str.len() > 0].tolist()

    total_lines = 0
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for name in project_names:
            path = os.path.join(args.base_completion_out, name, args.subfolder, args.jsonl_name)
            if not os.path.isfile(path):
                print(f"[skip] {name}: not found {path}", file=sys.stderr)
                continue
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    out_f.write(line + "\n")
                    count += 1
            total_lines += count
            print(f"[ok] {name}: {count} lines -> {path}", file=sys.stderr)

    print(f"Done: {total_lines} lines written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
