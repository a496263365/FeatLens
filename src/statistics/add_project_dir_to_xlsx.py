
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
from typing import Optional

import pandas as pd


def compute_project_dir(project_name: str, project_path: str) -> str:
    name = str(project_name).strip()
    path = str(project_path).strip()
    if not name or not path:
        return ""

    path = path.replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""

    name_norm = name.lower()
    idx: Optional[int] = None
    for i, part in enumerate(parts):
        if part.lower() == name_norm:
            idx = i
            break

    if idx is None or idx == 0:
        return ""

    return f"{parts[idx - 1]}/{parts[idx]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(_external_path('FEATLENS_PROJECT_LIST_ROOT', 'external/project_lists', 'deveval_project_path.xlsx')),
    )
    parser.add_argument(
        "--output",
        default=str(_external_path('FEATLENS_PROJECT_LIST_ROOT', 'external/project_lists', 'deveval_project_path_with_dir.xlsx')),
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    df = pd.read_excel(input_path, dtype=str).fillna("")
    for col in ("project_name", "project_path"):
        if col not in df.columns:
            raise ValueError(f"missing column: {col}; columns={list(df.columns)}")

    project_dirs = [
        compute_project_dir(project_name, project_path)
        for project_name, project_path in zip(
            df["project_name"].astype(str).tolist(),
            df["project_path"].astype(str).tolist(),
        )
    ]

    if "project_dir" in df.columns:
        df["project_dir"] = project_dirs
    else:
        insert_at = len(df.columns)
        if "project_path" in df.columns:
            insert_at = int(list(df.columns).index("project_path")) + 1
        df.insert(insert_at, "project_dir", project_dirs)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
