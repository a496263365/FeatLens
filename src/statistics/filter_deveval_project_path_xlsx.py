
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

import pandas as pd


# KEEP_PROJECT_NAMES = {"boto", "alembic", "diffprivlib", "mopidy"}
KEEP_PROJECT_NAMES = {"mingus"}


def main() -> None:
    input_path = str(_external_path('FEATLENS_PROJECT_LIST_ROOT', 'external/project_lists', 'deveval_project_path_with_dir.xlsx'))
    output_path = str(_external_path('FEATLENS_PROJECT_LIST_ROOT', 'external/project_lists', 'deveval_four_projects.xlsx'))

    df = pd.read_excel(input_path, dtype=str).fillna("")
    if "project_name" not in df.columns:
        raise ValueError(f"missing column: project_name; columns={list(df.columns)}")

    project_name_norm = df["project_name"].astype(str).str.strip().str.lower()
    out_df = df.loc[project_name_norm.isin(KEEP_PROJECT_NAMES)].copy()

    out_df.to_excel(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
