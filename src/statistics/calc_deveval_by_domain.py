"""
Aggregate DevEval statistics by domain: merge Excel project metrics with JSONL tasks.

Default input paths are set below (DEFAULT_EXCEL_PATH / DEFAULT_JSONL_PATH); edit them for your
machine, or pass --excel / --jsonl on the command line.
"""

from __future__ import annotations

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
from typing import Any, DefaultDict, Dict, Optional, Set, Tuple

import pandas as pd

SKIP_PROJECT_NAMES = {
    "Telethon",
    "pyramid",
    "pyinfra",
    "kinto",
    "capirca",
    "sqlitedict",
    "Django",
    "TPOT",
    "aioxmpp",
    "albumentations",
    "arctic-latest",
    "cupy",
    "datasets",
    "djangorestframework-simplejwt",
    "mongo-doc-manager",
    "natasha",
    "proxybroker",
    "psd-tools",
    "pymc",
    "pytube",
    "sshuttle",
    "wandb",
    "xmnlp",
    "ydata-profiling",
    "python-for-android",
}

# Override these defaults when the script lives on another machine or layout.
DEFAULT_EXCEL_PATH = str(_package_path('data/deveval/project_statistics_source.xlsx'))
DEFAULT_JSONL_PATH = str(_package_path('scripts/metrics/deveval/data.jsonl'))


def _dependency_count(dep: Any) -> int:
    if not isinstance(dep, dict):
        return 0
    n = 0
    for key in ("intra_class", "intra_file", "cross_file"):
        v = dep.get(key)
        if isinstance(v, list):
            n += len(v)
    return n


def load_excel_stats(
    excel_path: Path,
    skip_names: Set[str],
) -> Tuple[Dict[str, Dict[str, float]], Set[str]]:
    """
    Returns:
      project_name -> {files, lines, functions}
      set of skipped project_name (for diagnostics)
    """
    df = pd.read_excel(excel_path, sheet_name=0)
    required = (
        "project_name",
        "python_file_count",
        "python_total_line_count",
        "python_function_count",
    )
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Excel missing column {c!r}; have {list(df.columns)}")

    stats: Dict[str, Dict[str, float]] = {}
    skipped: Set[str] = set()

    for idx, row in df.iterrows():
        name = str(row["project_name"]).strip()
        if not name or name.lower() == "nan":
            print(f"[excel] skip row {idx}: empty project_name", file=sys.stderr)
            continue
        if name in skip_names:
            skipped.add(name)
            print(f"[excel] SKIP_PROJECT_NAMES: ignoring row project_name={name!r}", file=sys.stderr)
            continue

        def _num(col: str) -> float:
            v = row[col]
            if pd.isna(v):
                return float("nan")
            return float(v)

        stats[name] = {
            "files": _num("python_file_count"),
            "lines": _num("python_total_line_count"),
            "functions": _num("python_function_count"),
        }

    return stats, skipped


def parse_project_path(project_path: str) -> Tuple[str, str]:
    s = str(project_path).strip().strip("/")
    if "/" not in s:
        raise ValueError(f"project_path must be Domain/project_name, got {project_path!r}")
    domain, project = s.split("/", 1)
    domain = domain.strip()
    project = project.strip()
    if not domain or not project:
        raise ValueError(f"invalid project_path: {project_path!r}")
    return domain, project


def scan_jsonl(
    jsonl_path: Path,
) -> Tuple[
    DefaultDict[str, Set[str]],
    DefaultDict[str, int],
    DefaultDict[str, int],
]:
    """
    Returns:
      domain -> set of project names
      domain -> task count
      domain -> sum of dependency counts (all tasks)
    """
    projects: DefaultDict[str, Set[str]] = defaultdict(set)
    task_counts: DefaultDict[str, int] = defaultdict(int)
    dep_sums: DefaultDict[str, int] = defaultdict(int)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[jsonl] line {line_no}: JSON error: {e}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue

            pp = obj.get("project_path")
            if pp is None:
                print(f"[jsonl] line {line_no}: missing project_path", file=sys.stderr)
                continue

            try:
                domain, proj = parse_project_path(str(pp))
            except ValueError as e:
                print(f"[jsonl] line {line_no}: {e}", file=sys.stderr)
                continue

            projects[domain].add(proj)
            task_counts[domain] += 1
            dep_sums[domain] += _dependency_count(obj.get("dependency"))

    return projects, task_counts, dep_sums


def domain_metric_averages(
    domain: str,
    project_names: Set[str],
    excel_stats: Dict[str, Dict[str, float]],
) -> Tuple[Optional[float], Optional[float], Optional[float], int, list[str]]:
    """Mean files / lines / functions over projects in domain that have Excel stats; list missing."""
    files: list[float] = []
    lines: list[float] = []
    funcs: list[float] = []
    missing: list[str] = []

    for name in sorted(project_names):
        row = excel_stats.get(name)
        if not row:
            missing.append(name)
            continue
        fv, lv, gv = row["files"], row["lines"], row["functions"]
        if fv == fv:  # not NaN
            files.append(fv)
        if lv == lv:
            lines.append(lv)
        if gv == gv:
            funcs.append(gv)

    def _mean(xs: list[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return _mean(files), _mean(lines), _mean(funcs), len(files), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="DevEval per-domain statistics.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path(DEFAULT_EXCEL_PATH),
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path(DEFAULT_JSONL_PATH),
    )
    args = parser.parse_args()

    excel_path = args.excel.resolve()
    jsonl_path = args.jsonl.resolve()
    if not excel_path.is_file():
        raise SystemExit(f"Excel not found: {excel_path}")
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    excel_stats, _ = load_excel_stats(excel_path, SKIP_PROJECT_NAMES)
    projects_by_domain, task_counts, dep_sums = scan_jsonl(jsonl_path)

    domains = sorted(projects_by_domain.keys())

    rows_out = []
    for domain in domains:
        projs = projects_by_domain[domain]
        n_proj = len(projs)
        n_tasks = task_counts[domain]
        avg_dep = dep_sums[domain] / n_tasks if n_tasks else 0.0

        avg_f, avg_l, avg_g, n_with_excel, missing = domain_metric_averages(
            domain, projs, excel_stats
        )
        if missing:
            print(
                f"[warn] domain={domain!r}: {len(missing)} project(s) have tasks but no "
                f"(non-skipped) Excel stats (e.g. {missing[:3]}{'...' if len(missing) > 3 else ''})",
                file=sys.stderr,
            )

        rows_out.append(
            {
                "domain": domain,
                "project_count": n_proj,
                "avg_python_file_count": avg_f,
                "avg_python_function_count": avg_g,
                "avg_python_line_count": avg_l,
                "task_count": n_tasks,
                "avg_dependency_count_per_task": avg_dep,
                # Mean above is over projects that still have a row in Excel after SKIP_PROJECT_NAMES
                "projects_with_excel_stats": n_with_excel,
            }
        )

    col_order = [
        "domain",
        "project_count",
        "avg_python_file_count",
        "avg_python_function_count",
        "avg_python_line_count",
        "task_count",
        "avg_dependency_count_per_task",
        "projects_with_excel_stats",
    ]
    out_df = pd.DataFrame(rows_out)[col_order]
    # Human-readable console table
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}" if x == x else "nan")
    print(out_df.to_string(index=False))

    # Totals row (mean of domain-level averages is not meaningful; sum tasks/projects)
    total_projects = sum(len(projects_by_domain[d]) for d in domains)
    total_tasks = sum(task_counts[d] for d in domains)
    total_deps = sum(dep_sums[d] for d in domains)
    print()
    print(
        f"ALL domains: distinct projects={total_projects}, tasks={total_tasks}, "
        f"avg deps per task (global)={total_deps / total_tasks if total_tasks else 0:.4f}"
    )


if __name__ == "__main__":
    main()
