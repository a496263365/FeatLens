
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
import csv
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Set, Tuple

from main_mopidy import main as summarize_project


SKIP_PROJECTS = {
    ("Communications", "Telethon"),
    ("Internet", "pyramid"),
    ("System", "pyinfra"),
    ("Internet", "kinto"),
    ("Security", "capirca"),
    ("Database", "sqlitedict"),
}


def discover_projects(deveval_root: Path) -> Iterator[Tuple[str, str, Path]]:
    """Yield (category, project_name, project_path) for all DevEval projects."""
    for category_dir in sorted(deveval_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for project_dir in sorted(category_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            yield category_dir.name, project_dir.name, project_dir


def run_one_project(project_path: Path, output_dir: Path) -> Dict[str, object]:
    """Run summarization for one project and normalize result schema."""
    os.makedirs(output_dir, exist_ok=True)
    result = summarize_project(project_root=str(project_path), output_dir=str(output_dir))

    if result is None:
        return {
            "status": "failed",
            "reason": "main_mopidy returned None",
            "total_functions": 0,
            "total_features": 0,
        }

    return {
        "status": result.get("status", "unknown"),
        "reason": result.get("reason", ""),
        "total_functions": int(result.get("total_functions", 0) or 0),
        "total_features": int(result.get("total_features", 0) or 0),
        "description_issue_features": int(result.get("description_issue_features", 0) or 0),
        "description_fallback_features": int(result.get("description_fallback_features", 0) or 0),
        "description_issue_feature_ids": str(result.get("description_issue_feature_ids", [])),
    }


def load_processed_projects(report_file: Path) -> Set[Tuple[str, str]]:
    processed: Set[Tuple[str, str]] = set()
    if not report_file.exists():
        return processed

    with open(report_file, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            category = (row.get("category") or "").strip()
            project = (row.get("project") or "").strip()
            if category and project:
                processed.add((category, project))
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run main_mopidy for DevEval projects.")
    parser.add_argument(
        "--deveval-root",
        default=str(_external_path('FEATLENS_DEVEVAL_BENCHMARK_ROOT', 'external/deveval', '')),
        help="DevEval dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        default=str(_package_path('results/retrieval/deveval/featlens_and_rag')),
        help="Root output directory for project summaries.",
    )
    parser.add_argument(
        "--report-file",
        default=str(_package_path('results/retrieval/deveval/featlens_and_rag/run_summary.csv')),
        help="CSV file to store per-project function/feature statistics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most N newly executed projects. Skipped projects do not count. 0 means no limit.",
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help="Run at most N newly executed projects (skipped projects do not count). 0 means no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    deveval_root = Path(args.deveval_root).resolve()
    output_root = Path(args.output_root).resolve()
    report_file = Path(args.report_file).resolve()

    if not deveval_root.exists() or not deveval_root.is_dir():
        raise FileNotFoundError(f"Invalid DevEval root: {deveval_root}")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(report_file.parent, exist_ok=True)

    projects = list(discover_projects(deveval_root))

    processed_projects = load_processed_projects(report_file)

    # Backward compatible effective cap for newly executed runs.
    effective_new_runs_limit = 0
    if args.max_new_runs and args.max_new_runs > 0:
        effective_new_runs_limit = args.max_new_runs
    elif args.limit and args.limit > 0:
        effective_new_runs_limit = args.limit

    fieldnames = [
        "category",
        "project",
        "project_root",
        "output_dir",
        "status",
        "reason",
        "total_functions",
        "total_features",
        "description_issue_features",
        "description_fallback_features",
        "description_issue_feature_ids",
        "start_time",
        "end_time",
        "duration_seconds",
    ]

    write_header = not report_file.exists() or report_file.stat().st_size == 0
    with open(report_file, "a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        total = len(projects)
        executed_count = 0
        for idx, (category, project_name, project_path) in enumerate(projects, start=1):
            if effective_new_runs_limit > 0 and executed_count >= effective_new_runs_limit:
                print(f"Reached run limit={effective_new_runs_limit} (new runs only). Stop batch early.")
                break

            start_ts = datetime.now()
            start_time = time.time()
            project_output_dir = output_root / category / project_name

            if (category, project_name) in SKIP_PROJECTS:
                print(f"[{idx}/{total}] Skip {category}/{project_name} | in skip list")
                continue

            if (category, project_name) in processed_projects:
                print(f"[{idx}/{total}] Skip {category}/{project_name} | already in report")
                continue

            if project_output_dir.exists():
                print(f"[{idx}/{total}] Skip {category}/{project_name} | output exists: {project_output_dir}")
                continue

            print(f"[{idx}/{total}] Running {category}/{project_name}")

            status = "failed"
            reason = ""
            total_functions = 0
            total_features = 0
            description_issue_features = 0
            description_fallback_features = 0
            description_issue_feature_ids = "[]"

            try:
                summary = run_one_project(project_path, project_output_dir)
                status = summary["status"]
                reason = str(summary["reason"])
                total_functions = int(summary["total_functions"])
                total_features = int(summary["total_features"])
                description_issue_features = int(summary["description_issue_features"])
                description_fallback_features = int(summary["description_fallback_features"])
                description_issue_feature_ids = str(summary["description_issue_feature_ids"])
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

            end_ts = datetime.now()
            duration_seconds = round(time.time() - start_time, 3)

            row = {
                "category": category,
                "project": project_name,
                "project_root": str(project_path),
                "output_dir": str(project_output_dir),
                "status": status,
                "reason": reason,
                "total_functions": total_functions,
                "total_features": total_features,
                "description_issue_features": description_issue_features,
                "description_fallback_features": description_fallback_features,
                "description_issue_feature_ids": description_issue_feature_ids,
                "start_time": start_ts.isoformat(timespec="seconds"),
                "end_time": end_ts.isoformat(timespec="seconds"),
                "duration_seconds": duration_seconds,
            }
            writer.writerow(row)
            fp.flush()
            processed_projects.add((category, project_name))
            executed_count += 1

            print(
                f"[{idx}/{total}] Done {category}/{project_name} | "
                f"status={status} functions={total_functions} features={total_features} "
                f"desc_issues={description_issue_features}"
            )

    print(f"Batch finished. Report saved to: {report_file}")


if __name__ == "__main__":
    main()
