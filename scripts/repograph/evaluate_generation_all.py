#!/usr/bin/env python3
"""Evaluate RepoGraph full-90 generations with the original DevEval evaluators."""

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
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = (_package_path())
PROJECTS_CSV = (_package_path() / 'data/deveval/projects_90.csv')
TASKS_FILE = (_package_path() / 'data/deveval/tasks_all_1430.jsonl')
PASS_SCRIPT = (_package_path() / 'scripts/metrics/deveval/pass_k.py')
RECALL_SCRIPT = (_package_path() / 'scripts/metrics/deveval/parser/recall_k.py')
PASS_SOURCE = (_package_path() / 'results/generation/deveval/locagent/test_source_pass')
RECALL_SOURCE = (_package_path() / 'results/generation/deveval/locagent/test_source_recall')
DEPENDENCY_ROOT = _external_path('FEATLENS_DEPENDENCY_DATA_ROOT', 'external/deveval/dependency_data')
EVAL_PYTHON = _PortablePath(_evaluation_python())
DEFAULT_RUN_DIR = REPO_ROOT / "results/repograph_full90_generation_20261001"


def read_projects(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [row["project"].strip() for row in csv.DictReader(f)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def deps(record: dict[str, Any]) -> set[str]:
    dep = record.get("dependency") or {}
    return {
        item
        for values in dep.values()
        for item in (values or [])
    }


def completion_token_len(code: str) -> int:
    import io
    import textwrap
    import tokenize

    skip = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
    text = textwrap.dedent(code or "").strip("\n") + "\n"
    count = 0
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type not in skip:
                count += 1
    except (tokenize.TokenError, IndentationError, TabError):
        pass
    return count


def run_project(
    model: str,
    project: str,
    run_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    destination = run_dir / "evaluation" / model / project
    destination.mkdir(parents=True, exist_ok=True)
    done_path = destination / "status.json"
    if done_path.is_file():
        try:
            status = json.loads(done_path.read_text(encoding="utf-8"))
            if status.get("state") == "complete":
                return status
        except Exception:
            pass

    completion_path = (run_dir / "generations" / model / project / "completion.jsonl").resolve()
    data_path = (
        (_package_path() / 'data/deveval/projects') / project / "filtered.jsonl"
    ).resolve()
    pass_log = (destination / "pass_output.jsonl").resolve()
    recall_log = (destination / "recall_output.jsonl").resolve()
    pass_stdout = (destination / "pass.log").resolve()
    recall_stdout = (destination / "recall.log").resolve()
    dependency_tmp = (destination / "dependency_tmp").resolve()

    env = os.environ.copy()
    eval_bin = str(EVAL_PYTHON.parent)
    env["PATH"] = eval_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    started = time.time()
    pass_cmd = [
        str(EVAL_PYTHON),
        str(PASS_SCRIPT),
        "--output_file",
        str(completion_path),
        "--log_file",
        str(pass_log),
        "--source_code_root",
        str(PASS_SOURCE),
        "--data_file",
        str(data_path),
        "--k",
        "1",
        "--n",
        "1",
        "--failure_log",
        str((destination / "pass_failure.log").resolve()),
    ]
    recall_cmd = [
        str(EVAL_PYTHON),
        str(RECALL_SCRIPT),
        "--output_file",
        str(completion_path),
        "--log_file",
        str(recall_log),
        "--source_code_root",
        str(RECALL_SOURCE),
        "--data_file",
        str(data_path),
        "--dependency_data_root",
        str(DEPENDENCY_ROOT),
        "--dependency_tmp_dir",
        str(dependency_tmp),
        "--k",
        "1",
    ]

    status: dict[str, Any] = {
        "model": model,
        "project": project,
        "state": "running",
        "started_at_epoch": started,
    }
    done_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for stage, cmd, stdout_path in (
        ("pass", pass_cmd, pass_stdout),
        ("recall", recall_cmd, recall_stdout),
    ):
        with stdout_path.open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        status[f"{stage}_returncode"] = returncode
        if returncode != 0:
            status["state"] = "failed"
            status["failed_stage"] = stage
            status["elapsed_seconds"] = time.time() - started
            done_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return status

    status["state"] = "complete"
    status["elapsed_seconds"] = time.time() - started
    done_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status


def aggregate_model(model: str, run_dir: Path) -> dict[str, Any]:
    benchmark = {record["namespace"]: record for record in read_jsonl(TASKS_FILE)}
    pass_records: dict[str, dict[str, Any]] = {}
    recall_records: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}

    for project in read_projects(PROJECTS_CSV):
        evaluation_dir = run_dir / "evaluation" / model / project
        for record in read_jsonl(evaluation_dir / "pass_output.jsonl"):
            pass_records[record["namespace"]] = record
        for record in read_jsonl(evaluation_dir / "recall_output.jsonl"):
            recall_records[record["namespace"]] = record
        for record in read_jsonl(run_dir / "generations" / model / project / "completion.jsonl"):
            completions[record["namespace"]] = record

    dependency_names = {
        namespace for namespace, record in benchmark.items() if deps(record)
    }
    ratios = []
    hits = total = 0
    for namespace in dependency_names:
        reference = deps(benchmark[namespace])
        generated = recall_records.get(namespace, {}).get("generated_dependency") or {}
        predicted = deps({"dependency": generated})
        ratio = (len(reference & predicted) / len(reference)) if reference else 0.0
        ratios.append(ratio)
        hits += len(reference & predicted)
        total += len(reference)

    passed = sum(record.get("Result") == "Pass" for record in pass_records.values())
    loc_values = [
        completion_token_len(record.get("completion", ""))
        for record in completions.values()
    ]
    return {
        "model": model,
        "pass_evaluated": len(pass_records),
        "pass_passed": passed,
        "pass_at_1_percent": 100.0 * passed / 1430 if pass_records else None,
        "dir_evaluated": len(recall_records),
        "dir_macro_percent": 100.0 * sum(ratios) / len(ratios) if ratios else None,
        "dir_micro_percent": 100.0 * hits / total if total else None,
        "loc_lexical_tokens": sum(loc_values) / len(loc_values) if loc_values else None,
        "completion_tasks": len(completions),
    }


def write_report(run_dir: Path, summaries: list[dict[str, Any]], phase: str) -> None:
    lines = [
        "# RepoGraph Full90 Generation Evaluation",
        "",
        f"- Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Phase: {phase}",
        "- Generation contexts: RepoGraph Top-15",
        "",
        "| Model | Pass@1 (%) | DIR macro (%) | DIR micro (%) | LOC (lexical tokens) | Tasks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['pass_at_1_percent']:.2f} | "
            f"{summary['dir_macro_percent']:.2f} | {summary['dir_micro_percent']:.2f} | "
            f"{summary['loc_lexical_tokens']:.2f} | {summary['completion_tasks']} |"
        )
    lines.extend(["", "Per-project details are in `evaluation/<model>/<project>/status.json`."])
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    (run_dir / "report" / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evaluation_progress(
    run_dir: Path, model: str, projects: list[str], phase: str
) -> None:
    rows = []
    complete = running = failed = pending = 0
    for project in projects:
        status_path = run_dir / "evaluation" / model / project / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        state = str(status.get("state") or "pending")
        if state == "complete":
            complete += 1
        elif state == "running":
            running += 1
        elif state == "failed":
            failed += 1
        else:
            pending += 1
        rows.append(
            f"| {project} | {state} | {status.get('pass_returncode', '/')} | "
            f"{status.get('recall_returncode', '/')} |"
        )
    text = "\n".join(
        [
            "# RepoGraph Full90 Evaluation Progress",
            "",
            f"- Updated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"- Model: `{model}`",
            f"- Phase: `{phase}`",
            "",
            f"- Complete projects: {complete}/{len(projects)}",
            f"- Running: {running}",
            f"- Failed: {failed}",
            f"- Pending: {pending}",
            "",
            "| Project | State | Pass rc | DIR rc |",
            "|---|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    tmp = run_dir / "EVALUATION_PROGRESS.md.tmp"
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(run_dir / "EVALUATION_PROGRESS.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--models", default="deepseek-v3.2,gpt-5-mini")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    projects = read_projects(PROJECTS_CSV)
    summaries = []
    for model in models:
        pending = list(projects)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_project, model, project, args.run_dir, args.timeout): project
                for project in pending
            }
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                print(
                    f"[{model}] {status['project']}: {status['state']}",
                    flush=True,
                )
                write_evaluation_progress(
                    args.run_dir, model, projects, "evaluation"
                )
        summaries.append(aggregate_model(model, args.run_dir))
        write_report(args.run_dir, summaries, f"{model}_complete")
        write_evaluation_progress(args.run_dir, model, projects, "finished")

    (args.run_dir / "report").mkdir(parents=True, exist_ok=True)
    (args.run_dir / "report" / "overall_metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(args.run_dir, summaries, "finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
