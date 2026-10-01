#!/usr/bin/env python3
"""Aggregate RepoGraph full-90 DR results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


K_VALUES = (10, 15, 20)
METRIC_KINDS = ("exact", "relaxed")


def read_projects(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_task_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok_records = [r for r in records if r.get("state") == "ok"]
    output: dict[str, Any] = {
        "tasks": len(ok_records),
        "errors": sum(1 for r in records if r.get("state") == "error"),
        "metrics": {},
    }
    for kind in METRIC_KINDS:
        output["metrics"][kind] = {}
        for k in K_VALUES:
            key = str(k)
            task_values = []
            hits = gt = pred = 0
            for record in ok_records:
                metrics = (record.get(kind) or {}).get(key) or {}
                recall = metrics.get("recall")
                if recall is not None:
                    task_values.append(float(recall))
                hits += int(metrics.get("hits", 0) or 0)
                gt += int(metrics.get("gt", 0) or 0)
                pred += int(metrics.get("pred", 0) or 0)
            macro = safe_mean(task_values)
            micro = (hits / gt) if gt else None
            precision = (hits / pred) if pred else 0.0
            f1 = (
                2 * precision * micro / (precision + micro)
                if micro is not None and precision + micro
                else 0.0
            )
            output["metrics"][kind][f"@{k}"] = {
                "tasks": len(task_values),
                "macro_dr": macro,
                "micro_dr": micro,
                "micro_precision": precision,
                "micro_f1": f1,
                "hits": hits,
                "gt": gt,
                "pred": pred,
            }
    return output


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "/"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--projects-csv", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path, required=True)
    args = parser.parse_args()

    projects = read_projects(args.projects_csv)
    expected_tasks = {row["project"]: int(row["nonempty_dependency_tasks"]) for row in projects}
    per_project_rows = []
    all_records = []

    for row in projects:
        project = row["project"]
        records = read_task_records(args.run_dir / "retrieval" / project / "task_metrics.jsonl")
        all_records.extend(records)
        aggregate = aggregate_records(records)
        per_project_rows.append(
            {
                "project": project,
                "expected_tasks": expected_tasks[project],
                "completed_tasks": aggregate["tasks"],
                "errors": aggregate["errors"],
                "exact_dr10": aggregate["metrics"]["exact"]["@10"]["macro_dr"],
                "exact_dr15": aggregate["metrics"]["exact"]["@15"]["macro_dr"],
                "exact_dr20": aggregate["metrics"]["exact"]["@20"]["macro_dr"],
                "relaxed_dr10": aggregate["metrics"]["relaxed"]["@10"]["macro_dr"],
                "relaxed_dr15": aggregate["metrics"]["relaxed"]["@15"]["macro_dr"],
                "relaxed_dr20": aggregate["metrics"]["relaxed"]["@20"]["macro_dr"],
            }
        )

    overall = aggregate_records(all_records)
    report_dir = args.run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_cache(report_dir / "overall_metrics.json", overall)

    fieldnames = [
        "project",
        "expected_tasks",
        "completed_tasks",
        "errors",
        "exact_dr10",
        "exact_dr15",
        "exact_dr20",
        "relaxed_dr10",
        "relaxed_dr15",
        "relaxed_dr20",
    ]
    with (report_dir / "per_project.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_project_rows)

    with (report_dir / "all_task_metrics.jsonl").open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# RepoGraph DevEval-90 DR Report",
        "",
        "- Scope: original DevEval-90 project list, 1146 dependency-bearing tasks.",
        '- Retrieval source: `<benchmark_root>/DevEval_no_targetMehod/Source_Code`.',
        "- Metric: task-macro and micro dependency recall at Top-10/15/20.",
        "- `Exact` uses normalized entity-signature matching without ENRE relaxations.",
        "- `Relaxed` additionally uses the project ENRE variable/module/package rules.",
        "",
        "## Overall",
        "",
        "| Metric | DR@10 | DR@15 | DR@20 |",
        "|---|---:|---:|---:|",
    ]
    for kind in METRIC_KINDS:
        lines.append(
            f"| {kind.capitalize()} macro | "
            f"{fmt(overall['metrics'][kind]['@10']['macro_dr'])} | "
            f"{fmt(overall['metrics'][kind]['@15']['macro_dr'])} | "
            f"{fmt(overall['metrics'][kind]['@20']['macro_dr'])} |"
        )
        lines.append(
            f"| {kind.capitalize()} micro | "
            f"{fmt(overall['metrics'][kind]['@10']['micro_dr'])} | "
            f"{fmt(overall['metrics'][kind]['@15']['micro_dr'])} | "
            f"{fmt(overall['metrics'][kind]['@20']['micro_dr'])} |"
        )
    lines.extend(
        [
            "",
            f"Completed tasks: {overall['tasks']} / 1146. Error records: {overall['errors']}.",
            "",
            "## Per Project",
            "",
            "| Project | Completed | Errors | Exact DR@15 | Relaxed DR@15 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in per_project_rows:
        lines.append(
            f"| {row['project']} | {row['completed_tasks']}/{row['expected_tasks']} | "
            f"{row['errors']} | {fmt(row['exact_dr15'])} | {fmt(row['relaxed_dr15'])} |"
        )
    lines.extend(["", "Raw per-task records: `report/all_task_metrics.jsonl`."])
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"completed={overall['tasks']} errors={overall['errors']} "
        f"relaxed_macro_dr@15={fmt(overall['metrics']['relaxed']['@15']['macro_dr'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
