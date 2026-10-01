#!/usr/bin/env python3
"""Evaluate RepoScope's selected entities on EvoCodeBench.

The evaluator uses the selected dependency-candidate entity paths stored by
RepoScope in ``retrieval_prompts.jsonl``. It reports both task-macro DR over
nonempty-dependency tasks and the project/task-count-weighted metric used by
the DevEval RepoScope reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEPENDENCY_KEYS = ("intra_class", "intra_file", "cross_file")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_path(value: str) -> str:
    value = str(value).replace("\\", "/").strip()
    if value.endswith(".py"):
        value = value[:-3]
    return value.replace("/", ".").strip(".").replace(".__init__.", ".")


def path_variants(value: str) -> set[str]:
    normalized = normalize_path(value)
    variants = {normalized}
    if normalized.startswith("src."):
        variants.add(normalized.removeprefix("src."))
    return variants


def dependencies(task: dict[str, Any]) -> list[str]:
    dependency = task.get("dependency") or {}
    return [
        str(item)
        for key in DEPENDENCY_KEYS
        for item in dependency.get(key, [])
    ]


def selected_paths(prompt: dict[str, Any]) -> list[str]:
    retrieval = (prompt.get("metadata") or {}).get("retrieval") or {}
    paths = retrieval.get("selected_dependency_candidate_paths")
    if paths is None:
        paths = retrieval.get("selected_entity_paths") or []
    return sorted({
        variant
        for path in paths
        if str(path).strip()
        for variant in path_variants(path)
    })


def dependency_hit(dependency: str, paths: set[str]) -> bool:
    dependency = normalize_path(dependency)
    if dependency in paths:
        return True
    # A selected class member also demonstrates that its owning class was
    # selected. The reverse is intentionally not treated as a method hit.
    return any(path.startswith(dependency + ".") for path in paths)


def evaluate(
    benchmark: dict[str, dict[str, Any]],
    prompts: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    missing_benchmark = sorted(set(benchmark) - set(prompts))
    unexpected_prompts = sorted(set(prompts) - set(benchmark))
    if missing_benchmark or unexpected_prompts:
        raise ValueError(
            f"Benchmark/prompt mismatch: missing={len(missing_benchmark)}, "
            f"unexpected={len(unexpected_prompts)}"
        )

    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    for namespace in sorted(benchmark):
        task = benchmark[namespace]
        prompt = prompts[namespace]
        ground_truth = sorted(set(dependencies(task)))
        paths = set(selected_paths(prompt))
        hits = sorted(
            dependency for dependency in ground_truth
            if dependency_hit(dependency, paths)
        )
        relevant = len(ground_truth)
        recall = len(hits) / relevant if relevant else None
        result = {
            "namespace": namespace,
            "project_path": task.get("project_path", ""),
            "relevant": relevant,
            "retrieved_entity_paths": len(paths),
            "hit_count": len(hits),
            "dependency_recall": recall,
            "hits": hits,
            "misses": sorted(set(ground_truth) - set(hits)),
            "selected_paths": sorted(paths),
        }
        by_task.append(result)
        by_project[result["project_path"]].append(result)
        rule_counts["relevant_dependencies"] += relevant
        rule_counts["hit_dependencies"] += len(hits)

    nonempty = [row for row in by_task if row["relevant"]]
    macro_department = (
        statistics.mean(row["dependency_recall"] for row in nonempty)
        if nonempty else 0.0
    )
    micro_department = (
        rule_counts["hit_dependencies"] / rule_counts["relevant_dependencies"]
        if rule_counts["relevant_dependencies"] else 0.0
    )

    project_rows: list[dict[str, Any]] = []
    for project, rows in sorted(by_project.items()):
        relevant = sum(row["relevant"] for row in rows)
        hits = sum(row["hit_count"] for row in rows)
        nonempty_rows = [row for row in rows if row["relevant"]]
        project_rows.append({
            "project_path": project,
            "tasks": len(rows),
            "nonempty_dependency_tasks": len(nonempty_rows),
            "relevant": relevant,
            "hits": hits,
            "project_dependency_recall": hits / relevant if relevant else 0.0,
        })

    total_tasks = len(by_task)
    weighted_department = sum(
        row["tasks"] * row["project_dependency_recall"] for row in project_rows
    ) / total_tasks

    summary = {
        "dataset": "EvoCodeBench five projects",
        "tasks": total_tasks,
        "tasks_with_dependencies": len(nonempty),
        "tasks_without_dependencies": total_tasks - len(nonempty),
        "relevant_dependencies": rule_counts["relevant_dependencies"],
        "hit_dependencies": rule_counts["hit_dependencies"],
        "macro_dr_nonempty_tasks": macro_department,
        "micro_dr": micro_department,
        "project_task_weighted_dr": weighted_department,
        "matching_rule": (
            "Normalized selected entity path equals a dependency, or a selected "
            "entity is a member of a dependency class. Target paths are excluded "
            "by RepoScope's selected_dependency_candidate_paths field."
        ),
        "projects": project_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "by_task.jsonl").open("w", encoding="utf-8") as handle:
        for row in by_task:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow({
            key: value if not isinstance(value, (list, dict)) else json.dumps(value, ensure_ascii=False)
            for key, value in summary.items()
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    benchmark = {row["namespace"]: row for row in load_jsonl(args.benchmark)}
    prompts = {row["task_id"]: row for row in load_jsonl(args.prompts)}
    summary = evaluate(benchmark, prompts, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
