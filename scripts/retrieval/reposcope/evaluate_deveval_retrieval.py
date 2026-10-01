#!/usr/bin/env python3
"""Evaluate RepoScope retrieval with CodeContextSearch's DevEval matching rules."""

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
import importlib.util
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType


DEFAULT_BENCHMARK = Path(
    str(_package_path('results/generation/deveval/reposcope/benchmark.jsonl'))
)
DEFAULT_PROMPTS = Path(
    str(_package_path('results/retrieval/deveval/reposcope/results/retrieval_prompts.jsonl'))
)
DEFAULT_ENRE_ROOT = (_package_path() / 'results/retrieval/deveval/featlens_and_rag')
DEFAULT_TASK_RECALL = Path(
    str(_package_path('src/generation/dev_eval/utils/task_recall.py'))
)
VIEWS = (
    "similar_fragments",
    "callers",
    "call_chains",
    "similar_functions",
    "combined",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_task_recall(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("codecontextsearch_task_recall", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import task recall module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ground_truth_dependencies(task: dict) -> list[str]:
    dependency = task.get("dependency", {})
    return [
        path
        for category in ("intra_class", "intra_file", "cross_file")
        for path in dependency.get(category, [])
    ]


def dotted_path(path: str) -> str:
    return path.replace("\\", "/").replace("/", ".").strip(".")


def normalize_init_path(path: str) -> str:
    return dotted_path(path).replace(".__init__.", ".")


def function_context(path: str, code: str, view: str) -> dict:
    return {
        "sig": path,
        "method_signature": path,
        "method_code": code,
        "source_view": view,
    }


def is_function(record: dict) -> bool:
    return record.get("type") == "function" and bool(record.get("path"))


def load_enre_function_index(path: Path) -> tuple[set[str], dict[tuple[str, int], set[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    functions = [
        element for element in data.get("variables", [])
        if element.get("category") == "Function" and element.get("qualifiedName")
    ]
    names = {element["qualifiedName"] for element in functions}
    by_location: dict[tuple[str, int], set[str]] = defaultdict(set)
    for element in functions:
        file = element.get("File")
        line = element.get("location", {}).get("startLine")
        if file and isinstance(line, int):
            by_location[(file, line)].add(element["qualifiedName"])
    return names, by_location


def canonical_enre_function(
    record: dict,
    names: set[str],
    by_location: dict[tuple[str, int], set[str]],
    audit: Counter,
) -> str | None:
    if not is_function(record):
        audit["reposcope_non_function_filtered"] += 1
        return None
    raw_name = dotted_path(record["path"])
    location_matches = by_location.get(
        (record.get("file"), int(record.get("start_lineno", -2)) + 1), set()
    )
    if len(location_matches) == 1:
        canonical = next(iter(location_matches))
        audit["enre_function_location_mapped"] += 1
        if canonical != raw_name:
            audit["qualified_name_corrected"] += 1
        return canonical
    if raw_name in names:
        audit["enre_function_name_mapped"] += 1
        return raw_name
    audit["not_enre_function_filtered"] += 1
    return None


def selected_view_contexts(
    retrieval: dict,
    fragment_rule: str,
    task_id: str,
    enre_function_names: set[str],
    enre_functions_by_location: dict[tuple[str, int], set[str]],
    mapping_audit: Counter,
) -> dict[str, list[dict]]:
    selected = retrieval["selected"]
    fragment_key = (
        "defined_entities" if fragment_rule == "defined" else "overlapping_entities"
    )
    views: dict[str, list[dict]] = {name: [] for name in VIEWS[:-1]}

    def append_context(view: str, record: dict, code: str) -> None:
        signature = canonical_enre_function(
            record, enre_function_names, enre_functions_by_location, mapping_audit
        )
        if signature is not None:
            views[view].append(function_context(signature, code, view))

    for fragment in selected["similar_fragments"]:
        for node in fragment[fragment_key]:
            append_context("similar_fragments", node, fragment["context"])
    for record in selected["callers"]:
        append_context("callers", record, record["serialized_context"])
    for chain in selected["call_chains"]:
        for node in chain["nodes"]:
            append_context("call_chains", node, chain["serialized_context"])
    for record in selected["similar_functions"]:
        append_context(
            "similar_functions", record, record["serialized_context"]
        )

    target = retrieval.get("target_function")
    target_signature = dotted_path(target["path"]) if target else None
    target_signatures = {task_id, target_signature}
    target_signatures.update(
        signature.replace(".__init__.", ".")
        for signature in list(target_signatures) if signature
    )
    for view in VIEWS[:-1]:
        views[view] = merge_function_contexts(
            context for context in views[view]
            if context["method_signature"] not in target_signatures
        )
    views["combined"] = merge_function_contexts(
        context for view in VIEWS[:-1] for context in views[view]
    )
    return views


def merge_function_contexts(contexts) -> list[dict]:
    merged: dict[str, dict] = {}
    codes: dict[str, list[str]] = defaultdict(list)
    source_views: dict[str, set[str]] = defaultdict(set)
    for context in contexts:
        signature = context["method_signature"]
        if signature not in merged:
            merged[signature] = dict(context)
        code = context.get("method_code", "")
        if code and code not in codes[signature]:
            codes[signature].append(code)
        context_views = context["source_view"]
        if isinstance(context_views, str):
            source_views[signature].add(context_views)
        else:
            source_views[signature].update(context_views)
    for signature, context in merged.items():
        context["method_code"] = "\n\n".join(codes[signature])
        context["source_view"] = sorted(source_views[signature])
    return [merged[signature] for signature in sorted(merged)]


def matched_dependencies(
    dependencies: list[str], contexts: list[dict], task_recall: ModuleType
) -> tuple[set[str], Counter]:
    def normalized_signature(context: dict) -> str:
        signature = str(context.get("method_signature", ""))
        return signature.split("(", 1)[0].replace(".__init__", "")

    retrieved = {normalized_signature(context) for context in contexts}
    matched = set(dependencies) & retrieved
    match_rules = Counter({"function_exact": len(matched)})
    for dependency in dependencies:
        if dependency in matched:
            continue
        if dependency in task_recall.variables_enre:
            variable = dependency.rsplit(".", 1)[-1]
            if any(variable in context.get("method_code", "") for context in contexts):
                matched.add(dependency)
                match_rules["variable_code"] += 1
        elif dependency in task_recall.unresolved_attribute_enre:
            class_name, attribute = dependency.rsplit(".", 1)
            if any(
                str(context.get("sig", "")).startswith(f"{class_name}.")
                and f"self.{attribute}" in context.get("method_code", "")
                for context in contexts
            ):
                matched.add(dependency)
                match_rules["unresolved_attribute"] += 1
        elif dependency in task_recall.module_enre:
            if any(str(context.get("sig", "")).startswith(dependency) for context in contexts):
                matched.add(dependency)
                match_rules["module_prefix"] += 1
        elif dependency in task_recall.package_enre:
            if any(str(context.get("sig", "")).startswith(dependency) for context in contexts):
                matched.add(dependency)
                match_rules["package_prefix"] += 1
    return matched, match_rules


def task_metrics(
    dependencies: list[str], contexts: list[dict], task_recall: ModuleType
) -> dict:
    recall_info = task_recall.compute_task_recall(dependencies, contexts)
    matched, match_rules = matched_dependencies(dependencies, contexts, task_recall)
    if recall_info["dependency_hit"] != len(matched):
        raise RuntimeError(
            "Local match audit differs from task_recall.compute_task_recall: "
            f"{len(matched)} != {recall_info['dependency_hit']}"
        )
    relevant = recall_info["dependency_total"]
    retrieved = len(contexts)
    true_positive = recall_info["dependency_hit"]
    precision = true_positive / retrieved if retrieved else 0.0
    recall = recall_info["recall"]
    f1 = (
        2 * precision * recall / (precision + recall)
        if recall is not None and precision + recall > 0 else 0.0
    )
    dependency_set = set(dependencies)
    return {
        "relevant": relevant,
        "retrieved_functions": retrieved,
        "true_positive": true_positive,
        "precision": precision,
        "dependency_recall": recall,
        "f1": f1 if relevant else None,
        "match_rules": dict(sorted(match_rules.items())),
        "retrieved_function_signatures": [
            context["method_signature"] for context in contexts
        ],
        "matched_dependencies": sorted(matched),
        "missed_dependencies": sorted(dependency_set - matched),
    }


def ratio_metrics(true_positive: int, retrieved: int, relevant: int) -> dict:
    precision = true_positive / retrieved if retrieved else 0.0
    recall = true_positive / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "relevant": relevant,
        "retrieved_functions": retrieved,
        "true_positive": true_positive,
        "precision": precision,
        "dependency_recall": recall,
        "f1": f1,
    }


def aggregate(per_task: list[dict], view: str) -> dict:
    metrics = [item["views"][view] for item in per_task]
    nonempty = [metric for metric in metrics if metric["relevant"]]
    summed = ratio_metrics(
        sum(metric["true_positive"] for metric in metrics),
        sum(metric["retrieved_functions"] for metric in metrics),
        sum(metric["relevant"] for metric in metrics),
    )
    macro = {
        "tasks": len(nonempty),
        "precision": statistics.mean(metric["precision"] for metric in nonempty),
        "dependency_recall": statistics.mean(metric["dependency_recall"] for metric in nonempty),
        "f1": statistics.mean(metric["f1"] for metric in nonempty),
    } if nonempty else {"tasks": 0, "precision": 0.0, "dependency_recall": 0.0, "f1": 0.0}
    return {"project_accumulated": summed, "macro_nonempty_ground_truth": macro}


def project_weighted(projects: dict[str, dict], view: str) -> dict:
    total_tasks = sum(project["task_count"] for project in projects.values())
    precision = sum(
        project["views"][view]["project_accumulated"]["precision"] * project["task_count"]
        for project in projects.values()
    ) / total_tasks
    recall = sum(
        project["views"][view]["project_accumulated"]["dependency_recall"] * project["task_count"]
        for project in projects.values()
    ) / total_tasks
    f1 = sum(
        project["views"][view]["project_accumulated"]["f1"] * project["task_count"]
        for project in projects.values()
    ) / total_tasks
    f1_from_weighted_precision_recall = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return {
        "tasks": total_tasks,
        "precision": precision,
        "dependency_recall": recall,
        "f1": f1,
        "f1_from_weighted_precision_recall": f1_from_weighted_precision_recall,
        "note": "Per-project accumulated P, DR, and F1 are each weighted by the project's task count.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--enre-root", type=Path, default=DEFAULT_ENRE_ROOT)
    parser.add_argument("--task-recall", type=Path, default=DEFAULT_TASK_RECALL)
    parser.add_argument(
        "--fragment-entity-rule", choices=("defined", "overlapping"), default="defined"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    task_recall = load_task_recall(args.task_recall)
    benchmark = {item["namespace"]: item for item in load_jsonl(args.benchmark)}
    prompts = {item["task_id"]: item for item in load_jsonl(args.prompts)}
    if set(benchmark) != set(prompts):
        raise RuntimeError(
            "Benchmark and prompt task IDs differ: "
            f"missing={len(set(benchmark) - set(prompts))}, "
            f"unexpected={len(set(prompts) - set(benchmark))}"
        )

    tasks_by_project: dict[str, list[dict]] = defaultdict(list)
    for task in benchmark.values():
        tasks_by_project[task["project_path"]].append(task)

    per_task = []
    enre_paths = {}
    mapping_audits: dict[str, Counter] = {}
    for project_path, project_tasks in sorted(tasks_by_project.items()):
        enre_path = args.enre_root / project_path / "report-enre.json"
        if not enre_path.is_file():
            raise FileNotFoundError(f"Project ENRE report not found: {enre_path}")
        enre_paths[project_path] = str(enre_path)
        task_recall.clear_enre_elements()
        task_recall.load_enre_elements(str(enre_path))
        enre_function_names, enre_functions_by_location = load_enre_function_index(
            enre_path
        )
        mapping_audit = Counter()
        mapping_audits[project_path] = mapping_audit
        for task in sorted(project_tasks, key=lambda item: item["namespace"]):
            task_id = task["namespace"]
            dependencies = ground_truth_dependencies(task)
            target = prompts[task_id]["metadata"]["retrieval"].get("target_function")
            target_path = dotted_path(target["path"]) if target else None
            contexts = selected_view_contexts(
                prompts[task_id]["metadata"]["retrieval"],
                args.fragment_entity_rule,
                task_id,
                enre_function_names,
                enre_functions_by_location,
                mapping_audit,
            )
            per_task.append({
                "task_id": task_id,
                "project_path": project_path,
                "retrieved_target_function": target_path,
                "target_path_matches_task": bool(
                    target_path and normalize_init_path(target_path) == task_id
                ),
                "ground_truth_dependencies": sorted(set(dependencies)),
                "views": {
                    view: task_metrics(dependencies, contexts[view], task_recall)
                    for view in VIEWS
                },
            })

    project_summaries = {}
    for project_path, project_tasks in sorted(tasks_by_project.items()):
        result_tasks = [item for item in per_task if item["project_path"] == project_path]
        project_summaries[project_path] = {
            "task_count": len(project_tasks),
            "ground_truth_nonempty_tasks": sum(
                bool(item["ground_truth_dependencies"]) for item in result_tasks
            ),
            "views": {view: aggregate(result_tasks, view) for view in VIEWS},
        }

    rule_totals = {
        view: dict(sorted(sum(
            (Counter(item["views"][view]["match_rules"]) for item in per_task), Counter()
        ).items()))
        for view in VIEWS
    }
    summary = {
        "task_count": len(per_task),
        "project_task_counts": {
            project: len(tasks) for project, tasks in sorted(tasks_by_project.items())
        },
        "ground_truth_nonempty_tasks": sum(
            bool(item["ground_truth_dependencies"]) for item in per_task
        ),
        "ground_truth_empty_tasks": sum(
            not item["ground_truth_dependencies"] for item in per_task
        ),
        "target_mapping_audit": {
            "matched": sum(item["target_path_matches_task"] for item in per_task),
            "mismatched": sum(
                not item["target_path_matches_task"] for item in per_task
            ),
            "mismatched_tasks": [
                {
                    "task_id": item["task_id"],
                    "retrieved_target_function": item["retrieved_target_function"],
                }
                for item in per_task if not item["target_path_matches_task"]
            ],
        },
        "matching_policy": {
            "task_recall_module": str(args.task_recall.resolve()),
            "enre_reports": enre_paths,
            "fragment_entity_rule": args.fragment_entity_rule,
            "function_context_filter": (
                "RepoScope node type == function and mapped ENRE category == Function"
            ),
            "function_mapping": "Map by source file and 1-based definition line to ENRE category == Function; fall back to exact qualifiedName; exclude unmapped records",
            "signature_normalization": "remove parameters; remove .__init__",
            "special_rules": [
                "Variable: variable name occurs in retrieved function-context text",
                "Unresolved Attribute: function signature is in the class and text contains self.attribute",
                "Module/Package: module or package is a retrieved function-signature prefix",
            ],
            "precision_denominator": (
                "Number of mapped, deduplicated ENRE Function contexts; RepoScope "
                "has no equivalent of the GML evaluator's all-node count"
            ),
            "metric_name": "DR_context (RepoScope has no unified @N function ranking)",
        },
        "function_mapping_audit": {
            "all_projects": dict(sorted(sum(mapping_audits.values(), Counter()).items())),
            "projects": {
                project: dict(sorted(audit.items()))
                for project, audit in sorted(mapping_audits.items())
            },
        },
        "match_rule_totals": rule_totals,
        "views": {view: aggregate(per_task, view) for view in VIEWS},
        "user_project_task_weighted": {
            view: project_weighted(project_summaries, view) for view in VIEWS
        },
        "projects": project_summaries,
    }

    output_dir = args.output_dir or args.prompts.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "retrieval_metrics_user_logic.json"
    task_path = output_dir / "retrieval_metrics_user_logic_by_task.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with task_path.open("w", encoding="utf-8") as handle:
        for item in sorted(per_task, key=lambda record: record["task_id"]):
            handle.write(json.dumps(item) + "\n")
    print(json.dumps({
        "task_count": summary["task_count"],
        "combined_project_weighted": summary["user_project_task_weighted"]["combined"],
        "combined_project_accumulated": summary["views"]["combined"]["project_accumulated"],
        "combined_macro": summary["views"]["combined"]["macro_nonempty_ground_truth"],
        "combined_match_rules": summary["match_rule_totals"]["combined"],
        "summary": str(summary_path),
        "per_task": str(task_path),
    }, indent=2))


if __name__ == "__main__":
    main()
