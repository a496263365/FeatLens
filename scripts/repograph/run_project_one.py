#!/usr/bin/env python3
"""Run RepoGraph retrieval and DR scoring for one full DevEval project."""

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
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = (_package_path())
REPOGRAPH_DIR = REPO_ROOT / "implementation/baselines/RepoGraph/repograph"
sys.path.insert(0, str(REPOGRAPH_DIR))

from context_search import GraphContextSearcher, llm_generate_search_terms, rank_aggregate_one_task  # noqa: E402
from utils.dev_eval_task import parse_task  # noqa: E402
from utils.llm_clients import BackendName, make_client  # noqa: E402
from utils.source_code_utils import resolve_signature  # noqa: E402
from utils.task_recall import compute_task_recall, load_enre_elements  # noqa: E402


K_VALUES = (10, 15, 20)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def normalize_signature(value: Any) -> str:
    text = str(value or "")
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.replace(".__init__", "")


def canonicalize_signature(value: Any, project_heads: set[str]) -> str:
    """Remove repository-root prefixes introduced by different graph roots.

    RepoGraph's original five-project graphs use the main package directory as
    their root. A full benchmark may contain tasks from several modules, so we
    canonicalize a graph qualified name to the shortest suffix whose first
    component is one of the task namespace heads.
    """
    text = normalize_signature(value)
    parts = text.split(".")
    for index in range(len(parts)):
        candidate = ".".join(parts[index:])
        if candidate.split(".", 1)[0] in project_heads:
            return candidate
    return text


def canonicalize_context(
    context: list[dict[str, Any]], project_heads: set[str]
) -> list[dict[str, Any]]:
    canonical = []
    for item in context:
        updated = dict(item)
        raw = updated.get("method_signature") or updated.get("sig")
        normalized = canonicalize_signature(raw, project_heads)
        updated["raw_method_signature"] = str(raw or "")
        updated["method_signature"] = normalized
        updated["sig"] = normalized
        canonical.append(updated)
    return canonical


def exact_metrics(dependency: list[str], context: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top = context[:k]
    ground_truth = set(dependency or [])
    retrieved = {
        normalize_signature(item.get("method_signature") or item.get("sig"))
        for item in top
        if item.get("method_signature") or item.get("sig")
    }
    hits = len(ground_truth & retrieved)
    total = len(ground_truth)
    precision = hits / len(top) if top else 0.0
    recall = hits / total if total else None
    f1 = (2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else 0.0
    return {
        "hits": hits,
        "gt": total,
        "pred": len(top),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def relaxed_metrics(dependency: list[str], context: list[dict[str, Any]], k: int) -> dict[str, Any]:
    top = context[:k]
    result = compute_task_recall(dependency, top)
    hits = int(result["dependency_hit"])
    total = int(result["dependency_total"])
    precision = hits / len(top) if top else 0.0
    recall = result["recall"]
    f1 = (2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else 0.0
    return {
        "hits": hits,
        "gt": total,
        "pred": len(top),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def load_project_tasks(task_file: Path, project: str) -> list[dict[str, Any]]:
    tasks = []
    with task_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("project_path") == project:
                tasks.append(record)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enre-path", type=Path)
    parser.add_argument("--backend", default="openai", choices=["openai", "ollama", "mock"])
    parser.add_argument("--model", default='deepseek_v3_2')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-terms", type=int, default=5)
    parser.add_argument("--max-errors", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_metrics_path = args.output_dir / "task_metrics.jsonl"
    status_path = args.output_dir / "status.json"

    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("state") == "complete":
                return 0
        except Exception:
            pass

    tasks = load_project_tasks(args.tasks_file, args.project)
    project_heads = {
        str(record.get("namespace", "")).split(".", 1)[0]
        for record in tasks
        if record.get("namespace")
    }
    done: set[str] = set()
    if task_metrics_path.is_file():
        with task_metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("state") == "ok" and record.get("namespace"):
                    done.add(str(record["namespace"]))

    status: dict[str, Any] = {
        "project": args.project,
        "state": "running",
        "total_tasks": len(tasks),
        "completed_tasks": len(done),
        "error_tasks": 0,
        "model": args.model,
        "backend": args.backend,
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        "started_at_epoch": time.time(),
        "updated_at_epoch": time.time(),
    }
    atomic_json(status_path, status)

    graph_path = args.graph_dir / "graph.pkl"
    tags_path = args.graph_dir / "tags.json"
    for path in (graph_path, tags_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.enre_path and args.enre_path.is_file():
        load_enre_elements(str(args.enre_path))

    searcher = GraphContextSearcher(str(graph_path), str(tags_path))
    client = make_client(
        backend=args.backend,  # type: ignore[arg-type]
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=None,
        timeout_s=args.timeout_s,
    )

    started = time.time()
    error_count = int(status["error_tasks"])
    for index, record in enumerate(tasks, start=1):
        namespace = str(record.get("namespace", ""))
        if namespace in done:
            continue

        status["current_namespace"] = namespace
        status["current_index"] = index
        status["updated_at_epoch"] = time.time()
        atomic_json(status_path, status)

        try:
            task = parse_task(record)
            abs_file, signature = resolve_signature(
                str(args.source_root), task.completion_path, task.signature_position
            )
            search_terms, usage = llm_generate_search_terms(
                client,
                requirement_text=task.requirement_text,
                signature=signature,
                max_terms=args.max_terms,
            )
            if not search_terms:
                fallback = task.namespace.split(".")[-1]
                search_terms = [fallback] if fallback else []

            agg = rank_aggregate_one_task(searcher, search_terms)
            ranked_nodes = agg["ranked_nodes"]
            centers = [x for x in ranked_nodes if x.get("role") == "center"]
            neighbors = [x for x in ranked_nodes if x.get("role") != "center"]
            ranked_nodes = centers + neighbors

            contexts = []
            for item in ranked_nodes:
                ctx = searcher.node_to_context(item["node"])
                ctx["rank_info"] = item
                contexts.append(ctx)
            contexts = canonicalize_context(contexts, project_heads)

            exact = {str(k): exact_metrics(task.dependency or [], contexts, k) for k in K_VALUES}
            relaxed = {str(k): relaxed_metrics(task.dependency or [], contexts, k) for k in K_VALUES}
            predictions = [
                normalize_signature(item.get("method_signature") or item.get("sig"))
                for item in contexts[: max(K_VALUES)]
            ]

            out_record = {
                "state": "ok",
                "project": args.project,
                "namespace": namespace,
                "file": abs_file,
                "search_terms": search_terms,
                "token_usage": usage,
                "exact": exact,
                "relaxed": relaxed,
                "top_predictions": predictions,
            }
            append_jsonl(task_metrics_path, out_record)
            done.add(namespace)
            status["completed_tasks"] = len(done)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                status["tokens"][key] += int(usage.get(key, 0))
            status["tokens"]["calls"] += 1
        except Exception as exc:
            error_count += 1
            append_jsonl(
                task_metrics_path,
                {
                    "state": "error",
                    "project": args.project,
                    "namespace": namespace,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
            )
            if error_count >= args.max_errors:
                status.update(
                    {
                        "state": "failed",
                        "error_tasks": error_count,
                        "elapsed_seconds": time.time() - started,
                        "updated_at_epoch": time.time(),
                    }
                )
                atomic_json(status_path, status)
                return 1

        status["error_tasks"] = error_count
        status["elapsed_seconds"] = time.time() - started
        status["updated_at_epoch"] = time.time()
        atomic_json(status_path, status)

    if len(done) != len(tasks):
        status.update(
            {
                "state": "incomplete",
                "completed_tasks": len(done),
                "error_tasks": error_count,
                "elapsed_seconds": time.time() - started,
                "updated_at_epoch": time.time(),
            }
        )
        atomic_json(status_path, status)
        return 1

    status.update(
        {
            "state": "complete",
            "completed_tasks": len(done),
            "error_tasks": error_count,
            "elapsed_seconds": time.time() - started,
            "updated_at_epoch": time.time(),
        }
    )
    status.pop("current_namespace", None)
    status.pop("current_index", None)
    atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
