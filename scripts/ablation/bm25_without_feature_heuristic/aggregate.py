#!/usr/bin/env python3
"""Aggregate the four missing graph ablations and build an updated table."""

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


import csv
import json
import statistics
from pathlib import Path

ROOT = (_package_path())
OUT = (_package_path() / 'results/ablation/deveval/bm25_without_feature_heuristic')
CONTROLLED = (_package_path() / 'results/ablation/deveval/bm25_and_unixcoder')
COMBINED = (
    ROOT
    / "results/ablation_with_locagent_newdr_20260930_044642"
    / "combined_new_relaxed_dr.csv"
)
HISTORICAL = (_package_path() / 'results/retrieval/combined/featlens_scores')


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values)


def read_project_metrics(project):
    path = (_package_path() / 'results/ablation/deveval/bm25_without_feature_heuristic/project_results') / project / "task_metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compact(rows, arm, direct_dr=None, seed_dr=None, final_dr=None, exact_dr=None):
    selected = [row for row in rows if row["arm"] == arm]
    seed = [row for row in selected if row["stage"] == "seed"]
    final = [row for row in selected if row["stage"] == "final15"]
    assert len(seed) == len(final) == 1146
    return dict(
        arm=arm,
        tasks=len(final),
        projects=len({row["project_path"] for row in final}),
        direct15_dr=direct_dr,
        seed_dr=(
            mean([row["dr"] for row in seed])
            if seed_dr is None
            else seed_dr
        ),
        final15_dr=(
            mean([row["dr"] for row in final])
            if final_dr is None
            else final_dr
        ),
        final_exact_dr=(
            mean([row["exact_dr"] for row in final])
            if exact_dr is None
            else exact_dr
        ),
        mean_final_nodes=mean([row["n"] for row in final]),
    )


def main():
    if not ((_package_path() / 'results/ablation/deveval/bm25_without_feature_heuristic/run_finished.json')).exists():
        raise SystemExit("Full missing-ablation run has not finished")
    scope = json.loads(((_package_path() / 'results/ablation/deveval/bm25_and_unixcoder/primary_scope.json')).read_text())["projects"]
    all_rows = []
    for project in scope:
        all_rows.extend(read_project_metrics(project))
    assert len(all_rows) == 4 * 2 * 1146

    previous = {row["method"]: row for row in read_csv(COMBINED)}
    historical = json.loads(((_package_path() / 'results/retrieval/combined/featlens_scores/summary.json')).read_text())
    historical_seed = json.loads(
        ((_package_path() / 'results/retrieval/combined/featlens_scores/seed_scores/summary.json')).read_text()
    )

    missing = [
        compact(
            all_rows,
            "bm25_code_top15",
            direct_dr=float(previous["BM25代码：same-feature✓"]["direct15_dr"]),
            seed_dr=float(previous["BM25代码：same-feature✓"]["seed_dr"]),
        ),
        compact(
            all_rows,
            "bm25_description_top15",
            direct_dr=float(previous["BM25摘要：same-feature✓"]["direct15_dr"]),
            seed_dr=float(previous["BM25摘要：same-feature✓"]["seed_dr"]),
        ),
        compact(
            all_rows,
            "bm25_signature_top15",
            direct_dr=float(previous["BM25签名：same-feature✓"]["direct15_dr"]),
            seed_dr=float(previous["BM25签名：same-feature✓"]["seed_dr"]),
        ),
        compact(
            all_rows,
            "feature_promote_no_samefeature",
            seed_dr=float(previous["FeatLens：位置提升✓，same-feature✓"]["seed_dr"]),
        ),
    ]
    write_csv((_package_path() / 'results/ablation/deveval/bm25_without_feature_heuristic/missing_conditions.csv'), missing)

    updated = []
    for row in read_csv(COMBINED):
        updated.append(
            dict(
                method=row["method"],
                tasks=int(row["tasks"]),
                projects=int(row["projects"]),
                direct15_dr=row["direct15_dr"],
                seed_dr=row["seed_dr"],
                final15_dr=row["final_dr"],
                final_exact_dr="",
                mean_final_nodes=row["mean_returned_entities"],
            )
        )

    # Replace the controlled-rerun FeatLens row with the requested historical
    # output rescored under the new metric.
    for row in updated:
        if row["method"] == "FeatLens：位置提升✓，same-feature✓":
            row.update(
                seed_dr=historical_seed["new_seed_macro_dr"],
                final15_dr=historical["new_relaxed_task_macro_dr"],
                final_exact_dr=historical["new_exact_task_macro_dr"],
            )

    updated.extend(
        [
            dict(
                method="BM25代码：same-feature✗",
                tasks=missing[0]["tasks"],
                projects=missing[0]["projects"],
                direct15_dr=missing[0]["direct15_dr"],
                seed_dr=missing[0]["seed_dr"],
                final15_dr=missing[0]["final15_dr"],
                final_exact_dr=missing[0]["final_exact_dr"],
                mean_final_nodes=missing[0]["mean_final_nodes"],
            ),
            dict(
                method="BM25摘要：same-feature✗",
                tasks=missing[1]["tasks"],
                projects=missing[1]["projects"],
                direct15_dr=missing[1]["direct15_dr"],
                seed_dr=missing[1]["seed_dr"],
                final15_dr=missing[1]["final15_dr"],
                final_exact_dr=missing[1]["final_exact_dr"],
                mean_final_nodes=missing[1]["mean_final_nodes"],
            ),
            dict(
                method="BM25签名：same-feature✗",
                tasks=missing[2]["tasks"],
                projects=missing[2]["projects"],
                direct15_dr=missing[2]["direct15_dr"],
                seed_dr=missing[2]["seed_dr"],
                final15_dr=missing[2]["final15_dr"],
                final_exact_dr=missing[2]["final_exact_dr"],
                mean_final_nodes=missing[2]["mean_final_nodes"],
            ),
            dict(
                method="Feature：位置提升✓，same-feature✗",
                tasks=missing[3]["tasks"],
                projects=missing[3]["projects"],
                direct15_dr="",
                seed_dr=missing[3]["seed_dr"],
                final15_dr=missing[3]["final15_dr"],
                final_exact_dr=missing[3]["final_exact_dr"],
                mean_final_nodes=missing[3]["mean_final_nodes"],
            ),
        ]
    )
    write_csv((_package_path() / 'results/ablation/deveval/bm25_without_feature_heuristic/ablation_metrics.csv'), updated)

    verification = dict(
        projects=len(scope),
        tasks=1146,
        missing_conditions=missing,
        expected_rows=4 * 2 * 1146,
        observed_rows=len(all_rows),
        only_same_feature_flag_changed=True,
        source_seed_files_unchanged=True,
        featlens_graph_value=historical["new_relaxed_task_macro_dr"],
        featlens_graph_value_source=str((_package_path() / 'results/retrieval/combined/featlens_scores/summary.json')),
    )
    ((_package_path() / 'results/ablation/deveval/bm25_without_feature_heuristic/aggregate_verification.json')).write_text(
        json.dumps(verification, ensure_ascii=False, indent=2)
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
