#!/usr/bin/env python3
"""Compute cyclomatic complexity for the retained generation result files."""

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


import csv
import json
from pathlib import Path

from cyclomatic_complexity_stats import summarize


ROOT = Path(str(_package_path()))
OUT = _package_path("work/statistics/code_complexity")

REGISTRY = [('no_context (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/no_context/deepseek_v3_2/completion.jsonl'), ('no_context (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/no_context/gpt_5_mini/completion.jsonl'), ('bm25 (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/bm25/deepseek_v3_2/completion.jsonl'), ('bm25 (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/bm25/gpt_5_mini/completion.jsonl'), ('unixcoder (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/unixcoder/deepseek_v3_2/completion.jsonl'), ('unixcoder (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/unixcoder/gpt_5_mini/completion.jsonl'), ('featlens (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/featlens/deepseek_v3_2/completion.jsonl'), ('featlens (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/featlens/gpt_5_mini/completion.jsonl'), ('repograph (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/repograph/deepseek_v3_2_combined_completion.jsonl'), ('reposcope (deepseek_v3_2, 359 tasks, B20)', 'results/generation/deveval/reposcope/deepseek_v3_2/completions.jsonl'), ('locagent (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/locagent/deepseek_v3_2_1430_completion.jsonl'), ('oracle (deepseek_v3_2, 1430 tasks)', 'results/generation/deveval/oracle/deepseek_v3_2_completion.jsonl'), ('repograph (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/repograph/gpt_5_mini_combined_completion.jsonl'), ('reposcope (gpt_5_mini, 359 tasks, B20)', 'results/generation/deveval/reposcope/gpt_5_mini/completions.jsonl'), ('locagent (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/locagent/gpt_5_mini_1430_completion.jsonl'), ('oracle (gpt_5_mini, 1430 tasks)', 'results/generation/deveval/oracle/gpt_5_mini_completion.jsonl'), ('codexgraph (deepseek_v3_2, full completion archive)', 'results/generation/deveval/codexgraph/deepseek_v3_2/completions/deepseek_v3_2_q2_final.jsonl'), ('codexgraph (gpt_5_mini, full completion archive)', 'results/generation/deveval/codexgraph/gpt_5_mini/completions/gpt_5_mini_evaluated.jsonl')]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, relative in REGISTRY:
        path = ROOT / relative
        if not path.is_file():
            rows.append(
                {
                    "method": label,
                    "path": relative,
                    "tasks": 0,
                    "parsed": 0,
                    "parse_failures": 0,
                    "parse_failure_rate": "",
                    "mean_cc": "",
                    "median_cc": "",
                    "p90_cc": "",
                    "max_cc": "",
                    "mean_loc": "",
                    "missing_file": True,
                }
            )
            continue
        summary = summarize(path)
        rows.append(
            {
                "method": label,
                "path": relative,
                "tasks": summary["tasks"],
                "parsed": summary["parsed"],
                "parse_failures": summary["parse_failures"],
                "parse_failure_rate": summary["parse_failure_rate"],
                "mean_cc": summary["mean_cc"],
                "median_cc": summary["median_cc"],
                "p90_cc": summary["p90_cc"],
                "max_cc": summary["max_cc"],
                "mean_loc": summary["mean_loc"],
                "missing_file": False,
            }
        )

    csv_path = OUT / "cc_inventory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = OUT / "cc_inventory.json"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Generated-code cyclomatic complexity inventory",
        "",
        "CC uses AST-McCabe complexity per completion: 1 plus decision points "
        "for if/for/while/except/with/assert/conditional-expression/boolean "
        "operators/comprehension ifs/match cases. Parse failures are excluded "
        "from the mean and reported separately.",
        "",
        "| Method | Tasks | Parsed | Mean CC | Median | P90 | Max | Mean LOC | Parse fail |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["missing_file"]:
            lines.append(f"| {row['method']} | MISSING | | | | | | | |")
            continue
        lines.append(
            "| {method} | {tasks} | {parsed} | {mean_cc:.3f} | "
            "{median_cc:.1f} | {p90_cc:.1f} | {max_cc} | "
            "{mean_loc:.3f} | {parse_failures} |".format(**row)
        )
    report = OUT / "CC_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
