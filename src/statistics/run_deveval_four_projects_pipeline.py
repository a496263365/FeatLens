
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
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


EXCEL_PATH = str(_external_path('FEATLENS_PROJECT_LIST_ROOT', 'external/project_lists', 'deveval_four_projects.xlsx'))
ROOT_DIR = Path(str(_package_path('')))
OUTPUT_ROOT = ROOT_DIR / "src" / "statistics" / "generated_pipeline"
TEST_REPO_ROOT = Path(str(_package_path('results/retrieval/deveval/featlens_and_rag')))
DEV_EVAL_SOURCE_ROOT = Path(str(_external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code', '')))


@dataclass
class ScriptPlan:
    source_path: Path
    generated_path: Path
    run_cwd: Path


def replace_assignment(content: str, var_name: str, value: str) -> Tuple[str, bool]:
    pattern = rf"^{re.escape(var_name)}\s*=.*$"
    replacement = f"{var_name} = {json.dumps(value, ensure_ascii=False)}"
    new_content, n = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    return new_content, n > 0


def replace_first_existing(content: str, var_names: List[str], value: str) -> str:
    for name in var_names:
        new_content, ok = replace_assignment(content, name, value)
        if ok:
            return new_content
    raise ValueError(f"变量不存在: {var_names}")


def repo_slug_from_row(project_name: str, project_path: str) -> str:
    path = str(project_path).strip().rstrip("/\\")
    if path:
        leaf = os.path.basename(path)
        if leaf:
            return leaf
    return str(project_name).strip()


def build_replacements(project_name: str, project_path: str, project_dir: str) -> Dict[str, Dict[str, str]]:
    repo_slug = repo_slug_from_row(project_name, project_path)
    category = str(project_dir).split("/", 1)[0].strip()
    test_repo_dir = TEST_REPO_ROOT / repo_slug
    project_root_for_expand = DEV_EVAL_SOURCE_ROOT / category

    return {
        "bm25_search.py": {
            "PROJECT_DIR_OR_PATH": project_dir,
            "METHODS_CSV": str(test_repo_dir / "methods.csv"),
            "METHODS_DESC_CSV": str(test_repo_dir / "methods_with_desc.csv"),
            "FILTERED_PATH": str(test_repo_dir / "filtered.jsonl"),
            "refined_queries_cache_path": str(test_repo_dir / "refined_queries.json"),
            "ENRE_JSON": str(test_repo_dir / f"{repo_slug}-report-enre.json"),
        },
        "unixcoder_based_search.py": {
            "METHODS_CSV": str(test_repo_dir / "methods.csv"),
            "FILTERED_FILE": str(test_repo_dir / "filtered.jsonl"),
            "refined_queries_cache_path": str(test_repo_dir / "refined_queries.json"),
            "ENRE_JSON": str(test_repo_dir / f"{repo_slug}-report-enre.json"),
        },
        "feature_based_search.py": {
            "PROJECT_DIR_OR_PATH": project_dir,
            "FEATURE_CSV": str(test_repo_dir / "features.csv"),
            "METHODS_CSV": str(test_repo_dir / "methods.csv"),
            "FILTERED_PATH": str(test_repo_dir / "filtered.jsonl"),
            "refined_queries_cache_path": str(test_repo_dir / "refined_queries.json"),
            "ENRE_JSON": str(test_repo_dir / f"{repo_slug}-report-enre.json"),
        },
        "build_graph.py": {
            "METHODS_CSV": str(test_repo_dir / "methods.csv"),
            "ENRE_JSON": str(test_repo_dir / f"{repo_slug}-report-enre.json"),
            "FILTERED_PATH": str(test_repo_dir / "filtered.jsonl"),
            "DIAGNOSTIC_JSONL": str(test_repo_dir / "diagnostic_***feature.jsonl"),
            "OUTPUT_GRAPH_PATH": str(test_repo_dir / "graph_results_***all"),
        },
        "expand_and_rank_graph_exclude_TM_last.py": {
            "METHODS_CSV": str(test_repo_dir / "methods.csv"),
            "ENRE_JSON": str(test_repo_dir / f"{repo_slug}-report-enre.json"),
            "FILTERED_PATH": str(test_repo_dir / "filtered.jsonl"),
            "OUTPUT_GRAPH_PATH": str(test_repo_dir / "graph_results_***all"),
            "PROJECT_PATH": str(project_root_for_expand),
        },
        "compare_graph_recall.py": {
            "GRAPH_RESULTS_DIR": str(test_repo_dir / "graph_results_***all"),
            "FILTERED_JSONL_PATH": str(test_repo_dir / "filtered.jsonl"),
            "OUTPUT_REPORT_FILE": str(test_repo_dir / "303_expand_graph_match_comparison_report.csv"),
            "ENRE_JSON": str(test_repo_dir / f"{repo_slug}-report-enre.json"),
        },
    }


def generate_script(source_path: Path, generated_path: Path, kv: Dict[str, str]) -> None:
    content = source_path.read_text(encoding="utf-8")
    for key, value in kv.items():
        if key == "PROJECT_DIR_OR_PATH":
            content = replace_first_existing(content, ["PROJECT_DIR", "PROJECT_PATH"], value)
        else:
            content, ok = replace_assignment(content, key, value)
            if not ok:
                raise ValueError(f"{source_path} 中缺少变量 {key}")
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    generated_path.write_text(content, encoding="utf-8")


def run_python_file(script_path: Path, cwd: Path, log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script_path)]
    start = time.time()
    env = os.environ.copy()
    py_paths = [str(cwd), str(ROOT_DIR / "src")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    with log_file.open("w", encoding="utf-8") as fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fp.write(line)
        proc.wait()
    sec = time.time() - start
    print(f"[完成] {script_path.name} exit_code={proc.returncode} 耗时={sec:.1f}s")
    return int(proc.returncode)


def build_plans(project_key: str) -> List[ScriptPlan]:
    output_dir = OUTPUT_ROOT / project_key / "scripts"
    search_dir = ROOT_DIR / "src" / "search"
    graph_dir = ROOT_DIR / "src" / "graph"
    return [
        ScriptPlan(search_dir / "bm25_search.py", output_dir / "bm25_search.py", search_dir),
        ScriptPlan(search_dir / "unixcoder_based_search.py", output_dir / "unixcoder_based_search.py", search_dir),
        ScriptPlan(search_dir / "feature_based_search.py", output_dir / "feature_based_search.py", search_dir),
        ScriptPlan(graph_dir / "build_graph.py", output_dir / "build_graph.py", graph_dir),
        ScriptPlan(
            graph_dir / "expand_and_rank_graph_exclude_TM_last.py",
            output_dir / "expand_and_rank_graph_exclude_TM_last.py",
            graph_dir,
        ),
        ScriptPlan(graph_dir / "compare_graph_recall.py", output_dir / "compare_graph_recall.py", graph_dir),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", default=EXCEL_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-projects", type=int, default=0)
    parser.add_argument("--only-project", default="")
    args = parser.parse_args()

    df = pd.read_excel(args.excel, dtype=str).fillna("")
    required_cols = {"project_name", "project_path", "project_dir"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}; 实际列: {list(df.columns)}")

    records = df.to_dict(orient="records")
    if args.only_project:
        only_key = args.only_project.strip().lower()
        records = [r for r in records if str(r.get("project_name", "")).strip().lower() == only_key]
    if args.max_projects and args.max_projects > 0:
        records = records[: args.max_projects]
    if not records:
        print("没有可执行项目")
        return 0

    total = len(records)
    any_failed = False
    total_steps = 6
    for idx, row in enumerate(records, start=1):
        project_name = str(row.get("project_name", "")).strip()
        project_path = str(row.get("project_path", "")).strip()
        project_dir = str(row.get("project_dir", "")).strip()
        repo_slug = repo_slug_from_row(project_name, project_path)
        project_key = repo_slug.lower()
        print("")
        print(f"[项目 {idx}/{total}] name={project_name} repo={repo_slug} dir={project_dir}")

        replacements = build_replacements(project_name, project_path, project_dir)
        plans = build_plans(project_key)

        for plan in plans:
            name = plan.source_path.name
            print(f"[生成] {name} -> {plan.generated_path}")
            print("plan.source_path ",plan.source_path)
            generate_script(plan.source_path, plan.generated_path, replacements[name])

        if args.dry_run:
            print("[跳过执行] dry-run 模式")
            continue

        logs_dir = OUTPUT_ROOT / project_key / "logs"
        for step_i, plan in enumerate(plans, start=1):
            print("")
            print(f"[执行 {step_i}/{total_steps}] {plan.generated_path.name}")
            log_file = logs_dir / f"{step_i}_{plan.generated_path.stem}.log"
            code = run_python_file(plan.generated_path, plan.run_cwd, log_file)
            if code != 0:
                any_failed = True
                print(f"[失败] {plan.generated_path.name}，日志: {log_file}")
                break
        else:
            print(f"[完成] 项目 {project_name} 全部步骤执行成功")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
