"""Run DevEval Pass@1 or DIR per project shard in parallel."""

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
import os
import subprocess
import sys
from pathlib import Path


ROOT = (_package_path())
GEN_ROOT = (_package_path() / 'results/generation/deveval/locagent')
PASS_SCRIPT = Path(str(_package_path('scripts/metrics/deveval/pass_k.py')))
RECALL_SCRIPT = (_package_path() / 'scripts/metrics/deveval/parser/recall_k.py')
DEP_ROOT = Path(str(_external_path('FEATLENS_DEPENDENCY_DATA_ROOT', 'external/deveval/dependency_data', '')))

SOURCE_ROOTS = {
    ('deepseek_v3_2', "pass"): _external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code'),
    ('deepseek_v3_2', "recall"): _external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code'),
    ('gpt_5_mini', "pass"): _external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code'),
    ('gpt_5_mini', "recall"): _external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code'),
}


def run_one(args: argparse.Namespace, data_file: Path) -> tuple[str, int]:
    project = data_file.stem
    dest = (_package_path() / 'results/generation/deveval/locagent/full_tests') / args.model / args.kind
    dest.mkdir(parents=True, exist_ok=True)
    completion = GEN_ROOT / f"{args.model}_shards" / "completion" / f"{project}.jsonl"
    log = dest / f"{project}.log"
    failure = dest / f"{project}_failure.log"
    source_root = SOURCE_ROOTS[(args.model, args.kind)]

    if args.kind == "pass":
        cmd = [
            str(_PortablePath(_evaluation_python())),
            str(PASS_SCRIPT),
            "--output_file", str(completion),
            "--log_file", str(dest / f"{project}_test_output.jsonl"),
            "--source_code_root", str(source_root),
            "--data_file", str(data_file),
            "--k", "1",
            "--n", "1",
            "--failure_log", str(failure),
        ]
        cwd = ROOT
    else:
        cmd = [
            str(_PortablePath(_evaluation_python())),
            str(RECALL_SCRIPT),
            "--output_file", str(completion),
            "--log_file", str(dest / f"{project}_recall_output.jsonl"),
            "--k", "1",
            "--source_code_root", str(source_root),
            "--dependency_data_root", str(DEP_ROOT),
            "--data_file", str(data_file),
            "--dependency_tmp_dir", str(dest / f"tmp_{project}"),
        ]
        cwd = (_package_path() / 'scripts/metrics/deveval/parser')

    env = os.environ.copy()
    eval_bin = _PortablePath(_evaluation_python()).parent
    env["PATH"] = str(eval_bin) + os.pathsep + env.get("PATH", "")
    env["PYTHONNOUSERSITE"] = "1"
    with log.open("w") as out:
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT
        )
    return project, proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['deepseek_v3_2', 'gpt_5_mini'], required=True)
    parser.add_argument("--kind", choices=["pass", "recall"], required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    data_dir = GEN_ROOT / f"{args.model}_shards" / "data"
    files = sorted(data_dir.glob("*.jsonl"))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args, p): p for p in files}
        for fut in concurrent.futures.as_completed(futures):
            project, rc = fut.result()
            print(f"[{args.model}/{args.kind}] {project}: rc={rc}", flush=True)
            if rc != 0:
                failures.append((project, rc))
    if failures:
        print(f"FAILURES: {failures}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
