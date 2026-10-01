#!/usr/bin/env python3
"""Blank function body line ranges from JSONL task records.

For each JSON object in a JSONL file, this script reads:
- completion_path
- body_position: [start_line, end_line] (1-based, inclusive)

Then it joins base_repo_path/completion_path, and replaces every line in the
body range with a blank line so total line count stays unchanged.
"""

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
from pathlib import Path
from typing import List, Tuple


# Global defaults. Edit these two values if you want to run without CLI args.
DEFAULT_JSONL_PATH = str(_package_path('data/evocodebench/tasks_all_123.jsonl'))
DEFAULT_BASE_REPO_PATH = str(_external_path('FEATLENS_EVOCODEBENCH_SOURCE_ROOT', 'external/evocodebench/source_code', ''))


def _blank_line_keep_newline(original: str) -> str:
    """Return an empty line while preserving original newline style."""
    if original.endswith("\r\n"):
        return "\r\n"
    if original.endswith("\n"):
        return "\n"
    return ""


def blank_body_range(file_path: Path, start_line: int, end_line: int) -> Tuple[bool, str]:
    """Blank [start_line, end_line] in file. Returns (changed, message)."""
    if start_line < 1 or end_line < start_line:
        return False, f"invalid body_position [{start_line}, {end_line}]"

    if not file_path.exists():
        return False, f"file not found: {file_path}"

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False, f"decode error (not utf-8): {file_path}"
    except Exception as exc:
        return False, f"read failed: {file_path} ({exc})"

    lines = content.splitlines(keepends=True)
    total = len(lines)
    if end_line > total:
        return False, f"body_position out of range [{start_line}, {end_line}], total lines={total}"

    changed = False
    for idx in range(start_line - 1, end_line):
        replacement = _blank_line_keep_newline(lines[idx])
        if lines[idx] != replacement:
            lines[idx] = replacement
            changed = True

    if not changed:
        return False, "already blank"

    try:
        file_path.write_text("".join(lines), encoding="utf-8")
    except Exception as exc:
        return False, f"write failed: {file_path} ({exc})"

    return True, "updated"


def process_jsonl(jsonl_path: Path, base_repo_path: Path) -> int:
    """Process all JSONL entries. Returns shell-style exit code."""
    if not jsonl_path.exists():
        print(f"ERROR: jsonl not found: {jsonl_path}")
        return 2

    updated = 0
    unchanged = 0
    errors: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue

            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid json ({exc})")
                continue

            completion_path = item.get("completion_path")
            body_position = item.get("body_position")

            if not isinstance(completion_path, str):
                errors.append(f"line {line_no}: missing/invalid completion_path")
                continue

            if (
                not isinstance(body_position, list)
                or len(body_position) != 2
                or not all(isinstance(x, int) for x in body_position)
            ):
                errors.append(f"line {line_no}: missing/invalid body_position")
                continue

            start_line, end_line = body_position
            target = base_repo_path / completion_path
            changed, msg = blank_body_range(target, start_line, end_line)

            if changed:
                updated += 1
                print(f"OK line {line_no}: {target} [{start_line}, {end_line}] -> {msg}")
            else:
                if msg == "already blank":
                    unchanged += 1
                    print(f"SKIP line {line_no}: {target} [{start_line}, {end_line}] -> {msg}")
                else:
                    errors.append(
                        f"line {line_no}: {target} [{start_line}, {end_line}] -> {msg}"
                    )

    print("\nSummary")
    print(f"- updated: {updated}")
    print(f"- unchanged: {unchanged}")
    print(f"- errors: {len(errors)}")
    if errors:
        print("\nErrors")
        for err in errors:
            print(f"- {err}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Blank body_position line ranges for each JSONL entry while preserving line counts."
        )
    )
    parser.add_argument(
        "--jsonl",
        default=DEFAULT_JSONL_PATH,
        help="Path to JSONL file containing completion_path and body_position.",
    )
    parser.add_argument(
        "--base-repo-path",
        default=DEFAULT_BASE_REPO_PATH,
        help="Base repo path to join with completion_path.",
    )
    args = parser.parse_args()

    if not args.jsonl:
        parser.error("jsonl path is required: set DEFAULT_JSONL_PATH or pass --jsonl")
    if not args.base_repo_path:
        parser.error(
            "base repo path is required: set DEFAULT_BASE_REPO_PATH or pass --base-repo-path"
        )

    return args


def main() -> int:
    args = parse_args()
    return process_jsonl(Path(args.jsonl), Path(args.base_repo_path))


if __name__ == "__main__":
    raise SystemExit(main())
