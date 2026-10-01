"""
统计 JSONL 实验结果中每条任务的 completion（Python 片段）的「代码长度」平均值。

长度定义（与注释、标识符规则一致）：
- 先用 tokenize 按 Python 词法切分，注释（#、行尾注释等）不计入；
- 每个非空白结构单元计 1：关键字、运算符、单个标识符（含 apple_tree_num 整体为 1）、
  数字/字符串字面量各计 1；
- 缩进换行（INDENT/DEDENT/NEWLINE/NL）不计入。
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
import io
import json
import sys
import textwrap
import tokenize
from typing import Iterable


DEFAULT_JSONL_PATH = str(_package_path('results/generation/deveval/repograph/deepseek_v3_2_combined_completion.jsonl'))

_SKIP_TYPES = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
)


def python_completion_token_len(code: str) -> int:
    """对一段 Python 代码统计语义 token 个数（注释与纯空白结构不计）。"""
    if not code or not str(code).strip():
        return 0
    # Completions are usually indented blocks; dedenting them better matches the top-level structure expected by tokenize
    code = textwrap.dedent(code).strip("\n") + "\n"
    n = 0
    readline = io.StringIO(code).readline
    try:
        for tok in tokenize.generate_tokens(readline):
            if tok.type in _SKIP_TYPES:
                continue
            n += 1
        return n
    except (tokenize.TokenError, IndentationError, TabError):
        return n


def iter_completion_lengths(path: str, field: str = "completion") -> Iterable[tuple[int, str]]:
    """逐行读取 jsonl，产出 (token_len, namespace 或空)。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no}: JSON decode error: {e}", file=sys.stderr)
                continue
            raw = obj.get(field)
            completion = raw if isinstance(raw, str) else ""
            ns = obj.get("namespace", "")
            ns_str = ns if isinstance(ns, str) else ""
            yield python_completion_token_len(completion), ns_str


def main() -> None:
    p = argparse.ArgumentParser(
        description="统计 JSONL 中每条任务 completion 的 Python 语义 token 长度平均值"
    )
    p.add_argument("--jsonl_path", default=DEFAULT_JSONL_PATH, help="输入 JSONL 路径")
    p.add_argument(
        "--field",
        default="completion",
        help="存放代码片段的字段名（默认 completion）",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="逐行打印每条任务的 token 长度与 namespace",
    )
    args = p.parse_args()

    lengths: list[int] = []
    for n, ns in iter_completion_lengths(args.jsonl_path, field=args.field):
        lengths.append(n)
        if args.list:
            print(f"{n}\t{ns}")

    if not lengths:
        print("No valid lines with completion field; cannot compute average.", file=sys.stderr)
        sys.exit(1)

    total = sum(lengths)
    avg = total / len(lengths)
    print(f"tasks: {len(lengths)}")
    print(f"total_token_len: {total}")
    print(f"avg_token_len: {avg:.6f}")


if __name__ == "__main__":
    main()
