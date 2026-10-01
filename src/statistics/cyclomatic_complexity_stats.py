#!/usr/bin/env python3
"""Compute AST-based McCabe cyclomatic complexity for generated Python snippets.

The unit is one generated completion.  A completion may be a function body, a
complete function, or a small block; it is normalized with ``textwrap.dedent``
and, if necessary, wrapped in a temporary function before parsing.

Complexity is 1 plus the number of decision points:

- if / for / async for / while
- except
- with / async with
- assert
- conditional expression
- boolean ``and`` / ``or`` operands beyond the first
- each comprehension ``for`` and each comprehension ``if``
- each match case

Parse failures are reported separately and are not silently counted as 0.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import statistics
import sys
import textwrap
import tokenize
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_completion(code: str) -> Optional[ast.AST]:
    if not code or not code.strip():
        return None
    dedented = textwrap.dedent(code).strip("\n")
    candidates = [dedented]
    candidates.append(
        "def __featlens_completion__():\n"
        + textwrap.indent(dedented, "    ")
    )
    last_error: Optional[SyntaxError] = None
    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except SyntaxError as exc:
            last_error = exc
    if last_error is not None:
        return None
    return None


class McCabeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.complexity = 1

    def _add(self, amount: int = 1) -> None:
        self.complexity += amount

    def visit_If(self, node: ast.If) -> None:
        self._add()
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._add()
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._add()
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._add()
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._add()
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._add()
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._add()
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._add()
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._add()
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._add(max(0, len(node.values) - 1))
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._add(1 + len(node.ifs))
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._add(len(node.cases))
        self.generic_visit(node)


def complexity(code: str) -> Optional[int]:
    tree = parse_completion(code)
    if tree is None:
        return None
    visitor = McCabeVisitor()
    visitor.visit(tree)
    return visitor.complexity


_SKIP_TOKEN_TYPES = frozenset(
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


def semantic_token_length(code: str) -> int:
    if not code or not str(code).strip():
        return 0
    normalized = textwrap.dedent(code).strip("\n") + "\n"
    count = 0
    try:
        for token in tokenize.generate_tokens(
            io.StringIO(normalized).readline
        ):
            if token.type not in _SKIP_TOKEN_TYPES:
                count += 1
    except (tokenize.TokenError, IndentationError, TabError):
        pass
    return count


def iter_completion_records(
    path: Path, field: str
) -> Iterable[Tuple[int, str, int, Optional[int], str]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            value = record.get(field)
            code = value if isinstance(value, str) else ""
            yield (
                line_no,
                str(record.get("namespace", "")),
                semantic_token_length(code),
                complexity(code),
                code,
            )


def percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(
    path: Path, field: str = "completion", per_task: bool = False
) -> Dict[str, Any]:
    cc_values: List[int] = []
    loc_values: List[int] = []
    failed = 0
    empty = 0
    task_rows: List[Dict[str, Any]] = []
    for line_no, namespace, loc, cc, _code in iter_completion_records(
        path, field
    ):
        loc_values.append(loc)
        if cc is None:
            failed += 1
            if loc == 0:
                empty += 1
        else:
            cc_values.append(cc)
        if per_task:
            task_rows.append(
                {
                    "line": line_no,
                    "namespace": namespace,
                    "loc": loc,
                    "cyclomatic_complexity": cc,
                }
            )
    total = len(loc_values)
    return {
        "jsonl_path": str(path),
        "field": field,
        "tasks": total,
        "parsed": len(cc_values),
        "parse_failures": failed,
        "empty_completions": empty,
        "parse_failure_rate": (failed / total) if total else None,
        "mean_cc": (sum(cc_values) / len(cc_values)) if cc_values else None,
        "median_cc": statistics.median(cc_values) if cc_values else None,
        "p90_cc": percentile([float(x) for x in cc_values], 0.90),
        "max_cc": max(cc_values) if cc_values else None,
        "mean_loc": (sum(loc_values) / len(loc_values)) if loc_values else None,
        "tasks_detail": task_rows if per_task else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute cyclomatic complexity for generated Python completions."
    )
    parser.add_argument("--jsonl_path", type=Path, required=True)
    parser.add_argument("--field", default="completion")
    parser.add_argument("--list", action="store_true", dest="per_task")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = summarize(args.jsonl_path, args.field, args.per_task)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
