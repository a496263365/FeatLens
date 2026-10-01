import argparse
import ast
import os
import tokenize
from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple


@dataclass(frozen=True)
class ProjectStats:
    python_file_count: int
    total_line_count: int
    function_count: int


class _DefCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_count = 0
        self._function_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._function_depth == 0:
            self.function_count += 1
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._function_depth == 0:
            self.function_count += 1
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1


def _iter_python_files(project_root: str, exclude_dirnames: Set[str]) -> Iterable[str]:
    for root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirnames]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            if os.path.isfile(path):
                yield path


def _read_source_for_ast(file_path: str) -> Optional[str]:
    try:
        with tokenize.open(file_path) as f:
            return f.read()
    except Exception:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return None


def _count_physical_lines(file_path: str) -> int:
    try:
        with open(file_path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _count_functions_in_file(file_path: str) -> int:
    source = _read_source_for_ast(file_path)
    if source is None:
        return 0
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return 0
    counter = _DefCounter()
    counter.visit(tree)
    return counter.function_count


def analyze_project(project_root: str, exclude_dirnames: Optional[Set[str]] = None) -> ProjectStats:
    project_root = os.path.abspath(project_root)
    if os.path.isfile(project_root) and project_root.endswith(".py"):
        return ProjectStats(
            python_file_count=1,
            total_line_count=_count_physical_lines(project_root),
            function_count=_count_functions_in_file(project_root),
        )
    if not os.path.isdir(project_root):
        return ProjectStats(python_file_count=0, total_line_count=0, function_count=0)

    exclude_dirnames = exclude_dirnames or {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "site-packages",
        "node_modules",
        "dist",
        "build",
    }

    python_file_count = 0
    total_line_count = 0
    function_count = 0

    for file_path in _iter_python_files(project_root, exclude_dirnames):
        python_file_count += 1
        total_line_count += _count_physical_lines(file_path)
        function_count += _count_functions_in_file(file_path)

    return ProjectStats(
        python_file_count=python_file_count,
        total_line_count=total_line_count,
        function_count=function_count,
    )


def _try_read_excel(excel_path: str, sheet_name: Optional[str]):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("缺少依赖 pandas，无法读取 .xlsx") from e
    if sheet_name is None:
        df = pd.read_excel(excel_path, sheet_name=0)
    else:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if isinstance(df, dict):
        if not df:
            raise ValueError("Excel 没有可读取的 sheet")
        first_key = next(iter(df.keys()))
        df = df[first_key]
    return df


def _normalize_columns(df) -> Tuple[str, str]:
    cols = list(df.columns)
    normalized = {str(c).strip().lower(): c for c in cols}
    name_col = normalized.get("project_name")
    path_col = normalized.get("project_path") or normalized.get("project_root")
    if name_col is None or path_col is None:
        raise ValueError(
            f"Excel 需要包含列 project_name，以及 project_path 或 project_root（项目根目录路径），当前列为: {cols}"
        )
    return name_col, path_col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel",
        default="../../docs/deveval_project_path.xlsx",
    )
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--exclude", default=None)
    parser.add_argument("--file_bins", default=None)
    parser.add_argument("--func_bins", default=None)
    args = parser.parse_args()

    excel_path = os.path.abspath(args.excel)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(excel_path)

    exclude_dirnames: Optional[Set[str]] = None
    if args.exclude:
        exclude_dirnames = {x.strip() for x in args.exclude.split(",") if x.strip()}

    df = _try_read_excel(excel_path, sheet_name=args.sheet)
    name_col, path_col = _normalize_columns(df)

    python_file_counts = []
    total_line_counts = []
    function_counts = []

    for _, row in df.iterrows():
        project_path = row.get(path_col)
        if project_path is None:
            stats = ProjectStats(0, 0, 0)
        else:
            project_path_str = str(project_path).strip()
            if not project_path_str or project_path_str.lower() == "nan":
                stats = ProjectStats(0, 0, 0)
            else:
                stats = analyze_project(project_path_str, exclude_dirnames=exclude_dirnames)
        python_file_counts.append(stats.python_file_count)
        total_line_counts.append(stats.total_line_count)
        function_counts.append(stats.function_count)

    df["python_file_count"] = python_file_counts
    df["python_total_line_count"] = total_line_counts
    df["python_function_count"] = function_counts

    out_path = args.out
    if not out_path:
        base, ext = os.path.splitext(excel_path)
        out_path = f"{base}_stats{ext}"

    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("缺少依赖 pandas，无法写入 .xlsx") from e

    file_edges_default = [0, 5, 10, 20, 30, 40, 50, 75, 100, 200, 500, 1000, 2000]
    func_edges_default = [0, 20, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 2000, 5000]
    def _parse_bins(text, default_edges, max_val):
        if text is None or not str(text).strip():
            edges = list(default_edges)
        else:
            parts = [p.strip() for p in str(text).split(",") if p.strip()]
            edges = []
            for p in parts:
                try:
                    edges.append(int(p))
                except Exception:
                    pass
            if not edges:
                edges = list(default_edges)
        edges = sorted(set(edges))
        if edges[0] > 0:
            edges = [0] + edges
        if max_val > edges[-1]:
            edges.append(max_val)
        if len(edges) < 2:
            edges = [0, 1]
        if edges[0] == edges[-1]:
            edges = [edges[0], edges[0] + 1]
        return edges
    max_file = int(max(python_file_counts)) if python_file_counts else 0
    max_func = int(max(function_counts)) if function_counts else 0
    file_edges = _parse_bins(args.file_bins, file_edges_default, max_file)
    func_edges = _parse_bins(args.func_bins, func_edges_default, max_func)
    file_labels = [f"{file_edges[i]}-{file_edges[i+1]}" for i in range(len(file_edges)-1)]
    func_labels = [f"{func_edges[i]}-{func_edges[i+1]}" for i in range(len(func_edges)-1)]
    file_bins_series = pd.cut(df["python_file_count"], bins=file_edges, right=True, include_lowest=True, labels=file_labels)
    func_bins_series = pd.cut(df["python_function_count"], bins=func_edges, right=True, include_lowest=True, labels=func_labels)
    file_counts_by_interval = file_bins_series.value_counts(dropna=False).reindex(file_labels, fill_value=0)
    func_counts_by_interval = func_bins_series.value_counts(dropna=False).reindex(func_labels, fill_value=0)
    files_distribution = file_counts_by_interval.rename_axis("interval").reset_index(name="project_count")
    functions_distribution = func_counts_by_interval.rename_axis("interval").reset_index(name="project_count")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="stats")
        files_distribution.to_excel(writer, index=False, sheet_name="files_distribution")
        functions_distribution.to_excel(writer, index=False, sheet_name="functions_distribution")

    print(out_path)


if __name__ == "__main__":
    main()
