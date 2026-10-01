"""
从 Excel「项目根目录」路径中，截取相对 SOURCE_CODE_DIR 的路径片段，供 data.jsonl 的 project_path 或磁盘拼接使用。

支持两种数据集布局：
- two_segments: Source_Code 后取两段，如 System/mrjob
- one_segment: Source_Code 后只取一段，如 hbmqtt（路径形如 .../Source_Code/hbmqtt/hbmqtt）
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional

ProjectRelMode = Literal["two_segments", "one_segment"]

# Default when a batch script does not pass mode; batch files can override it with a top-level constant
DEFAULT_PROJECT_REL_MODE: ProjectRelMode = "two_segments"


def _find_source_basename_index(parts: List[str], source_code_dir: str) -> Optional[int]:
    norm_source = os.path.normpath(source_code_dir)
    base = os.path.basename(norm_source)
    try:
        return parts.index(base)
    except ValueError:
        return None


def _project_rel_two_segments(project_root: str, source_code_dir: str) -> str:
    norm_root = os.path.normpath(project_root)
    parts = norm_root.split(os.sep)
    idx = _find_source_basename_index(parts, source_code_dir)
    if idx is None:
        if len(parts) >= 3:
            return "/".join(parts[-3:-1])
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]

    start = idx + 1
    end = start + 2
    sub_parts = parts[start:end]
    return "/".join(sub_parts)


def _project_rel_one_segment(project_root: str, source_code_dir: str) -> str:
    norm_root = os.path.normpath(project_root)
    parts = norm_root.split(os.sep)
    idx = _find_source_basename_index(parts, source_code_dir)
    if idx is None:
        return parts[-1] if parts else ""

    start = idx + 1
    if start < len(parts):
        return parts[start]
    return parts[-1] if parts else ""


def project_rel_from_project_root(
    project_root: str,
    source_code_dir: str,
    *,
    mode: ProjectRelMode = DEFAULT_PROJECT_REL_MODE,
) -> str:
    if mode == "one_segment":
        return _project_rel_one_segment(project_root, source_code_dir)
    return _project_rel_two_segments(project_root, source_code_dir)
