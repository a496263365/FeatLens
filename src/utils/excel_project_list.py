"""
Excel 项目列表批处理脚本的公共逻辑：列名规范化、ENRE 路径推导。

各 batch 脚本从实验计划 xlsx 读取「项目名称」「项目根目录」「ENRE路径」等列时使用。
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def normalize_excel_project_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将中英文列名统一为 project_name、project_root、enre_json（后者可选）。"""
    col_map: dict[Any, str] = {}
    for c in df.columns:
        c_str = str(c).strip()
        if c_str in ("项目名称", "project_name"):
            col_map[c] = "project_name"
        elif c_str in ("项目根目录", "project_root", "GRAPH_PROJECT_PATH"):
            col_map[c] = "project_root"
        elif c_str in ("ENRE路径", "enre_json"):
            col_map[c] = "enre_json"
    return df.rename(columns=col_map)


def default_enre_json_path(project_name: str, base_enre: str) -> str:
    """未在 Excel 中填写 enre_json 时：{base_enre}/{project_name}/report-enre.json"""
    return os.path.join(base_enre, project_name, "report-enre.json")


def resolve_enre_json_cell(row: pd.Series, project_name: str, base_enre: str) -> str:
    """从行中读取 enre_json；空则按 default_enre_json_path 推导。"""
    enre_json = row.get("enre_json")
    if enre_json is None or pd.isna(enre_json) or not str(enre_json).strip():
        return default_enre_json_path(project_name, base_enre)
    return str(enre_json).strip()
