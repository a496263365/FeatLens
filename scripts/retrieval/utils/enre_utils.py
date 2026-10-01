import json
import os
from typing import Any, Dict, Optional


# Global ENRE element sets shared by all search scripts that use this utility module
variables_enre: set[str] = set()
unresolved_attribute_enre: set[str] = set()
module_enre: set[str] = set()
package_enre: set[str] = set()


def clear_enre_elements() -> None:
    """清空 ENRE 元素集合。批量跑多项目时，每切换项目前调用。"""
    variables_enre.clear()
    unresolved_attribute_enre.clear()
    module_enre.clear()
    package_enre.clear()


def load_enre_elements(json_path: str) -> None:
    """读取 ENRE 解析结果文件，收集 Variable / Unresolved Attribute / Module / Package 类型元素。"""
    if not os.path.exists(json_path):
        print(f"Warning: ENRE JSON file not found at {json_path}")
        return

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"Error reading ENRE JSON: {e}")
        return

    variables = data.get("variables", [])
    for var in variables:
        category = var.get("category")
        qname = var.get("qualifiedName")
        if not qname:
            continue
        if category == "Variable":
            variables_enre.add(qname)
        elif category == "Unresolved Attribute":
            # The File field must exist and contain '/'; otherwise the symbol may be an attribute of an external library
            if "File" in var and "/" in str(var.get("File", "")):
                unresolved_attribute_enre.add(qname)
        elif category == "Module":
            module_enre.add(qname)
        elif category == "Package":
            package_enre.add(qname)


def _normalize_symbol(s: str) -> str:
    if "(" in s:
        return s.split("(", 1)[0]
    return s


def compute_task_recall(
    dependency: Optional[list[str]],
    searched_context_code_list: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """根据 ENRE 元素和检索到的上下文代码，计算单个任务的依赖召回情况。"""
    dep = dependency or []
    dep_set = {x for x in dep}

    # Matches here can only be functions or classes
    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    # Special case: convert retrieved elements such as xx.__init__.yy to xx.yy
    retrieved_set = {x.replace(".__init__", "") for x in retrieved_set}

    dep_total = len(dep_set)

    hit_set = dep_set & retrieved_set
    hit = len(hit_set) if dep_total > 0 else 0

    # Further relax matching based on ENRE elements
    for x in dep:
        if x in variables_enre:
            # For a variable, check whether it appears in any retrieved code
            var_name = x.split(".")[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code", "")
                if var_name in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in unresolved_attribute_enre:
            # self.xxx attribute inside a class
            attr_name = x.split(".")[-1]
            class_name = ".".join(x.split(".")[:-1])
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                code_detail = context_code.get("method_code", "")
                if sig.startswith(f"{class_name}.") and f"self.{attr_name}" in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in module_enre:
            module_name = x
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                if sig.startswith(module_name):
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in package_enre:
            package_name = x
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                if sig.startswith(package_name):
                    hit_set.add(x)
                    hit += 1
                    break

    recall = (hit / dep_total) if dep_total > 0 else None
    return {
        "dependency_total": dep_total,
        "dependency_hit": hit,
        "recall": recall,
    }


def is_method_hit(method_signature: str, method_code: str, deps_list: list[str]) -> bool:
    """判断某个检索到的方法是否命中依赖集合（带 ENRE 放宽规则）。"""
    norm_sig = _normalize_symbol(method_signature).replace(".__init__", "")
    for x in deps_list:
        if x == norm_sig:
            return True
        if x in variables_enre:
            var_name = x.split(".")[-1]
            if var_name and var_name in (method_code or ""):
                return True
        if x in unresolved_attribute_enre:
            attr_name = x.split(".")[-1]
            class_name = ".".join(x.split(".")[:-1])
            if norm_sig.startswith(f"{class_name}.") and f"self.{attr_name}" in (method_code or ""):
                return True
        if x in module_enre:
            if norm_sig.startswith(x):
                return True
        if x in package_enre:
            if norm_sig.startswith(x):
                return True
    return False


__all__ = [
    "variables_enre",
    "unresolved_attribute_enre",
    "module_enre",
    "package_enre",
    "clear_enre_elements",
    "load_enre_elements",
    "_normalize_symbol",
    "compute_task_recall",
    "is_method_hit",
]

