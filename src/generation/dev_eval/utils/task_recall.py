import json
import os
from typing import Any, Dict, Optional

# ENRE element sets populated by load_enre_elements and used by compute_task_recall
variables_enre: set = set()
unresolved_attribute_enre: set = set()
module_enre: set = set()
package_enre: set = set()


def clear_enre_elements() -> None:
    """清空 ENRE 元素集合。批量跑多项目时，每切换项目前调用，再调用 load_enre_elements 加载当前项目。"""
    variables_enre.clear()
    unresolved_attribute_enre.clear()
    module_enre.clear()
    package_enre.clear()


# Read the ENRE result file, focusing on Variable, Unresolved Attribute, Module, and Package entries
def load_enre_elements(json_path: str) -> None:
    if not os.path.exists(json_path):
        print(f"Warning: ENRE JSON file not found at {json_path}")
        return
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading ENRE JSON: {e}")
        return
    variables = data.get("variables", [])
    for var in variables:
        if var.get("category") == "Variable":
            qname = var.get("qualifiedName")
            if qname:
                variables_enre.add(qname)
        elif var.get("category") == "Unresolved Attribute":
            if "File" in var and "/" in var.get("File", ""):
                qname = var.get("qualifiedName")
                if qname:
                    unresolved_attribute_enre.add(qname)
        elif var.get("category") == "Module":
            qname = var.get("qualifiedName")
            if qname:
                module_enre.add(qname)
        elif var.get("category") == "Package":
            qname = var.get("qualifiedName")
            if qname:
                package_enre.add(qname)


def _normalize_symbol(s: str) -> str:
    if "(" in s:
        return s.split("(", 1)[0]
    return s


def compute_task_recall(
    dependency: Optional[list[str]],
    searched_context_code_list: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    计算检索到的代码片段对任务 dependency 的召回率。
    context 项可包含 "sig" 或 "method_signature"（二者取一即可）、"method_code"。
    """
    dep = dependency or []
    dep_set = {x for x in dep}

    def _sig(ctx: dict) -> str:
        return ctx.get("sig", ctx.get("method_signature", ""))

    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    retrieved_set = {x.replace(".__init__", "") for x in retrieved_set}

    dep_total = len(dep_set)
    hit_set = dep_set & retrieved_set
    hit = len(hit_set) if dep_total > 0 else 0

    for x in dep:
        if x in variables_enre:
            var_name = x.split(".")[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code", "")
                if var_name in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in unresolved_attribute_enre:
            attr_name = x.split(".")[-1]
            class_name = ".".join(x.split(".")[:-1])
            for context_code in searched_context_code_list:
                sig = _sig(context_code)
                code_detail = context_code.get("method_code", "")
                if sig.startswith(f"{class_name}.") and f"self.{attr_name}" in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in module_enre:
            module_name = x
            for context_code in searched_context_code_list:
                sig = _sig(context_code)
                if sig.startswith(module_name):
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in package_enre:
            package_name = x
            for context_code in searched_context_code_list:
                sig = _sig(context_code)
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
