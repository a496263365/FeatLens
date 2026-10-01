"""
从单个 Python 源文件中，找出「指定的一些函数/类」所使用到的「该文件内的模块级变量和类」（不含函数）。

用途：在 graph RAG 里，把用到的全局变量和全局类列出来，让 LLM 更关注这些。
方法：用 tree-sitter 做静态分析，不执行代码。

=== 整体思路 ===
1. 先找出这个文件里「模块级变量」和「模块级类」的名字（顶层赋值变量、类名；不含顶层函数名）→ module_vars_and_classes
2. 对每个我们关心的函数/类，找到对应的 AST 节点
3. 在该节点内部：收集「被使用」的标识符，排除「在该节点内定义的」局部名
4. 剩下的名字里，若在 module_vars_and_classes 里，才加入结果（即只保留用到的全局变量和类，不保留用到的函数）

=== 各函数调用关系（从入口 get_used_globals_in_file 看）===
  get_used_globals_in_file
    → _get_py_parser() 解析文件
    → _module_level_variable_and_class_names() 得到「仅变量+类」的模块级名字集合
    → _find_definitions_by_name() 得到 (类型,短名) -> [AST节点]
    → 对每个 node_key:
        _resolve_ast_node() 得到唯一节点
        _names_defined_in_scope() 得到该节点内「局部定义」
        _identifiers_used_in_scope(..., exclude_defs=局部定义) 得到「使用到的名字」
        → 与 module_vars_and_classes 取交集加入结果（只统计变量和类）
  辅助: _slice, _effective_def_node, _params_from_* 等
"""
import os
import re
from typing import List, Optional, Set, Tuple

# (category, short name, optional full signature)
# For example, ("Function", "write", "boto.roboto.param.Converter.convert_dir(cls, param, value)")
NodeKey = Tuple[str, str, Optional[str]]


def _get_py_parser():
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    return parser, PY_LANGUAGE


def _slice(content: bytes, start: int, end: int) -> str:
    return content[start:end].decode("utf-8", errors="replace")


def _module_level_variable_and_class_names(content: bytes, root) -> Set[str]:
    """
    收集「模块顶层」的变量名和类名。
    只遍历 root 的直接子节点。
    包括：
    - 顶层类名（class_definition 的 name）
    - 顶层赋值左侧的简单变量名（assignment / expression_statement 里的 assignment，且 left 是 identifier）
    不包括：function_definition、decorated_definition（即所有顶层函数都排除）。
    用于最终只把「用到的全局变量和类」加入结果，不把用到的函数算进去。
    """
    names = set()
    for child in root.children:
        if child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            if name_node:
                names.add(_slice(content, name_node.start_byte, name_node.end_byte))
        elif child.type == "assignment":
            left = child.child_by_field_name("left")
            if left and left.type == "identifier":
                names.add(_slice(content, left.start_byte, left.end_byte))
        elif child.type == "expression_statement":
            inner = child.child(0)
            if inner and inner.type == "assignment":
                left = inner.child_by_field_name("left")
                if left and left.type == "identifier":
                    names.add(_slice(content, left.start_byte, left.end_byte))
    return names


def _params_from_method_signature(method_signature: str) -> Optional[List[str]]:
    """
    从「方法签名字符串」里解析出参数名列表，用于和 AST 里的参数列表比对，区分重载。
    输入例如："pkg.Cls.method(self, a, b)" 或 "Writer.write(self, data)"。
    用正则取出括号里的部分，再按逗号分割；对每一段去掉默认值（=...）和类型注解（:...），
    去掉 * / ** 前缀，得到纯参数名。返回 ["self", "a", "b"]；无括号则返回 None。
    """
    if not method_signature or "(" not in method_signature:
        return None
    match = re.search(r"\(([^)]*)\)", method_signature)
    if not match:
        return None
    param_str = match.group(1).strip()
    if not param_str:
        return []
    # "self, a, b" or "self, a=1, b" -> split by comma, take first token (name or *name)
    names = []
    for part in param_str.split(","):
        part = part.strip()
        # take the first identifier (before = or :)
        if "=" in part:
            part = part.split("=")[0].strip()
        if ":" in part:
            part = part.split(":")[0].strip()
        if part.startswith("*"):
            part = part.lstrip("*")
        if part:
            names.append(part)
    return names


def _params_from_ast(node, content: bytes) -> List[str]:
    """
    从 AST 的「函数定义节点」里读出参数名列表（顺序保持）。
    先通过 _effective_def_node 得到真正的 function_definition（处理装饰器），
    再遍历 parameters 子节点的 children：取 identifier 或 typed_parameter 的 name，
    过滤掉 "(", ")", "*", "/", "," 等非名字的 token。用于和 _params_from_method_signature 的结果比对。
    """
    node = _effective_def_node(node)
    params_node = node.child_by_field_name("parameters")
    if not params_node:
        return []
    names = []
    for c in params_node.children:
        if c.type == "identifier":
            name = _slice(content, c.start_byte, c.end_byte)
            if name and name not in ("(", ")", "*", "/", ","):
                names.append(name)
        elif c.type == "typed_parameter":
            id_node = c.child_by_field_name("name")
            if id_node:
                names.append(_slice(content, id_node.start_byte, id_node.end_byte))
    return names


def _find_definitions_by_name(content: bytes, root) -> dict:
    """
    遍历整棵 AST，找出所有「类定义」和「函数定义」节点，按 (类型, 短名) 归类。
    类型为 "Class" 或 "Function"，短名为类名/函数名（如 "Queue", "get_messages"）。
    同一文件里可能有多个同名函数（重载），所以 value 是「节点列表」而不是单个节点。
    返回: dict[(type, short_name)] = [node1, node2, ...]
    """
    result = {}  # (type, short_name) -> list of nodes
    def add(key, node):
        if key not in result:
            result[key] = []
        result[key].append(node)

    def walk(node, in_class_name: str = None):
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                cname = _slice(content, name_node.start_byte, name_node.end_byte)
                add(("Class", cname), node)
                for c in node.children:
                    walk(c, cname)
            return
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                fname = _slice(content, name_node.start_byte, name_node.end_byte)
                add(("Function", fname), node)
            for c in node.children:
                walk(c, in_class_name)
            return
        if node.type == "decorated_definition":
            def_node = node.child_by_field_name("definition")
            if def_node and def_node.type == "function_definition":
                name_node = def_node.child_by_field_name("name")
                if name_node:
                    fname = _slice(content, name_node.start_byte, name_node.end_byte)
                    add(("Function", fname), node)
            for c in node.children:
                walk(c, in_class_name)
            return
        for c in node.children:
            walk(c, in_class_name)
    walk(root)
    return result


def _effective_def_node(node) -> object:
    """
    若节点是「带装饰器的定义」(decorated_definition)，返回其内部的 function_definition；
    否则返回节点本身。这样后续取 body、parameters 时一定在真正的函数节点上取到。
    """
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner:
            return inner
    return node


def _names_defined_in_scope(node, content: bytes) -> Set[str]:
    """
    收集「在这个函数/类节点内部被定义」的名字（局部），用于排除「使用」时的误判。
    包括：形参名（parameters 里的 identifier，去掉 self 和 *）、
    赋值语句左侧的变量、for/with 的循环变量等。递归遍历 body 里的 assignment / for_statement / with_statement。
    这些名字在 _identifiers_used_in_scope 里会被排除，不当作「可能引用全局」的标识符。
    """
    defined = set()
    node = _effective_def_node(node)
    body = node.child_by_field_name("body")
    if not body:
        return defined
    params = node.child_by_field_name("parameters")
    if params:
        for c in params.children:
            if c.type == "identifier" and c.start_byte != c.end_byte:
                name = _slice(content, c.start_byte, c.end_byte)
                if name != "self" and name != "*" and not name.startswith("*"):
                    defined.add(name)
    def collect_assignments(n):
        if n.type == "assignment":
            left = n.child_by_field_name("left")
            if left and left.type == "identifier":
                defined.add(_slice(content, left.start_byte, left.end_byte))
            for c in n.children:
                collect_assignments(c)
        elif n.type in ("for_statement", "with_statement"):
            for c in n.children:
                if c.type == "identifier" and c.start_byte != c.end_byte:
                    defined.add(_slice(content, c.start_byte, c.end_byte))
                collect_assignments(c)
        else:
            for c in n.children:
                collect_assignments(c)
    collect_assignments(body)
    return defined


def _identifiers_used_in_scope(node, content: bytes, exclude_defs: Set[str]) -> Set[str]:
    """
    收集该节点 body 里出现的所有「标识符」名字（即被「使用」的名字），
    但排除 exclude_defs（即 _names_defined_in_scope 得到的局部定义）。
    递归遍历 body 下每个 AST 节点，遇到 identifier 就加入 used。
    注意：这里不区分是「读」还是「写」，只是所有出现过的名字；和 exclude_defs 取差后，
    剩下的是「可能是外部/全局引用」的名字，再和 module_defs 取交集就得到「用到的全局」。
    """
    used = set()
    node = _effective_def_node(node)
    def walk(n):
        if n.type == "identifier":
            name = _slice(content, n.start_byte, n.end_byte)
            if name and name not in exclude_defs:
                used.add(name)
        for c in n.children:
            walk(c)
    body = node.child_by_field_name("body")
    if body:
        walk(body)
    return used


def _resolve_ast_node(
    def_map: dict,
    content: bytes,
    category: str,
    short_name: str,
    method_signature: Optional[str],
) -> Optional[object]:
    """
    根据 (category, short_name, method_signature) 从 def_map 里找到「唯一」的 AST 节点。
    - 若 (category, short_name) 只对应一个节点，直接返回。
    - 若对应多个节点（同名函数重载），则用 method_signature 里的参数列表和每个节点的参数列表比对，
      完全一致则返回该节点；都比对不上则退回第一个。Class 没有参数列表，按 short_name 唯一即可。
    """
    key = (category, short_name)
    candidates = def_map.get(key, [])
    if not candidates:
        print(f"No candidates found for {key}")
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple overloads: match by parameter list
    if not method_signature or "(" not in method_signature:
        return candidates[0]
    want_params = _params_from_method_signature(method_signature)
    if want_params is None:
        return candidates[0]
    for node in candidates:
        ast_params = _params_from_ast(node, content)
        if ast_params == want_params:
            return node
    return candidates[0]


def get_used_globals_in_file(
    project_path: str,
    file_path: str,
    node_keys: List[Tuple[str, str, Optional[str]]],
) -> Set[str]:
    """
    对指定 Python 文件，找出 node_keys 里列出的那些函数/类所「使用到的」本文件模块级变量和类。

    参数:
    - project_path: 项目根目录，与 file_path 拼成绝对路径读文件。
    - file_path: 相对 project_path 的 .py 文件路径。
    - node_keys: 列表，每项为 (category, short_name, method_signature?)。
      category 为 "Function" 或 "Class"；short_name 为函数/类名；
      method_signature 为完整签名，用于区分重载（如 "pkg.Cls.method(self, a)"），Class 可省略。

    流程:
    1. 读文件，用 tree-sitter 解析得到 AST。
    2. _module_level_variable_and_class_names：得到本文件「仅变量+类」的模块级名字集合（不含函数）。
    3. _find_definitions_by_name：得到 (type, short_name) -> [节点列表] 的 def_map。
    4. 对每个 node_key：_resolve_ast_node 得到唯一 AST 节点；
       _names_defined_in_scope 得到该节点内「局部定义」；
       _identifiers_used_in_scope 得到该节点内「使用到的标识符」并去掉局部；
       这些标识符里属于「变量+类」集合的才加入结果（用到的函数名不加入）。
    5. 返回「被用到的模块级变量和类」的集合。
    """
    full_path = os.path.join(project_path, file_path.lstrip("/"))
    if not os.path.exists(full_path):
        if os.path.exists(file_path):
            full_path = file_path
        else:
            print(f"File {full_path} does not exist")
            return set()
    try:
        parser, _ = _get_py_parser()
        with open(full_path, "rb") as f:
            raw = f.read()
        tree = parser.parse(raw)
        root = tree.root_node

        module_vars_and_classes = _module_level_variable_and_class_names(raw, root)
        def_map = _find_definitions_by_name(raw, root)

        used_globals = set()
        for item in node_keys:
            if len(item) == 2:
                category, short_name = item[0], item[1]
                method_signature = None
            else:
                category, short_name, method_signature = item[0], item[1], (item[2] if len(item) > 2 else None)
            ast_node = _resolve_ast_node(def_map, raw, category, short_name, method_signature)
            if ast_node is None:
                continue
            local_defs = _names_defined_in_scope(ast_node, raw)
            used = _identifiers_used_in_scope(ast_node, raw, local_defs)
            for name in used:
                if name in module_vars_and_classes:
                    used_globals.add(name)
        return used_globals
    except Exception as e:
        print(f"Error extracting used globals from {full_path}: {e}")
        return set()
