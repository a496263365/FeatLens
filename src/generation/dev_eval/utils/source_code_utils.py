import os
from typing import Tuple


def read_line_range(file_path: str, start_line: int, end_line: int) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if start_line < 1 or end_line < 1:
        raise ValueError("start_line/end_line must be 1-based positive integers")
    if end_line > len(lines):
        raise ValueError(
            f"Requested lines {start_line}-{end_line} exceed file length {len(lines)}: {file_path}"
        )
    return "".join(lines[start_line - 1 : end_line])


# Read the function signature from the original Python file using the line number of the signature
def resolve_signature(
    source_code_dir: str, completion_path: str, signature_pos: Tuple[int, int]
) -> Tuple[str, str]:
    abs_file = os.path.join(source_code_dir, completion_path)
    signature = read_line_range(abs_file, signature_pos[0], signature_pos[1])
    return abs_file, signature


def get_class_skeleton(project_path, file_path, class_qname):
    """
    Extract only the "skeleton" of a class from a file:
    - Full class definition line and __init__ (to preserve self.xx member variables).
    - Other methods: only the function signature, body replaced by "...".
    This shortens the text for embedding.
    """
    full_path = os.path.join(project_path, file_path.lstrip('/'))
    if not os.path.exists(full_path):
        if os.path.exists(file_path):
            full_path = file_path
        else:
            return ""

    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python as tspython

        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)

        with open(full_path, 'r') as f:
            content = f.read()

        tree = parser.parse(bytes(content, "utf8"))
        root_node = tree.root_node
        short_name = class_qname.split('.')[-1]

        def find_class_node(node, target_name):
            if node.type == 'class_definition':
                name_node = node.child_by_field_name('name')
                if name_node and content[name_node.start_byte:name_node.end_byte] == target_name:
                    return node
            for child in node.children:
                res = find_class_node(child, target_name)
                if res:
                    return res
            return None

        target_node = find_class_node(root_node, short_name)
        if not target_node:
            print(f"Class {short_name} not found in {full_path}")
            return ""

        # Class body: tree-sitter-python uses field 'body' for the suite
        parts = []
        first_line_end = content.find('\n', target_node.start_byte)
        if first_line_end == -1:
            first_line_end = target_node.end_byte
        else:
            first_line_end += 1  # include the newline
        parts.append(content[target_node.start_byte:first_line_end])
        block_node = target_node.child_by_field_name('body')
        if not block_node:
            return content[target_node.start_byte:target_node.end_byte]

        def process_function(def_node, outer_start_byte, outer_end_byte):
            name_node = def_node.child_by_field_name('name')
            method_name = content[name_node.start_byte:name_node.end_byte] if name_node else ""
            body_node = def_node.child_by_field_name('body')
            if body_node is None:
                for c in reversed(def_node.children):
                    if c.type == 'block':
                        body_node = c
                        break
            if method_name == '__init__':
                return content[outer_start_byte:outer_end_byte], True
            if body_node is not None:
                sig_end = body_node.start_byte
                return content[outer_start_byte:sig_end] + "\n        ...\n", False
            return content[outer_start_byte:outer_end_byte], False

        # Preserve newlines/whitespace between statements: each node's range extends
        # to the next sibling's start_byte (gap between nodes is part of original format).
        children = block_node.children
        for i, child in enumerate(children):
            next_start = children[i + 1].start_byte if i + 1 < len(children) else child.end_byte
            gap_after = content[child.end_byte:next_start]  # newlines/indent before next stmt

            if child.type == 'function_definition':
                snippet, _ = process_function(child, child.start_byte, child.end_byte)
                parts.append(snippet + gap_after)
            elif child.type == 'decorated_definition':
                def_node = child.child_by_field_name('definition')
                if def_node and def_node.type == 'function_definition':
                    snippet, _ = process_function(def_node, child.start_byte, child.end_byte)
                    parts.append(snippet + gap_after)
                else:
                    parts.append(content[child.start_byte:next_start])
            else:
                parts.append(content[child.start_byte:next_start])

        return "".join(parts)

    except Exception as e:
        print(f"Error extracting class skeleton from {full_path}: {e}")
        try:
            with open(full_path, 'r') as f:
                return f.read()
        except Exception:
            return ""
