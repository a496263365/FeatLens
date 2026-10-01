"""Format-only extraction. Never change model logic or use tests to repair code."""
import ast, re, textwrap

def normalize(code,namespace):
    original=code; notes=[]
    code=code.replace('\r\n','\n').replace('\r','\n')
    m=re.fullmatch(r'\s*```(?:python|Python|py)?\s*\n(.*?)\n?```\s*',code,re.S)
    if m:code=m.group(1);notes.append('remove_markdown_fence')
    candidate=textwrap.dedent(code)
    try:
        tree=ast.parse(candidate)
        if len(tree.body)==1 and isinstance(tree.body[0],(ast.FunctionDef,ast.AsyncFunctionDef)) and tree.body[0].name==namespace.rsplit('.',1)[-1]:
            n=tree.body[0];lines=candidate.splitlines(keepends=True)
            first=n.body[0];last=n.body[-1]
            # Only remove a redundant outer target definition; keep its body byte-for-byte.
            if first.lineno==n.lineno:
                code=lines[first.lineno-1][first.col_offset:last.end_col_offset]
            else:code=''.join(lines[first.lineno-1:last.end_lineno])
            notes.append('unwrap_repeated_target_definition')
    except SyntaxError:pass
    return code, notes
