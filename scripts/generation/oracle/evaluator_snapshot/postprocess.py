"""Deterministic extraction only; no whitespace or logical program repairs."""
import re
import ast
import textwrap


def extract_code(text):
    # Unlike \\s*, [ \\t]* cannot swallow the newline and the first code indent.
    blocks = re.findall(r'```(?:[a-zA-Z0-9_+.-]+)?[ \t]*\r?\n([\s\S]*?)(?:\r?\n)?```', text)
    return (max(blocks,key=lambda b:len(b.strip())) if blocks else text).strip('\r\n')


def normalize_first_line(code):
    """Formatting-only recovery, independent of references and test outcomes."""
    def valid(s):
        try:
            ast.parse('async def __oracle_format_probe__():\n'+textwrap.indent(textwrap.dedent(s),'    '))
            return True
        except (SyntaxError,ValueError):return False
    if valid(code):return code,'unchanged_syntax_valid'
    lines=code.splitlines(keepends=True)
    ids=[i for i,l in enumerate(lines) if l.strip()]
    if len(ids)<2:return code,'unchanged_no_unique_repair'
    first=ids[0];indent=len(lines[first])-len(lines[first].lstrip(' '))
    levels=sorted({len(lines[i])-len(lines[i].lstrip(' ')) for i in ids[1:]})
    candidates=[]
    for n in levels:
        if n<=indent:continue
        changed=list(lines);changed[first]=' '*n+lines[first].lstrip(' ')
        candidate=''.join(changed)
        if valid(candidate):candidates.append(candidate)
    if len(candidates)==1:return candidates[0],'first_line_indent_only_unique_syntax_repair'
    return code,'unchanged_no_unique_repair'
