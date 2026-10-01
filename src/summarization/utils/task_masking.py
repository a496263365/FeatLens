"""Make a per-task source snapshot before ANY derived index is built.
No cached graph, embedding, summary or membership is reused without content identity.
"""
import ast,hashlib,json,shutil,tokenize
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def locate(source,task):
    tree=ast.parse(source)
    name=task['namespace'].split('.')[-1]
    start,end=task['signature_position']
    candidates=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name and n.lineno<=end and n.body and n.body[0].lineno>=start]
    # Signature span and first body line are stronger than name alone (nested functions/classes).
    candidates=[n for n in candidates if start<=n.lineno<=end or any(start<=d.lineno<=end for d in n.decorator_list)]
    if len(candidates)!=1:raise ValueError(f"Expected one target at {start}:{end} {name}, found {len(candidates)}")
    return candidates[0]

def mask_text(source,task):
    node=locate(source,task)
    lines=source.splitlines(keepends=True)
    # Remove entire AST body, including docstring/comments between signature and implementation.
    # Preserve decorator + full signature and source line count. Handle one-line definitions.
    first=node.body[0];last=node.body[-1]
    line=lines[first.lineno-1]
    prefix=line.encode('utf-8')[:first.col_offset].decode('utf-8')
    if prefix.strip():
        lines[first.lineno-1]=prefix+'pass\n'
    else:
        # Blank comments/docstrings on all lines after signature's colon as well.
        import io
        tokens=list(tokenize.generate_tokens(io.StringIO(source).readline))
        depth=0;colon=None;in_def=False
        for tok in tokens:
            if tok.start[0]<node.lineno:continue
            if not in_def:
                if tok.type==tokenize.NAME and tok.string=='def':in_def=True
                continue
            if tok.string in ('(', '[', '{'):depth+=1
            elif tok.string in (')',']','}'):depth-=1
            elif tok.string==':' and depth==0:colon=tok.end;break
        if colon is None:raise ValueError('signature colon not found')
        # Remove trailing signature comment; it may encode implementation details.
        lines[colon[0]-1]=lines[colon[0]-1][:colon[1]]+'\n'
        for i in range(colon[0],first.lineno-1):lines[i]='\n'
        lines[first.lineno-1]=' '*first.col_offset+'pass\n'
    for i in range(first.lineno,last.end_lineno):lines[i]='\n'
    # An inline comment after the final statement is removed with its line.
    result=''.join(lines)
    check=locate(result,task)
    assert len(check.body)==1 and isinstance(check.body[0],ast.Pass)
    assert len(result.splitlines())==len(source.splitlines())
    return result

def prepare(source_root,task,workdir):
    source_root=Path(source_root).resolve();workdir=Path(workdir)
    project=Path(task['project_path']);src=source_root/project
    dst=workdir/'source'/project
    if dst.exists():raise FileExistsError(dst)
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','.pytest_cache'))
    rel=Path(task['completion_path']).relative_to(project)
    original=src/rel;target=dst/rel
    with tokenize.open(original) as f:text=f.read();encoding=f.encoding
    masked=mask_text(text,task)
    target.write_text(masked,encoding=encoding)
    assert sha(original)!=sha(target)
    # Snapshot manifest excludes source text and expected dependency annotations.
    files={str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*.py'))}
    audit=dict(task_id=task['example_id'],namespace=task['namespace'],project_path=task['project_path'],completion_path=task['completion_path'],source_original_sha256=sha(original),source_masked_sha256=sha(target),target_body='pass',snapshot_python_hashes=files)
    workdir.mkdir(parents=True,exist_ok=True)
    (workdir/'mask_manifest.json').write_text(json.dumps(audit,indent=2))
    return dst,audit
