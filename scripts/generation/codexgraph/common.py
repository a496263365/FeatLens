
# Locate the package from this file, not from the caller's working directory.
from pathlib import Path as _PortablePath
import sys as _portable_sys
_PORTABLE_ROOT = next(p for p in _PortablePath(__file__).resolve().parents
                      if (p / "featlens_paths.py").is_file())
if str(_PORTABLE_ROOT) not in _portable_sys.path:
    _portable_sys.path.insert(0, str(_PORTABLE_ROOT))
from featlens_paths import (package_path as _package_path,
                            external_path as _external_path,
                            model_location as _model_location,
                            evaluation_python as _evaluation_python,
                            temporary_path as _temporary_path)

from pathlib import Path
import json, hashlib, os
ROOT = (_package_path())
BASE = (_package_path() / 'results/generation/deveval/codexgraph/deepseek_v3_2')
OUT = (_package_path() / 'results/generation/deveval/codexgraph/gpt_5_mini')
CODE = (_package_path() / 'scripts/generation/codexgraph')
REFERENCE = _external_path('FEATLENS_DEVEVAL_SOURCE_ROOT', 'external/deveval/source_code')
MODEL = 'gpt-5-mini'
def rows(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def sha_bytes(v): return hashlib.sha256(v).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def dump(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n'); tmp.replace(p)
def jsonl(p,xs):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp'); tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in xs)); tmp.replace(p)
def slug(p): return p.replace('/','__')
