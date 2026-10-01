"""Opt-in content-addressed cache. Never keys derived text by task/function ID.
Only successful outputs are persisted. New run namespaces must be used when
model weights, prompt implementations or preprocessing change.
"""
import hashlib,json,os,threading,uuid
from pathlib import Path
_locks={};_guard=threading.Lock()
def cache_path(kind,payload):
 root=os.getenv('FEATLENS_CONTENT_CACHE')
 if not root:return None
 digest=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 p=Path(root)/kind/(digest+'.json');p.parent.mkdir(parents=True,exist_ok=True)
 return p
def read(p):return json.loads(p.read_text()) if p and p.exists() else None
def write(p,value):
 if p is None:return
 tmp=p.with_suffix('.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False));os.replace(tmp,p)
def lock(p):
 with _guard:return _locks.setdefault(str(p),threading.Lock())

class CachedCompletions:
    def __init__(self,base):self.base=base
    def create(self,**kwargs):
        from types import SimpleNamespace
        import time
        p=cache_path('llm-response',kwargs)
        with lock(p if p else ('uncached-call',uuid.uuid4().hex)):
            value=read(p)
            if value is None:
                start=time.monotonic();response=self.base.create(**kwargs)
                content=response.choices[0].message.content
                # Cache valid JSON only; invalid responses must remain retryable.
                try: json.loads(content)
                except (ValueError,TypeError):return response
                value={'content':content,'usage':response.usage.model_dump() if response.usage else {},'seconds':time.monotonic()-start}
                write(p,value)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=value['content']))])
