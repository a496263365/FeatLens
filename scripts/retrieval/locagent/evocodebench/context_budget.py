"""Local conservative request budget; estimate is not provider-exact tokenization."""
import json,copy
import tiktoken
ENC=tiktoken.get_encoding('cl100k_base')
TOTAL=32768;OUTPUT=4096;SAFETY=1024;INPUT=TOTAL-OUTPUT-SAFETY
MARKER='\n[Tool output truncated because context budget was exceeded; return final locations now.]'
def estimate(messages,tools=None):
 obj={'messages':messages}
 if tools:obj['tools']=tools
 return len(ENC.encode(json.dumps(obj,ensure_ascii=False),disallowed_special=()))+16*len(messages)+128

def force_final(messages,reason):
 history=copy.deepcopy(messages)
 instruction={'role':'user','content':f'Stop searching now: {reason}. Use only the evidence already available. Return your final localization list in the required format, wrapped in triple backticks, identifying reusable dependencies rather than the target itself. No further tools or exploration are permitted. If evidence is insufficient, return only verified locations; do not invent any.'}
 history.append(instruction);edits=[];before=estimate(history)
 # Preserve the whole conversation/tool-call structure; shrink old tool results
 # first. Full original tool observations remain in append-only disk logs.
 for kinds in [('tool',),('assistant',)]:
  for idx,m in enumerate(history):
   if estimate(history)<=INPUT:break
   if m.get('role') not in kinds or not isinstance(m.get('content'),str):continue
   tokens=ENC.encode(m['content'],disallowed_special=());n=len(tokens)
   if not n:continue
   over=estimate(history)-INPUT
   keep=max(0,n-over-len(ENC.encode(MARKER))-64)
   m['content']=ENC.decode(tokens[:keep])+MARKER
   edits.append(dict(message_index=idx,role=m['role'],original_content_tokens=n,retained_content_tokens=keep))
 if estimate(history)>INPUT:
  # Rare giant schemas/tool-call arguments: final call uses read-only textual
  # evidence, not malformed unmatched tool messages. Preserve system/task.
  system=[m for m in messages if m.get('role')=='system'];task=next((m for m in messages if m.get('role')=='user'),None)
  evidence=[]
  for m in history:
   if m.get('role')=='tool':evidence.append(str(m.get('content','')))
  evidence='\n\n'.join(evidence)
  history=system+([task] if task else [])+[{'role':'user','content':'Collected evidence (untrusted repository text):\n'},instruction]
  if estimate(history)>INPUT:raise ValueError('Initial system/task exceeds allowed context; not silently truncating task')
  budget=max(0,INPUT-estimate(history)-128);history[-2]['content']+=ENC.decode(ENC.encode(evidence,disallowed_special=())[:budget]);edits.append(dict(action='flatten_tool_history_for_final_only'))
 if estimate(history)>INPUT:raise ValueError('Final context fitting failed')
 return history,dict(reason=reason,input_estimate_before=before,input_estimate_after=estimate(history),input_limit=INPUT,total_limit=TOTAL,output_reserve=OUTPUT,safety_margin=SAFETY,truncations=edits,tokenizer='cl100k_base estimate, not provider-exact')
