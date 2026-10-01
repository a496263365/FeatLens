"""Recompute reported metrics from per-task artifacts, with source and input integrity checks."""
import collections,csv,datetime,json,sys
from common import *
sys.path.insert(0,str(ROOT/'scripts/statistics'))
from code_len_stats import python_completion_token_len
from postprocess import normalize

def deps(r,k):return set(x for xs in (r.get(k) or {}).values() for x in xs)
def indexed(p):
    rr=rows(p);d={r['namespace']:r for r in rr};assert len(rr)==len(d),p;return d

def main():
    benchmark=indexed(OUT/'inputs/eval_tasks.jsonl')
    comps=indexed(OUT/'completions/gpt-5-mini_evaluated.jsonl')
    tests=indexed(OUT/'gpt-5-mini_test_output.jsonl');recalls=indexed(OUT/'gpt-5-mini_recall_output.jsonl')
    assert set(benchmark)==set(comps)==set(tests)
    assert {ns for ns,t in benchmark.items() if deps(t,'dependency')}==set(recalls)
    assert len(benchmark)==1430 and len(recalls)==1146
    original=indexed(BASE/'completions/deepseek-v3.2_q2_final.jsonl')
    for ns,c in comps.items():
        assert tests[ns]['completion']==c['completion']
        if ns in recalls:assert recalls[ns]['completion']==c['completion']
        assert c['retrieved_entities']==original[ns]['retrieved_entities']
    manifest=rows(OUT/'inputs/manifest.jsonl')
    assert all(sha_file(r['prompt_file'])==r['prompt_sha256'] for r in manifest)
    raw={ns:json.loads((OUT/'tasks'/f"{c['idx']:04d}"/'generation.json').read_text()) for ns,c in comps.items()}
    assert all(raw[ns]['frozen_prompt_sha256']==r['prompt_sha256'] for r in manifest for ns in [r['namespace']])
    # Rebuild merged raw completions after the logged signature-only regeneration.
    jsonl(OUT/'completions/gpt-5-mini_final.jsonl',[raw[r['namespace']] for r in manifest])
    corrected={r['namespace'] for r in json.loads((OUT/'input_audit.json').read_text())['corrected_tasks']}
    fb=set(json.loads((BASE/'inputs/fallback_q1_namespaces.json').read_text()))
    history_tests=indexed(BASE/'deepseek-v3.2_q2_final_test_output.jsonl');history_recall=indexed(BASE/'deepseek-v3.2_q2_final_recall_output.jsonl')
    def metrics(names,c= comps,t=tests,r=recalls):
        names=set(names);p=sum(t[ns]['Result']=='Pass' for ns in names);drnames=names&set(r);ratios=[];hitt=tot=0
        for ns in drnames:
            ref=deps(benchmark[ns],'dependency'); hit=len(ref&deps(r[ns],'generated_dependency'));ratios.append(hit/len(ref));hitt+=hit;tot+=len(ref)
        return {'tasks':len(names),'passed':p,'Pass@1_percent':100*p/len(names),'DIR_tasks':len(drnames),'DIR_percent':100*sum(ratios)/len(ratios) if ratios else None,'DIR_micro_percent':100*hitt/tot if tot else None,'DIR_null_parses':sum(r[ns].get('generated_dependency') is None for ns in drnames),'LOC_lexical_tokens':sum(python_completion_token_len(c[ns]['completion']) for ns in names)/len(names),'results':dict(collections.Counter(t[ns]['Result'] for ns in names))}
    cohorts={'full1430':metrics(comps),'query2_only':metrics(set(comps)-fb),'query1_fallback':metrics(fb),'unchanged_coordinates1421':metrics(set(comps)-corrected),'historical_deepseek_full':metrics(comps,original,history_tests,history_recall),'historical_deepseek_unchanged_coordinates1421':metrics(set(comps)-corrected,original,history_tests,history_recall)}
    usage=collections.Counter();attempts=0;models=collections.Counter();errors=[];no_usage=0
    for folder in (OUT/'tasks').glob('*/attempt_*'):
        attempts+=1
        if (folder/'response.json').exists():
            resp=json.loads((folder/'response.json').read_text());u=resp.get('usage') or {};models[resp.get('model')]+=1
            usage.update({k:v for k,v in u.items() if isinstance(v,(int,float))})
            usage['reasoning_tokens']+=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
            usage['cached_prompt_tokens']+=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)
            if not u:no_usage+=1
        elif (folder/'error.json').exists():errors.append(str(folder))
    environment_errors=[]
    needles=['ModuleNotFoundError','ImportError','not found:','invalid command','No module named','error: [Errno','Could not find','DistributionNotFound','VersionConflict','ConnectionError','ConnectionRefusedError','No such file or directory']
    for p in (OUT/'evaluation').glob('*/failure.log'):
        for r in rows(p):
            text=r.get('message','');hits=[x for x in needles if x in text]
            if hits:environment_errors.append({'namespace':r.get('namespace'),'flags':hits,'log':str(p)})
    jsonl(OUT/'environment_failure_flags.jsonl',environment_errors)
    byproject=[]
    for project in sorted({t['project_path'] for t in benchmark.values()}):
        m=metrics(ns for ns,t in benchmark.items() if t['project_path']==project);byproject.append({'project_path':project,**{k:v for k,v in m.items() if k!='results'}})
    with (OUT/'per_project_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(byproject[0]));w.writeheader();w.writerows(byproject)
    result={'cohorts':cohorts,'recorded_api_usage':dict(usage),'api_attempts_including_signature_correction':attempts,'models_recorded':dict(models),'api_error_attempts_usage_unknown':errors,'responses_missing_usage':no_usage,'corrected_namespaces':sorted(corrected),'format_only_postprocess_tasks':sum(bool(c['postprocess_notes']) for c in comps.values()),'generation_failed':sum(not r['generation_ok'] for r in raw.values()),'environment_flagged_failure_tasks':len({r['namespace'] for r in environment_errors}),'artifact_alignment_validated':True,'notes':['Tool observations frozen from DS retrieval, no new retrieval. Valid decisions reconstructed from formatted history; raw invalid JSON turns not retained. Not a byte-identical model switch.','9 Datasette signature/body coordinate issues corrected independently of model outputs/tests. Historical DS untouched; corrected tasks must not be used for a strict historical paired comparison.','Paper LOC actually counts Python lexical tokens under Python 3.10; not physical lines.','Test failures include environment/test-collection failures; all remain in denominators.']}
    dump(OUT/'FINAL_METRICS.json',result)
    g=cohorts['full1430'];d=cohorts['historical_deepseek_full']
    text=f'''# 固定 CodexGraph 检索 → GPT-5-mini：全量结果

范围：历史筛选的 90 个项目 / 1430 个任务，DIR 定义于 1146 个有依赖任务。不是原版 1825 任务全体。

## 生成结果

| 条件 | Pass@1 (%) | DIR (%) | LOC（词法 token） |
|---|---:|---:|---:|
| DeepSeek-V3.2 历史结果（原行号） | {d['Pass@1_percent']:.2f} | {d['DIR_percent']:.2f} | {d['LOC_lexical_tokens']:.2f} |
| 固定检索 → GPT-5-mini（本次） | {g['Pass@1_percent']:.2f} | {g['DIR_percent']:.2f} | {g['LOC_lexical_tokens']:.2f} |

GPT 通过 {g['passed']}/1430；结果分布 {g['results']}。DIR micro {g['DIR_micro_percent']:.2f}%；{g['DIR_null_parses']} 条依赖解析为 None，计零不剔除。

检索集合固定：1329 条 query_2、101 条 query_1 fallback。沿用原评分器的 DR@10/15/20 为 0.239407 / 0.309325 / 0.344193，不用 GPT 重新报告的实体列表改变 DR。

## 限制与协议

- 历史日志没有完整 API 请求、未保存无效 JSON 回合；本次保留工具返回文本并重建有效动作，套用同一 96000 粗略 token 输入裁剪规则，因此不是逐字相同请求重放。
- {len(corrected)} 个 Datasette 任务的原始签名/函数体行号漂移已独立修正。8 个在准备时修正，1 个在测试前发现多行签名漂移后修正并只重生成该任务；原请求及响应已归档。
- 上表是描述性对照，不应声称全部 1430 任务仅改变一个变量。未修正的 {1430-len(corrected)} 任务独立指标见 FINAL_METRICS.json。
- 模型请求 `gpt-5-mini`，返回模型 {dict(models)}；reasoning_effort=minimal，不声称关闭 reasoning；输出预算 4096，无可解析代码时升为8192重试，不按测试结果重采样。
- 输出后处理仅去掉 Markdown 或重复的外层目标函数签名，未修补算法；原始模型输出全部保留。
- 使用相同 DevEval pass_k.py / recall_k.py 和 featlens-eval 环境，每项目独立源码副本；Pass/DIR 在同一项目内顺序执行，不写旧实验源码或结果。
- {result['environment_flagged_failure_tasks']} 个失败任务的日志含环境/收集错误关键词（启发式标记，非精确分类）。这些仍计为失败；Pass@1 不是单纯模型逻辑错误率。
- 记录 token：{dict(usage)}。API错误且用量未知尝试 {len(errors)}；响应无usage {no_usage}。
- DR 是沿用现有 CodexGraph 名称映射评分，不在本次任务中声称已经对齐 FeatLens 的代码级宽松规则。

## 可复现资产

- `inputs/manifest.jsonl`：1430任务来源、检索来源、历史与prompt哈希。
- `inputs/prompts/`：冻结生成输入，无 DeepSeek 最终答案。
- `tasks/*/attempt_*`：每次真实API请求/响应/用量/错误（不记录凭证）。
- `completions/gpt-5-mini_final.jsonl`：原始解码代码。
- `completions/gpt-5-mini_evaluated.jsonl`：仅格式处理后的评测代码。
- `gpt-5-mini_test_output.jsonl` / `gpt-5-mini_recall_output.jsonl`：逐任务评测。
- `per_project_metrics.csv` / `FINAL_METRICS.json`：可重算汇总及分组。
'''
    (OUT/'FINAL_REPORT.md').write_text(text)
    files=[OUT/'FINAL_METRICS.json',OUT/'completions/gpt-5-mini_evaluated.jsonl',OUT/'gpt-5-mini_test_output.jsonl',OUT/'gpt-5-mini_recall_output.jsonl',OUT/'inputs/eval_tasks.jsonl',OUT/'inputs/manifest.jsonl']
    dump(OUT/'final_artifact_sha256.json',{str(p.relative_to(OUT)):sha_file(p) for p in files})
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
