"""Resolve candidate holdout historical exposure from existing run records only."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cde_v3.common import *

start=time.time(); exposed=set(); scanned=[]
def extract(obj):
    if isinstance(obj,dict):
        if isinstance(obj.get('sample_id'),str): exposed.add(obj['sample_id'])
        for value in obj.values(): extract(value)
    elif isinstance(obj,list):
        for value in obj: extract(value)
    elif isinstance(obj,str) and obj.startswith(('LSUI/','LSUI_author_examples_S1_only/')): exposed.add(obj)
for p in (ROOT/'runs').rglob('*.jsonl'):
    if RUN in p.parents: continue
    if 'source_snapshot' in p.parts or 'manifests' in p.parts: continue
    try:
        for line in p.read_text().splitlines(): extract(json.loads(line))
        scanned.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    except (UnicodeError,json.JSONDecodeError): raise RuntimeError('Cannot audit '+str(p))
# progress + per-image protocol IDs; don't treat frozen full manifest entries as execution.
for folder in ['s1','s2_grouped_v1','explore_ag_single_seed_v2_20261003']:
    for p in (ROOT/'runs'/folder).rglob('protocol.json'):
        extract(read(p)); scanned.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
report=read(RUN/'holdout_candidate_audit.json')
usedstems={s.rsplit('/',1)[-1] for s in exposed if s.startswith('LSUI')}
kept=[r for r in report['kept'] if r['sample_id'] not in exposed and Path(r['image_path']).stem not in usedstems]
for p in sorted((ROOT/'runs/s2_grouped_v1/lsui').glob('*/progress.json')):
    assert read(p)['status']=='queued','LSUI historical training exposure needs review'
result={'status':'eligible_within_audited_exposure_scope','root':report['root'],'rows':kept,'rows_sha256':digest(kept),'count':len(kept),'legacy_records_sha256':digest(scanned),'scanned_record_files':scanned,'excluded_record_exposure':len(report['kept'])-len(kept),'upstream_exposure_unknown':['DepthAnythingV2','VGG19','LPIPS_VGG16'],'limitations':report['limitations'],'no_scores_accessed':True,'same_domain_LSUI_not_new_data_domain':True,'cpu_seconds':time.time()-start}
write(RUN/'holdout_frozen_candidates.json',result)
(DOC/'HOLDOUT_AUDIT.md').write_text('# 新留出审计\n\n'+str(len(kept))+' 张LSUI旧验证图在现有可审计执行日志中无模型训练/开发评分记录，排除S1作者样例、与已知UIEB/LSUI测试的未决dHash关系。完整名单与记录文件哈希见holdout_frozen_candidates.json。\n\n该结论仅限日志与内容筛查覆盖范围；dHash不能证明绝对场景独立，DA/VGG及LPIPS上游暴露仍未知。它是同一LSUI域的新样本确认，不是新数据域。所有分数只能在final_freeze后生成。\n')
print({'eligible_count':len(kept),'scanned':len(scanned),'cpu_seconds':result['cpu_seconds']})
