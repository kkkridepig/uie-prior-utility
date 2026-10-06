"""Aggregate frozen test evidence without changing model selection."""
import json
from pathlib import Path
import numpy as np
from scripts.cde_v3.common import *

def final_summary():
    d=read(RUN/'development_decision.json'); method=d.get('method'); seeds=d['seeds']; all_results={}
    for seed in seeds:
        all_results[str(seed)]={}
        for p in sorted((RUN/'final'/str(seed)).glob('*/summary.json')): all_results[str(seed)][p.parent.name]=read(p)
    summaries={}; supported=bool(method and len(seeds)==3 and d['status']=='same_parent_multiseed_supported')
    holdout_present=all('confirm_holdout' in all_results[str(seed)] for seed in seeds)
    if method and holdout_present:
        controls=all_results[str(seeds[0])]['confirm_holdout']['comparisons'].keys()
        for control in controls:
            values=[all_results[str(s)]['confirm_holdout']['comparisons'][control]['psnr']['mean'] for s in seeds]
            summaries[control]={'seed_deltas':values,'mean':float(np.mean(values)),'std':float(np.std(values,ddof=1)) if len(values)>1 else None,'min':min(values),'max':max(values),'positive_each_seed':all(v>0 for v in values)}
            supported=supported and all(v>0 for v in values) and np.mean(values)>=.1
            ssim=[all_results[str(s)]['confirm_holdout']['comparisons'][control]['ssim']['mean'] for s in seeds]
            lpips=[all_results[str(s)]['confirm_holdout']['comparisons'][control]['lpips']['mean'] for s in seeds]
            summaries[control].update(ssim_deltas=ssim,lpips_deltas=lpips)
            supported=supported and all(v>=-.002 for v in ssim) and all(v<=.01 for v in lpips)
    status='candidate_confirmed_on_available_holdout' if supported else 'new_holdout_not_confirmed' if method and holdout_present else 'development_complete_confirmation_blocked' if method else 'pilot_no_supported_gain'
    result={'status':status,'development_status':d['status'],'method':method,'per_seed_datasets':all_results,'new_holdout_multiseed_psnr':summaries,'evidence_scope':'same pretrained parent; one pilot plus two fixed repeats; known exposure audit only','legacy_test_role':'legacy_exposed_regression','not_innovation_proof':True}
    write(RUN/'final_summary.json',result)
    (DOC/'FINAL_METRICS.md').write_text('# 冻结后指标\n\n'+status+'\n\n```json\n'+json.dumps(result,indent=2,ensure_ascii=False)+'\n```\n\n全部逐图记录、场景bootstrap、固定样例和后验失败面板在final/<seed>/<role>/中。\n')
    return status
