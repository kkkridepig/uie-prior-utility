"""Scene-aware paired summaries; intervals are exploratory, not innovation evidence."""
import numpy as np
from scripts.cde_v3.common import key

def paired_summary(rows,method,control,metric='psnr',seed=20261004):
    per_image={}
    for r in rows:
        a,b=r['scores'][method][metric],r['scores'][control][metric]
        if not isinstance(a,(float,int)) or not isinstance(b,(float,int)): raise ValueError('Infinite/undefined paired metric; retain event before defining aggregate')
        per_image.setdefault((r['sample_id'],r['scene_id']),[]).append(a-b)
    groups={}; values=[]
    for (sid,scene),ds in per_image.items():
        delta=float(np.mean(ds)); groups.setdefault(scene or sid,[]).append(delta); values.append(delta)
    values=np.array(values); groupvals=list(groups.values()); rng=np.random.RandomState(key(seed,'bootstrap',method,control,metric)%2**32)
    draws=[]
    for _ in range(1000):
        ix=rng.randint(len(groupvals),size=len(groupvals)); draws.append(float(np.mean([x for j in ix for x in groupvals[j]])))
    return {'images':len(values),'scene_groups':len(groups),'mean':float(values.mean()),'scene_equal_mean':float(np.mean([np.mean(v) for v in groupvals])),'median':float(np.median(values)),'improved_fraction':float(np.mean(values>0)),'scene_bootstrap_ci95':np.quantile(draws,[.025,.975]).tolist(),'remove_largest5_mean':float(np.sort(values)[:-5].mean()) if len(values)>5 else None,'resampling':'scene_group1000_image_weighted; one pretrained parent'}
