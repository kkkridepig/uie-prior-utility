import math


def fit_scales(guard,rows):
    """Rows hold real seven-view label tensors; FIT-only, source/view equal."""
    v2=[];u2=[];e2=[];fractions=[]
    for sample_id,views in rows:
        guard.check(sample_id,'label_scales')
        if len(views)!=7:raise ValueError('all seven real candidate views required')
        for l in views:
            active=l['active'];fractions.append(float(active.float().mean()))
            if active.any():v2.append(float(l['v'][active].square().mean()))
            u2.append(float(l['U'].square().mean()))
        e2.append(float(views[0]['l0'].mean()))
    if not v2:raise ValueError('no active labels; cannot hide behind scale floors')
    return {'s_v':max(math.sqrt(sum(v2)/len(v2)),1e-3),'s_U':max(math.sqrt(sum(u2)/len(u2)),1e-4),
            's_e2':max(sum(e2)/len(e2),1e-4),'source_count':len(e2),'active_image_view_count':len(v2),
            'active_fraction':sum(fractions)/len(fractions),'role':'utility_fit'}
