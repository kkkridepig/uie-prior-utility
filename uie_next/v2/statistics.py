import math
import numpy as np


def paired_stats(delta,groups,repeats=5000,seed=20261017):
    values=np.asarray(delta,dtype=np.float64);names=np.asarray(groups)
    if not len(values) or not np.isfinite(values).all():raise ValueError('nonfinite paired metrics')
    unique,group_index=np.unique(names,return_inverse=True)
    rng=np.random.RandomState(seed)
    sampled=rng.randint(len(unique),size=(repeats,len(unique)))
    group_counts=np.stack([np.bincount(s,minlength=len(unique)) for s in sampled])
    weights=group_counts[:,group_index];denom=weights.sum(1)
    draws=(weights*values).sum(1)/denom
    harm=(weights*(values < -.1)).sum(1)/denom
    order=np.argsort(values,kind='stable');sw=weights[:,order]
    target=np.ceil(.1*denom).astype(np.int64)
    previous=np.cumsum(sw,axis=1)-sw
    take=np.minimum(sw,np.maximum(0,target[:,None]-previous))
    tail=(take*values[order]).sum(1)/target
    return {'mean_image':float(values.mean()),'median_image':float(np.median(values)),
            'quantiles_image':np.quantile(values,[.05,.1,.9,.95]).tolist(),
            'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'worst10_mean':float(np.sort(values)[:math.ceil(.1*len(values))].mean()),
            'worst10_ci95':np.quantile(tail,[.025,.975]).tolist(),
            'harm_rate':float((values < -.1).mean()),'harm_rate_ci95':np.quantile(harm,[.025,.975]).tolist(),
            'improvement_rate':float((values > .1).mean()),'n_images':len(values),'n_groups':len(unique),
            'aggregation':'image_weighted','draws':repeats,'seed':seed,'scope':'exploratory_single_seed_not_selection_adjusted'}
