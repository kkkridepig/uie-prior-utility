def grid(method):
    if method in ['B2','B4']:
        return [{'alpha':a} for a in [0,.25,.5,.75,1]]
    if method.startswith('G'):
        return [{'return_base':True}]+[{'margin':m,'temperature':t} for m in [.5,.25,0] for t in [1,2,.5]]
    return [{'return_base':True}]+[{'tau':t,'lambda':l} for t in [1e-4,1e-5,0] for l in [1e-3,1e-4,1e-6]]


def select(rows):
    feasible=[(i,r) for i,r in enumerate(rows) if r['ssim_delta']>=-.001 and r['lpips_delta']<=.002]
    if not feasible: raise ValueError('return_base must be feasible')
    best=feasible[0]
    for item in feasible[1:]:
        if item[1]['psnr']>best[1]['psnr']+1e-8:best=item
    return best[0]
