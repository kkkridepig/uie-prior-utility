"""Deterministic, predeclared producer and finite deployment-grid selection."""
from functools import cmp_to_key


def eligibility(row):
    finite = row.get('finite_pass', False)
    sensitivity = row['S'] >= 1e-4
    if finite and sensitivity and row['H32'] >= .15 and row['G32'] >= .05 and row['Gp'] >= .10:
        return 'COARSE_HEADROOM'
    if finite and sensitivity and row['Hp'] >= .15 and row['Gp'] >= .10:
        return 'FINE_ONLY_HEADROOM'
    return 'NO_QUALIFIED_PRODUCER_ON_MODEL_VAL'


def choose_producer(rows, require_sensitivity=True):
    pool = [dict(r, qualification=eligibility(dict(r, S=r['S'] if require_sensitivity else 1.))) for r in rows if r['training_step'] > 0]
    coarse = [r for r in pool if r['qualification'] == 'COARSE_HEADROOM']
    qualified = coarse or [r for r in pool if r['qualification'] == 'FINE_ONLY_HEADROOM']
    def compare(a,b):
        keys = ['oracle_block32_psnr','oracle_pixel_psnr'] if coarse else ['oracle_pixel_psnr','oracle_block32_psnr']
        for k in keys:
            if abs(a[k]-b[k]) > 1e-8: return -1 if a[k] > b[k] else 1
        return -1 if (a['training_step'],a['checkpoint_id']) < (b['training_step'],b['checkpoint_id']) else 1
    ranking = sorted(qualified, key=cmp_to_key(compare))
    return {'selected': ranking[0] if ranking else None,'ranking':ranking,'all_candidates':pool,
            'status':ranking[0]['qualification'] if ranking else 'NO_QUALIFIED_PRODUCER_ON_MODEL_VAL',
            'selection_role':'model_val','zero_update_in_producer_pool':False}


def standalone(rows):
    def compare(a,b):
        if abs(a['endpoint_psnr']-b['endpoint_psnr']) > 1e-8:return -1 if a['endpoint_psnr'] > b['endpoint_psnr'] else 1
        return -1 if (a['training_step'],a['checkpoint_id']) < (b['training_step'],b['checkpoint_id']) else 1
    return sorted(rows,key=cmp_to_key(compare))[0]


def grid(method):
    if method.startswith('G'):
        return [{'return_base':True}]+[{'temperature':t,'margin':m} for m in [.5,.25,0] for t in [1,2,.5]]
    if method.startswith('B'):
        return [{'return_base':True,'alpha':0}]+[{'alpha':a} for a in [.25,.5,.75,1]]
    return [{'return_base':True}]+[{'tau':t,'lamb':l} for t in [1e-4,1e-5,0] for l in [1e-3,1e-4,1e-6]]


def choose_grid(rows):
    """Rows include step and fixed strategy_index; selection freezes network only."""
    valid=[r for r in rows if r.get('return_base',False) or (r['ssim_delta'] >= -.001 and r['lpips_delta'] <= .002)]
    if not valid:raise ValueError('missing safe return_base policy')
    def compare(a,b):
        if abs(a['psnr']-b['psnr']) > 1e-8:return -1 if a['psnr']>b['psnr'] else 1
        ka=(not a.get('return_base',False),a['step'],a['strategy_index'])
        kb=(not b.get('return_base',False),b['step'],b['strategy_index'])
        return -1 if ka < kb else (1 if ka > kb else 0)
    return sorted(valid,key=cmp_to_key(compare))[0]


def rescue_choice(row):
    relative=(row['endpoint_mse']-row['baseline_mse'])/row['baseline_mse']
    delta=row['endpoint_psnr']-row['baseline_psnr']
    return {'route':'LOSS_COMPARISON' if relative <= -.005 and delta <= -.02 else 'LOW_LR_CONTINUATION',
            'relative_mse_change':relative,'delta_psnr':delta,'decision_role':'model_val','checkpoint_id':row['checkpoint_id']}
