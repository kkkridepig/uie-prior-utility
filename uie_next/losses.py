import torch
import torch.nn.functional as F


def huber(x, delta=1.0):
    return F.huber_loss(x, torch.zeros_like(x), delta=delta, reduction='none')


def utility_loss(pred, target, scales, eta_pair=.25, eta_dec=.1):
    if target['v'].ndim!=5 or target['v'].shape[1]!=2:
        raise ValueError('expected [source,2,channel,height,width] paired views')
    active = target['active'].float()
    sv = max(float(scales.get('s_v', 1e-3)), 1e-3)
    su = max(float(scales.get('s_U', 1e-4)), 1e-4)
    se2 = max(float(scales.get('s_e2', 1e-4)), 1e-4)
    proj = huber((pred['v_hat'] - target['v']) / sv) * active
    counts=active.flatten(2).sum(-1)
    per_view=proj.flatten(2).sum(-1)/counts.clamp_min(1.)
    valid=(counts>0).to(per_view.dtype)
    per_image=(per_view*valid).sum(1)/valid.sum(1).clamp_min(1.)
    image_valid=(valid.sum(1)>0).to(per_image.dtype)
    proj=(per_image*image_valid).sum()/image_valid.sum().clamp_min(1.)
    pair = (active[:,0].bool() | active[:,1].bool()).float()
    pair_pred = pred.get('U_hat', 2 * pred['b_hat'] - target['a'])
    pair_true = target.get('U', torch.zeros_like(pair_pred))
    diff = (pair_pred[:,0]-pair_pred[:,1])-(pair_true[:,0]-pair_true[:,1])
    pair_term = (huber(diff/su)*pair).flatten(1).sum(-1)/pair.flatten(1).sum(-1).clamp_min(1.)
    pair_valid=(pair.flatten(1).sum(-1)>0).to(pair_term.dtype)
    pair_term = (pair_term*pair_valid).sum()/pair_valid.sum().clamp_min(1.)
    dec = (pred['output'] - target['target']).square().flatten(1).mean(-1).mean()/se2
    raw_proj=huber(pred['v_hat']-target['v'])*active
    raw_per_view=raw_proj.flatten(2).sum(-1)/counts.clamp_min(1.)
    raw_per_image=(raw_per_view*valid).sum(1)/valid.sum(1).clamp_min(1.)
    raw_proj=(raw_per_image*image_valid).sum()/image_valid.sum().clamp_min(1.)
    raw_pair=(huber(diff)*pair).flatten(1).sum(-1)/pair.flatten(1).sum(-1).clamp_min(1.)
    raw_pair=(raw_pair*pair_valid).sum()/pair_valid.sum().clamp_min(1.)
    return proj + eta_pair * pair_term + eta_dec * dec, {'proj': proj.detach(), 'pair': pair_term.detach(), 'dec': dec.detach(),
                                                      'proj_raw':raw_proj.detach(),'pair_raw':raw_pair.detach(),'dec_raw':(dec*se2).detach(),
                                                      'pair_live':pair_term,'dec_live':dec}
