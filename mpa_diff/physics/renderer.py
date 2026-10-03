import torch


def render(clean, distance, kappa_d, kappa_b, ambient):
    t = torch.exp(-kappa_d * distance)
    b = ambient * (-torch.expm1(-kappa_b * distance))
    return clean*t+b, {'t_D':t, 'b':b, 'A':ambient, 'kappa_D':kappa_d, 'kappa_B':kappa_b}


def invert(image, fields, floor=1e-6):
    t=fields['t_D']
    raw=(image-fields['b'])/t.clamp_min(floor)
    return raw.clamp(0,1), {'unclipped':raw,'clipped_mask':(raw<0)|(raw>1),'valid_mask':t>=floor}


def srgb_to_linear(x):
    return torch.where(x<=.04045,x/12.92,((x+.055)/1.055).clamp_min(0).pow(2.4))


def linear_to_srgb(x):
    return torch.where(x<=.0031308,12.92*x,1.055*x.clamp_min(1e-12).pow(1/2.4)-.055)
