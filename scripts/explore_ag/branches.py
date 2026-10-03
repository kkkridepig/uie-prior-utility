"""Minimal A/B/C/D/F screening branches around the frozen RGB parent topology."""
import copy
import math
import types

import torch
from torch import nn
import torch.nn.functional as F

from mpa_diff.contracts import PriorBundle
from mpa_diff.models.unet import HINResidualBlock
from mpa_diff.physics.renderer import render, invert, srgb_to_linear, linear_to_srgb
from mpa_diff.priors.provider import MPADiff
from mpa_diff.priors.kernels import sobel


BRANCHES = ('BASE_CONT', 'A_COUPLED', 'A_SPLIT', 'B_PROXY', 'C_CONV',
            'C_FIXED', 'C_ROUTED', 'D_PHASE', 'D_SOBEL_CONTROL', 'F_UNIFORM', 'F_ROUTED')


def phase_spatial(image):
    z = torch.fft.fft2(image.float(), norm='ortho')
    magnitude = z.abs()
    unit = torch.where(magnitude > 1e-6, z / magnitude.clamp_min(1e-6), torch.zeros_like(z))
    return torch.fft.ifft2(unit, norm='ortho').real


class ShallowCondition(nn.Module):
    def __init__(self, kind, width):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, width, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image):
        feature = phase_spatial(image) if self.kind == 'phase' else sobel(image)
        return self.net(feature)


class ConvControl(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.down = nn.Conv2d(channels, hidden, 1)
        self.residual = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h, extras, time):
        return self.up(F.silu(self.residual(F.silu(self.down(h)))))


class PriorControl(nn.Module):
    def __init__(self, channels, routed):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in (1, 3, 32)])
        self.query = nn.Conv2d(channels, channels, 1)
        self.value = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(3)])
        self.gate = nn.Conv2d(channels * 2 + 1, 4, 1) if routed else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h, extras, time):
        size = h.shape[-2:]
        inputs = [extras['depth'], extras['histogram'], extras['highfreq']]
        encoded = [net(F.interpolate(x, size=size, mode='bilinear', align_corners=False))
                   for net, x in zip(self.encoders, inputs)]
        query = self.query(h)
        updates = [value(F.silu(enc)) * torch.sigmoid((query * enc).sum(1, keepdim=True)/math.sqrt(h.shape[1]))
                   for value, enc in zip(self.value, encoded)]
        depth_valid = F.interpolate(extras['depth_valid'].float(), size=size, mode='nearest') > .5
        histogram_valid = torch.full_like(depth_valid, not extras.get('histogram_missing', False))
        highfreq_valid = torch.full_like(depth_valid, not extras.get('highfreq_missing', False))
        mask = torch.cat((depth_valid, histogram_valid, highfreq_valid), 1)
        if self.training:
            mask = mask & (torch.rand((h.shape[0], 3, 1, 1), device=h.device) >= .1)
        if self.gate is None:
            weights = mask.float()
            weights = weights / (1 + weights.sum(1, keepdim=True))
        else:
            t = time.float().view(-1, 1, 1, 1).expand(-1, 1, *size) / 1000
            logits = self.gate(torch.cat((h, sum(encoded), t), 1))
            logits = torch.cat((logits[:, :3].masked_fill(~mask, -1e4), logits[:, 3:]), 1)
            weights = logits.softmax(1)[:, :3]
        result = sum(weights[:, i:i+1] * updates[i] for i in range(3))
        return self.out(result) * mask.any(1, keepdim=True)


class ExpertControl(nn.Module):
    def __init__(self, channels, routed):
        super().__init__()
        self.experts = nn.ModuleList([ConvControl(channels, max(8, channels//8)) for _ in range(4)])
        self.token = nn.Sequential(nn.Linear(3 + 3 + 1, 64), nn.SiLU(), nn.Linear(64, 64)) if routed else None
        self.gate = nn.Linear(64, 5) if routed else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, h, extras, time):
        image = extras['image']
        depth = extras['depth']
        hist = extras['histogram']
        if self.gate is None:
            weights = torch.full((h.shape[0], 5), .2, device=h.device, dtype=h.dtype)
        else:
            features = torch.cat((image.mean((-2,-1)),
                                  hist.mean((-2,-1)), depth.mean((-2,-1))), 1)
            weights = self.gate(self.token(features)).softmax(1)
        return sum(weights[:, i:i+1, None, None] * expert(h, extras, time)
                   for i, expert in enumerate(self.experts))


class InjectedDenoiser(nn.Module):
    def __init__(self, base, branch, seed):
        super().__init__()
        self.base = base
        self.branch = branch
        factory = (lambda c: PriorControl(c, branch == 'C_ROUTED')) if branch.startswith('C_') else (lambda c: ExpertControl(c, branch == 'F_ROUTED'))
        if branch == 'C_CONV':
            def factory(c):
                target = sum(p.numel() for p in PriorControl(c, True).parameters())
                hidden = min(range(1, 129), key=lambda n: abs(2*c*n + 9*n*n + c + 2*n - target))
                return ConvControl(c, hidden)
        controls = {}
        for name, channels, offset in (('quarter', 64, 701), ('eighth', 96, 702)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + offset)
                controls[name] = factory(channels)
        self.controls = nn.ModuleDict(controls)

    def forward(self, state, index, condition, extras):
        net = self.base
        freq = torch.exp(torch.arange(net.base//2, device=state.device)*(-math.log(10000)/(net.base//2-1)))
        phase = index.float()[:, None] * freq[None]
        time = net.time(torch.cat((phase.sin(), phase.cos()), 1))
        x = torch.cat((state, condition['image'], condition['physical'], condition['histogram']), 1)
        h = net.encoder.input(x) + condition['highfreq']
        skips = [h]
        for i, layer in enumerate(net.encoder.layers):
            h = layer(h, time) if isinstance(layer, HINResidualBlock) else layer(h)
            if i == 3:
                h = h + self.controls['quarter'](h, extras, index)
            if i == 5:
                h = h + self.controls['eighth'](h, extras, index)
            skips.append(h)
        h = net.middle(h)
        for layer in net.decoder:
            if isinstance(layer, HINResidualBlock):
                h = layer(torch.cat((h, skips.pop()), 1), time)
            else:
                h = layer(h)
        return net.output(F.silu(h))


class ExplorationModel(nn.Module):
    def __init__(self, parent_state, branch):
        super().__init__()
        if branch not in BRANCHES:
            raise ValueError('Unknown or unsupported screening branch: ' + branch)
        self.branch = branch
        config = copy.deepcopy(parent_state['config'])
        if branch == 'B_PROXY' or branch.startswith('A_'):
            config['depth']['physics_coordinate'] = 'distance_proxy'
        self.config = config
        self.base = MPADiff(config)
        self.base.load_state_dict(parent_state['model'], strict=True)
        self.migration = {'loaded_parent_tensors': len(parent_state['model']),
                          'missing_parent_tensors': [], 'shape_mismatches': [],
                          'semantic_changes': []}
        if branch.startswith('A_'):
            head = self.base.priors.beta.head
            old = head[2]
            if branch == 'A_SPLIT':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(config['experiment']['seed'] + 703)
                    split = nn.Linear(old.in_features, 6)
                with torch.no_grad():
                    split.weight.copy_(old.weight.repeat(2, 1))
                    split.bias.copy_(old.bias.repeat(2))
                head[2] = split
                self.migration['semantic_changes'].append('Beta head copied to independent kappa_D and kappa_B outputs')
            head[3] = nn.Softplus()
            def beta_forward(beta, x):
                h, _ = beta.encoder(x)
                h = beta.projection(beta.middle(h)).mean((2, 3))
                return beta.head(h).view(x.shape[0], -1, 1, 1)
            self.base.priors.beta.forward = types.MethodType(beta_forward, self.base.priors.beta)
            self.migration['semantic_changes'].append('sigmoid to softplus; linear RGB; relative distance; not legacy-equivalent')
        if branch.startswith('D_'):
            kind = 'phase' if branch == 'D_PHASE' else 'sobel'
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config['experiment']['seed'] + 704)
                self.shallow = ShallowCondition(kind, config['model']['base_channels'])
        else:
            self.shallow = None
        if branch.startswith(('C_', 'F_')):
            self.injected = InjectedDenoiser(self.base.denoiser, branch,
                                             config['experiment']['seed'])
        else:
            self.injected = None
        self.schedule = self.base.schedule
        self.migration['new_trainable_parameters'] = sum(p.numel() for name, p in self.named_parameters()
                                                          if p.requires_grad and (name.startswith('shallow.') or name.startswith('injected.controls.')))

    def condition(self, image, static=None):
        if self.branch.startswith('A_'):
            static = self.base.priors.static(image) if static is None else static
            k = self.base.priors.beta(image)
            kd, kb = (k.chunk(2, 1) if self.branch == 'A_SPLIT' else (k, k))
            linear_image = srgb_to_linear(image.clamp(0,1))
            ambient = srgb_to_linear(static['ambient'].clamp(0,1))
            _, fields = render(linear_image, static['depth'].distance_proxy, kd, kb, ambient)
            physical_linear, diagnostics = invert(linear_image, fields)
            fields.update(diagnostics)
            physical = linear_to_srgb(physical_linear).clamp(0,1)
            prior = PriorBundle(image, static['depth'], physical, static['histogram'],
                                static['edges'], static['wavelet'], fields, {})
            shape = image.shape[-2:]
            condition = {'image': image, 'physical': physical,
                         'histogram': F.interpolate(prior.histogram, size=shape, mode='bilinear', align_corners=False),
                         'highfreq': self.base.highfreq(torch.cat((F.interpolate(prior.wavelet_high, size=shape, mode='bilinear', align_corners=False), prior.edges), 1))}
        else:
            condition, prior = self.base.condition(image, static)
        if self.shallow is not None:
            condition['highfreq'] = condition['highfreq'] + self.shallow(image)
        return condition, prior

    def predict(self, state, index, condition, prior):
        if self.injected is None:
            return self.base.denoiser(state, index, condition)
        extras = {'image': condition['image'], 'depth': prior.depth.distance_proxy,
                  'depth_valid': prior.depth.valid_mask,
                  'histogram': condition['histogram'], 'highfreq': condition['highfreq'],
                  'histogram_missing': prior.metadata.get('histogram_missing', False),
                  'highfreq_missing': prior.metadata.get('highfreq_missing', False)}
        return self.injected(state, index, condition, extras)

    def cycle_loss(self, prediction, image, prior):
        fields = prior.fields
        _, output = render(srgb_to_linear(prediction.clamp(0,1)), prior.depth.distance_proxy,
                           fields['kappa_D'], fields['kappa_B'], fields['A'])
        reconstruction = srgb_to_linear(prediction.clamp(0,1))*output['t_D']+output['b']
        return (reconstruction-srgb_to_linear(image.clamp(0,1))).square().mean()
