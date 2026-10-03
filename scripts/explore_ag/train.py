"""Independent warm-start training with exact local resume and atomic checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.data.cache import static_cached
from mpa_diff.data.manifest import read_manifest
from mpa_diff.engine.checkpoint import rng_state, restore_rng, seed_all, source_hash
from mpa_diff.engine.losses import masked_mse, VGG19Loss
from mpa_diff.priors.kernels import pad_image
from mpa_diff.utils.io import read_image, sha256, write_json
from scripts.explore_ag.branches import ExplorationModel, BRANCHES


def implementation_hash():
    paths = [Path(__file__), Path(__file__).with_name('branches.py')]
    return hashlib.sha256(''.join(sha256(p) for p in paths).encode()).hexdigest()


def parameter_groups(model):
    groups = {'inherited': [], 'new': [], 'physical': []}
    counts = {key: 0 for key in groups}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if model.branch.startswith('A_') and name.startswith('base.priors.beta.'):
            group = 'physical'
        elif name.startswith(('shallow.', 'injected.controls.')):
            group = 'new'
        else:
            group = 'inherited'
        groups[group].append(param)
        counts[group] += param.numel()
    return groups, counts


def make_optimizer(model):
    params, counts = parameter_groups(model)
    rates = {'inherited': 1e-5, 'new': 1e-4, 'physical': 1e-6}
    active = [(key, params[key]) for key in ('inherited','new','physical') if params[key]]
    optimizer = torch.optim.Adam([{'params': values, 'lr': rates[key], 'name': key} for key, values in active],
                                 betas=(.9,.999), weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - .9*min(step,10000)/10000)
    return optimizer, scheduler, counts


def save(path, model, optimizer, scheduler, parent_path, parent_hash, branch, step, order, cursor, best, config):
    state = {'protocol_id': 'explore_ag_single_seed_v2_20261003', 'branch': branch,
             'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'step': step,
             'data_sampler': {'order': order, 'cursor': cursor}, 'best_val_psnr': best,
             'parent_checkpoint': str(parent_path.resolve()), 'parent_checkpoint_sha256': parent_hash,
             'parent_training_steps': config['parent_training_steps'],
             'config': model.config, 'implementation_sha256': implementation_hash(),
             'source_code_sha256': source_hash(),
             'manifest_sha256': sha256(model.config['data']['manifest']),
             'migration': model.migration, 'parameter_groups': config['parameter_groups']}
    path = Path(path)
    temp = Path(str(path) + '.tmp')
    torch.save(state, temp)
    temp.replace(path)


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    parent_path = Path(args.parent)
    parent_hash = sha256(parent_path)
    parent = torch.load(parent_path, map_location='cpu')
    if args.branch not in BRANCHES:
        raise ValueError('Unimplemented branch')
    seed_all(parent['config']['experiment']['seed'], parent['config']['runtime']['threads'])
    model = ExplorationModel(parent, args.branch).to(args.device)
    cfg = model.config
    perceptual = VGG19Loss(cfg['loss']['vgg_checkpoint'], cfg['loss']['vgg_sha256']).to(args.device)
    optimizer, scheduler, counts = make_optimizer(model)
    train_rows = [r for r in read_manifest(cfg['data']['manifest']) if r['split'] == 'train']
    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0
    step = 0
    best = float('-inf')
    last = out/'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu')
        for key, value in [('parent_checkpoint_sha256', parent_hash), ('branch', args.branch),
                           ('implementation_sha256', implementation_hash()),
                           ('source_code_sha256', source_hash()),
                           ('manifest_sha256', sha256(cfg['data']['manifest']))]:
            if state[key] != value:
                raise ValueError('Resume contract changed: ' + key)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step = state['step']
        order = state['data_sampler']['order']
        cursor = state['data_sampler']['cursor']
        best = state['best_val_psnr']
        restore_rng(state['rng'])
    run_config = {'parent_checkpoint': str(parent_path.resolve()),
                  'parent_checkpoint_sha256': parent_hash, 'parent_training_steps': parent['step'],
                  'branch': args.branch, 'max_added_steps': 10000, 'target_steps': args.target_steps,
                  'train_manifest_sha256': sha256(cfg['data']['manifest']), 'parameter_groups': counts,
                  'migration': model.migration, 'micro_batch': args.micro_batch,
                  'source_code_sha256': source_hash(),
                  'gradient_accumulation': 4//args.micro_batch,
                  'implementation_sha256': implementation_hash()}
    existing = out/'run_config.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        for key in ('parent_checkpoint_sha256','branch','train_manifest_sha256','micro_batch','implementation_sha256'):
            if old[key] != run_config[key]:
                raise ValueError('Run configuration changed: ' + key)
    write_json(existing, run_config)
    if args.target_steps > 10000 or 4 % args.micro_batch:
        raise ValueError('Target or global batch invalid')
    if step == 0 and not last.exists():
        save(last, model, optimizer, scheduler, parent_path, parent_hash,
             args.branch, step, order, cursor, best, run_config)
        os.link(last, out/'step_00000.pt')
    def choose():
        nonlocal cursor
        selected = []
        for _ in range(args.micro_batch):
            if cursor == len(order):
                random.shuffle(order)
                cursor = 0
            selected.append(train_rows[order[cursor]])
            cursor += 1
        return selected
    status = 'training'
    while step < args.target_steps:
        if args.deadline_unix and time.time() >= args.deadline_unix - 900:
            status = 'budget_stopped'
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        begin = time.perf_counter()
        metrics = {'loss': 0., 'pixel_mse': 0., 'cycle_mse': 0.}
        for _ in range(4//args.micro_batch):
            batch = choose()
            x = torch.stack([read_image(Path(cfg['data']['data_root'])/r['image_path'], cfg['data']['resize_hw']) for r in batch]).to(args.device)
            y = torch.stack([read_image(Path(cfg['data']['data_root'])/r['reference_path'], cfg['data']['resize_hw']) for r in batch]).to(args.device)
            padded, _ = pad_image(x)
            yp, _ = pad_image(y)
            static = static_cached(model.base.priors, padded, cfg)
            index = torch.randint(model.schedule.steps, (x.shape[0],), device=args.device)
            noise = torch.randn_like(yp)
            state = model.schedule.q_sample(yp, index, noise)
            condition, prior = model.condition(padded, static)
            prediction = model.predict(state, index, condition, prior)
            cropped = prediction[..., :y.shape[-2], :y.shape[-1]]
            pixel = masked_mse(cropped, y)
            cycle = model.cycle_loss(prediction, padded, prior) if args.branch.startswith('A_') else pixel.new_zeros(())
            loss = pixel + .1*perceptual(cropped, y) + .05*cycle
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss at added step {}'.format(step))
            (loss/(4//args.micro_batch)).backward()
            metrics['loss'] += loss.item()/(4//args.micro_batch)
            metrics['pixel_mse'] += pixel.item()/(4//args.micro_batch)
            metrics['cycle_mse'] += cycle.item()/(4//args.micro_batch)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradient at added step {}'.format(step))
        optimizer.step()
        scheduler.step()
        step += 1
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        record = {'added_step': step, **metrics, 'seconds': time.perf_counter()-begin,
                  'lr': {g['name']: g['lr'] for g in optimizer.param_groups}}
        with (out/'train.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        if step % 25 == 0:
            write_json(out/'progress.json', {'status': 'training', 'added_step': step,
                                             'target_steps': args.target_steps, 'last': record})
            print(json.dumps(record), flush=True)
        if step % 1000 == 0:
            write_json(out/'progress.json', {'status': 'checkpointing', 'added_step': step})
            save(last, model, optimizer, scheduler, parent_path, parent_hash,
                 args.branch, step, order, cursor, best, run_config)
            if step in (2000,5000,10000):
                numbered = out/('step_{:05d}.pt'.format(step))
                if not numbered.exists():
                    os.link(last, numbered)
    if step and (status == 'budget_stopped' or step % 1000):
        write_json(out/'progress.json', {'status': 'checkpointing', 'added_step': step})
        save(last, model, optimizer, scheduler, parent_path, parent_hash,
             args.branch, step, order, cursor, best, run_config)
    write_json(out/'progress.json', {'status': status if status != 'training' else 'milestone_complete',
                                     'added_step': step, 'target_steps': args.target_steps})
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', required=True)
    p.add_argument('--branch', required=True, choices=BRANCHES)
    p.add_argument('--output', required=True)
    p.add_argument('--target-steps', type=int, required=True)
    p.add_argument('--micro-batch', type=int, default=1)
    p.add_argument('--deadline-unix', type=float)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    print(json.dumps({'added_steps': train(args)}), flush=True)


if __name__ == '__main__':
    main()
