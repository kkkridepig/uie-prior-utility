"""Verify an uploaded SS-UIE checkpoint without training or reading references."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uie_next.backbones.ssuie import load_official, postprocess
from uie_next.budget import DeviceBudget
from uie_next.data.manifest import inference_input
from uie_next.records import RUN, read, sha, write


def maximum_error(first, second):
    return float((first - second).abs().max().item())


def verify(args):
    path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the previous receipt; use a new output path.')
    receipt = {
        'checkpoint': str(path),
        'sha256': sha(path),
        'size_bytes': path.stat().st_size,
        'source_status': 'pending_user_download_source',
        'formal_training_started': False,
        'reference_images_read': False,
        'sealed_eval_read': False,
        'python': platform.python_version(),
        'torch_version': torch.__version__,
        'torch_path': torch.__file__,
        'checks': {},
    }
    write(output, {**receipt, 'status': 'VERIFYING'})
    try:
        state = torch.load(str(path), map_location='cpu')
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError('Expected a flat tensor state dictionary.')
        receipt['state_dict_entries'] = len(state)
        receipt['uniform_module_prefix'] = all(k.startswith('module.') for k in state)
        if not all(torch.isfinite(v).all().item() for v in state.values()):
            raise ValueError('Nonfinite checkpoint tensor.')
        del state
        base, identity = load_official(ROOT, path, expected_sha=args.sha256, policy='clip01')
        receipt['identity'] = identity
        receipt['checks']['strict_load'] = True
        receipt['parameters'] = sum(p.numel() for p in base.model.parameters())
        before = {k: v.detach().cpu().clone() for k, v in base.model.state_dict().items()}
        batchnorms = [m for m in base.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        receipt['batchnorm_modules'] = len(batchnorms)
        rows = [json.loads(line) for line in (RUN / 'roles.jsonl').read_text().splitlines()]
        rows = sorted((r for r in rows if r['role'] == 'model_fit'), key=lambda r: r['sample_id'])[:2]
        if len(rows) != 2:
            raise ValueError('Two audited model_fit inputs required.')
        for row in rows:
            if sha(row['input_path']) != row['input_sha256']:
                raise ValueError('Audited input changed: ' + row['sample_id'])
        receipt['inputs'] = [{k: r[k] for k in ('sample_id', 'role', 'input_path', 'input_sha256')} for r in rows]
        if not torch.cuda.is_available():
            raise RuntimeError('Real accelerator required; CPU fallback forbidden.')
        receipt['device'] = torch.cuda.get_device_name(0)
        receipt['device_count_used'] = 1
        torch.set_num_threads(2)
        inputs_cpu = torch.stack([inference_input(r['input_path']) for r in rows])
        with DeviceBudget(RUN) as budget:
            budget.start('uploaded_SS_UIE_weight_integration_verification', 600)
            begin = time.monotonic()
            torch.cuda.reset_peak_memory_stats(0)
            base = base.to('cuda:0')
            inputs = inputs_cpu.to('cuda:0')
            base.train()
            receipt['checks']['backbone_remains_eval_after_wrapper_train'] = not any(m.training for m in base.model.modules())
            receipt['checks']['all_backbone_parameters_frozen'] = all(not p.requires_grad for p in base.parameters())
            with torch.no_grad():
                first = base.model(inputs[:1])
                second = base.model(inputs[1:])
                batch = base.model(inputs)
                captured = []
                handle = base.model.register_forward_hook(lambda module, data, out: captured.append(out.detach()))
                try:
                    wrapped = base(inputs[:1])
                finally:
                    handle.remove()
            if not all(torch.isfinite(t).all().item() for t in (first, second, batch, wrapped)):
                raise ValueError('Nonfinite real-device output.')
            receipt['output_shape'] = list(batch.shape)
            receipt['output_dtype'] = str(batch.dtype)
            receipt['raw_output_range'] = [float(batch.min()), float(batch.max())]
            receipt['checks']['output_shape_float32'] = batch.shape == inputs.shape and batch.dtype == torch.float32
            errors = {
                'bare_vs_wrapper_raw': maximum_error(first, captured[0]),
                'wrapper_vs_clip_of_raw': maximum_error(wrapped, captured[0].clamp(0, 1)),
                'batch1_vs_batch2_raw': maximum_error(torch.cat([first, second]), batch),
            }
            for policy in ('clip01', 'official_minmax_float'):
                separate = torch.cat([postprocess(first, policy), postprocess(second, policy)])
                errors['batch1_vs_batch2_' + policy] = maximum_error(separate, postprocess(batch, policy))
            receipt['maximum_absolute_errors'] = errors
            receipt['tolerance'] = 1e-5
            receipt['checks'].update({name: value <= 1e-5 for name, value in errors.items()})
            torch.cuda.synchronize()
            receipt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(0)
            receipt['integration_wall_seconds'] = time.monotonic() - begin
            receipt['timing_is_deployment_benchmark'] = False
            changed = [k for k, v in base.model.state_dict().items() if not torch.equal(before[k], v.detach().cpu())]
            receipt['changed_state_dict_entries'] = changed
            receipt['checks']['parameters_and_buffers_unchanged'] = not changed
            receipt['checks']['uploaded_file_hash_unchanged'] = sha(path) == receipt['sha256']
            budget.guard()
        receipt['budget_used_device_seconds_after'] = read(RUN / 'budget.json')['used_device_seconds']
        passed = all(receipt['checks'].values())
        receipt['passed'] = passed
        receipt['status'] = 'STRICT_MATCH_GPU_PASS_SOURCE_PENDING' if passed else 'INTEGRATION_CHECK_FAILED'
        write(output, receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if passed else 2
    except Exception as exc:
        receipt.update(passed=False, status='VERIFICATION_FAILED', error=type(exc).__name__ + ': ' + str(exc))
        write(output, receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(verify(parser.parse_args()))
