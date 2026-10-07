"""CPU-only structural convolution MAC accounting, separate from latency.

Counts complete external modules at batch=1, RGB 256x256. The public backbone's
custom scan and pointwise operations are intentionally not guessed. No weights,
dataset image, quality score, checkpoint choice or GPU execution is involved.
"""
import time

import torch.nn as nn

from uie_next.records import ROOT, sha, write, append
from uie_next.models.candidate import Candidate
from uie_next.models.controls import Controller
from uie_next.v2.context import RUN_ID, METHOD_ORDER
from uie_next.v2.diagnostics import csv_write


def main():
    started = time.monotonic()
    run = ROOT / 'runs' / RUN_ID
    append(run / 'commands.jsonl', {'argv': ['-m', 'scripts.ssuie_v2_compute_costs'],
                                   'purpose': 'CPU-only structural cost accounting'})
    expected = {'B1': 115907, 'B3': 115907, 'B4': 115907,
                'G0': 10, 'G1': 76897, 'G2': 150256,
                'F0': 77475, 'R0': 76611, 'O': 76896,
                'O-NI': 76896, 'O-NP': 76896, 'O-NS': 77185,
                'O-ND': 76896}
    rows = []
    for name in ['B1', 'B3'] + METHOD_ORDER:
        model = Candidate() if name in ('B1', 'B3', 'B4') else Controller(name)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != expected[name]:
            raise ValueError('Protocol parameter mismatch: ' + name)
        repeated_calls = 2 if name in ('O', 'O-NI', 'O-NP', 'O-ND') else 1
        macs = 0
        for module in model.modules():
            if not isinstance(module, nn.Conv2d):
                continue
            # Every registered external Conv2d preserves the 256x256 grid.
            if module.stride != (1, 1) or module.dilation != (1, 1):
                raise ValueError('Unexpected convolution geometry')
            kh, kw = module.kernel_size
            if module.padding != (kh // 2, kw // 2):
                raise ValueError('Unexpected spatial shape change')
            macs += 256 * 256 * module.out_channels * (module.in_channels // module.groups) * kh * kw
        rows.append({'method': name, 'parameters': parameters,
                     'module_convolution_MACs': macs * repeated_calls,
                     'shared_network_forward_calls': repeated_calls,
                     'batch': 1, 'height': 256, 'width': 256,
                     'scope': 'external_module_only_excludes_backbone_prior_pointwise_and_IO',
                     'MAC_definition': 'one_multiply_accumulate_is_one_MAC',
                     'measurement': 'exact_structural_accounting_not_runtime_profile'})
    csv_write(run / 'timing/module_compute_costs.csv', rows)
    write(run / 'timing/module_compute_costs.json', {
        'script_sha256': sha(__file__), 'rows': rows,
        'device_work': False, 'dataset_images_read': False,
        'weights_loaded': False, 'selection_changed': False,
        'backbone_MACs': None,
        'backbone_MACs_status': 'custom_scan_cost_not_measured_not_guessed',
        'CPU_wall_seconds': time.monotonic() - started})
    print('Recorded external module parameters and convolution MACs; no device work')


if __name__ == '__main__':
    main()
