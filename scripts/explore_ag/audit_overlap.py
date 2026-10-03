"""Read-only image identity audit for frozen paired dataset splits."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mpa_diff.data.manifest import read_manifest
from mpa_diff.utils.io import sha256, write_json


ROOT = Path(__file__).resolve().parents[2]


def dhash(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert('L').resize((9, 8), Image.BILINEAR))
    bits = pixels[:, 1:] > pixels[:, :-1]
    return sum(int(value) << index for index, value in enumerate(bits.flat))


def main():
    source_manifest = ROOT/'manifests/uieb_recon_grouped_v1.jsonl'
    target_manifest = ROOT/'manifests/lsui_recon_grouped_v1.jsonl'
    source = read_manifest(source_manifest)
    target = read_manifest(target_manifest)
    collections = {
        'uieb_train_val': (ROOT/'data/uieb', [r for r in source if r['split'] in ('train', 'val')]),
        'uieb_test': (ROOT/'data/uieb', [r for r in source if r['split'] == 'test']),
        'lsui_test': (ROOT/'data/lsui', [r for r in target if r['split'] == 'test']),
    }
    indexed = {}
    for name, (root, rows) in collections.items():
        indexed[name] = [(row['sample_id'], row['image_sha256'], dhash(root/row['image_path']))
                         for row in rows]
    comparisons = {}
    for left, right in (('uieb_train_val', 'uieb_test'),
                        ('uieb_train_val', 'lsui_test')):
        exact, similar = [], []
        for a, b in itertools.product(indexed[left], indexed[right]):
            if a[1] == b[1]:
                exact.append([a[0], b[0]])
            distance = bin(a[2] ^ b[2]).count('1')
            if distance <= 4:
                similar.append({'source': a[0], 'target': b[0], 'dhash_distance': distance})
        comparisons[left+'__'+right] = {
            'exact_byte_pairs': exact,
            'near_duplicate_candidates': sorted(similar, key=lambda row: row['dhash_distance']),
            'near_duplicate_rule': '64-bit dHash Hamming distance <= 4; candidate only, requires image review',
        }
    output = ROOT/'runs/explore_ag_single_seed_v2_20261003/data_overlap_audit.json'
    write_json(output, {'source_manifest_sha256': sha256(source_manifest),
                        'target_manifest_sha256': sha256(target_manifest),
                        'counts': {name: len(rows) for name, rows in indexed.items()},
                        'comparisons': comparisons})
    print(json.dumps({'output': str(output),
                      'candidate_counts': {name: len(value['near_duplicate_candidates'])
                                           for name, value in comparisons.items()}}))


if __name__ == '__main__':
    main()
