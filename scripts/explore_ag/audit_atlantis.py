"""Audit Atlantis v1 as grouped synthetic relative-depth training material."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from PIL import Image


DEPTH = re.compile(r'Atlantis/depth/(\d{4})\.png$')
UNDERWATER = re.compile(r'Atlantis/underwater/(\d{4})_(\d{2})\.png$')


def audit(archive):
    with ZipFile(archive) as zf:
        names = zf.namelist()
        depth = {}
        images = {}
        unexpected = []
        for name in names:
            match_depth = DEPTH.fullmatch(name)
            match_image = UNDERWATER.fullmatch(name)
            if match_depth:
                if match_depth[1] in depth:
                    raise ValueError('Duplicate depth ID: ' + match_depth[1])
                depth[match_depth[1]] = name
            elif match_image:
                key = (match_image[1], match_image[2])
                if key in images:
                    raise ValueError('Duplicate generated image ID: ' + str(key))
                images[key] = name
            else:
                unexpected.append(name)
        if unexpected or len(depth) != 400 or len(images) != 3200:
            raise ValueError('Unexpected Atlantis v1 archive layout: {}'.format(
                {'depth': len(depth), 'underwater': len(images), 'unexpected': unexpected[:8]}))
        expected = {(key, '{:02d}'.format(index)) for key in depth for index in range(8)}
        if set(images) != expected:
            raise ValueError('Missing/misnamed underwater variants: {}'.format(len(expected-set(images))))
        ids = sorted(depth, key=lambda key: hashlib.sha256(('atlantis_v1_group:'+key).encode()).digest())
        split_of = {key: ('train' if index < 280 else 'val' if index < 340 else 'test')
                    for index, key in enumerate(ids)}
        sizes = {}
        dynamic = []
        depth_min, depth_max = 255, 0
        records = []
        for key in sorted(depth):
            with Image.open(BytesIO(zf.read(depth[key]))) as source:
                array = np.asarray(source)
                if array.ndim != 2 or array.dtype != np.uint8:
                    raise ValueError('Depth target is not 8-bit grayscale: ' + depth[key])
                size = source.size
            lo, hi = int(array.min()), int(array.max())
            depth_min = min(depth_min, lo)
            depth_max = max(depth_max, hi)
            dynamic.append(hi-lo)
            for variant in range(8):
                name = images[(key, '{:02d}'.format(variant))]
                with Image.open(BytesIO(zf.read(name))) as image:
                    if image.mode != 'RGB' or image.size != size:
                        raise ValueError('Image/depth mismatch: {} vs {}'.format(name, depth[key]))
                sizes[str(size)] = sizes.get(str(size), 0) + 1
                records.append({'sample_id': 'Atlantis/'+key+'_{:02d}'.format(variant),
                                'scene_id': 'Atlantis/'+key, 'split': split_of[key],
                                'image_zip_member': name, 'depth_zip_member': depth[key],
                                'image_size_wh': list(size), 'depth_kind': 'synthetic_relative_8bit',
                                'depth_units': 'unitless', 'valid_mask_source': None,
                                'source_archive_sha256': None})
    return records, {'archive_entries': len(names), 'depth_groups': len(depth),
                     'underwater_images': len(images), 'group_split':
                         {split: sum(value == split for value in split_of.values())
                          for split in ('train', 'val', 'test')},
                     'image_split': {split: sum(row['split'] == split for row in records)
                                     for split in ('train', 'val', 'test')},
                     'size_counts': sizes, 'depth_min': depth_min, 'depth_max': depth_max,
                     'constant_depth_groups': sum(value == 0 for value in dynamic),
                     'depth_dynamic_range_median': float(np.median(dynamic)),
                     'depth_orientation': 'unverified',
                     'limitation': 'Synthetic terrestrial depth, no independent valid mask or metric calibration'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    archive = Path(args.archive)
    records, receipt = audit(archive)
    sha = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            sha.update(chunk)
    digest = sha.hexdigest()
    for row in records:
        row['source_archive_sha256'] = digest
    receipt.update({'archive': str(archive.resolve()), 'sha256': digest,
                    'source': 'https://www.kaggle.com/datasets/zkawfanx/atlantis',
                    'kaggle_version': 1, 'license': 'CC BY-NC-SA 4.0',
                    'split_rule': 'sort SHA256(atlantis_v1_group:base_id), then 280/60/60 base groups',
                    'train_test_overlap_by_base_id': False,
                    'scientific_status': 'candidate_only_not_depth_ground_truth_for_real_underwater'})
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in records))
    temporary.replace(manifest)
    output = Path(args.receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    temporary.replace(output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
