import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_image(path, size=None):
    with Image.open(path) as im:
        im = im.convert('RGB')
        if size:
            im = im.resize((size[1], size[0]), Image.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255


def save_image(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = tensor.detach().cpu().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(a).save(path)


def verified_weight(path, expected):
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Explicit pretrained weight path required: %r; no random fallback' % path)
    if not expected or sha256(path) != expected:
        raise ValueError('Weight SHA256 missing or mismatched: ' + str(path))
    return Path(path)
