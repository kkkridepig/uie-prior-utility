import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_signature(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def scan_images(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.relative_to(root).with_suffix("").as_posix()
            if key in result:
                raise ValueError(f"Ambiguous image stem: {key}")
            result[key] = path
    if not result:
        raise ValueError(f"No images found in {root}")
    return result


def paired_entries(input_dir, target_dir=None, split="test", prefix="", groups=None):
    inputs = scan_images(input_dir)
    targets = scan_images(target_dir) if target_dir else {}
    if target_dir and inputs.keys() != targets.keys():
        missing = sorted(inputs.keys() - targets.keys())[:5]
        extra = sorted(targets.keys() - inputs.keys())[:5]
        raise ValueError(f"Image pair mismatch: missing references={missing}, extra references={extra}")
    entries = []
    for key, path in inputs.items():
        entry = {
            "id": prefix + key, "input": str(path), "split": split,
            "input_sha256": sha256_file(path), "input_pixels": image_signature(path),
        }
        if target_dir:
            entry.update(target=str(targets[key]), target_sha256=sha256_file(targets[key]),
                         target_pixels=image_signature(targets[key]))
        if groups is not None:
            if key not in groups:
                raise ValueError(f"Missing scene/group identifier: {key}")
            entry["group"] = str(groups[key])
        entries.append(entry)
    return entries


def assign_splits(entries, val_fraction, test_fraction, seed):
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("Invalid split fractions")
    # Union identical pixels (also across raw/reference roles) and supplied scene IDs.
    parent = list(range(len(entries)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, entry in enumerate(entries):
        signatures = [entry["input_pixels"], entry.get("target_pixels")]
        if "group" in entry:
            signatures.append("group:" + entry["group"])
        for signature in filter(None, signatures):
            if signature in seen:
                parent[find(i)] = find(seen[signature])
            seen[signature] = i
    buckets = {}
    for i, entry in enumerate(entries):
        buckets.setdefault(find(i), []).append(entry)
    keys = sorted(buckets)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_val = max(1, round(n * val_fraction)) if val_fraction else 0
    n_test = max(1, round(n * test_fraction)) if test_fraction else 0
    if n_val + n_test >= n:
        raise ValueError("Too few independent groups for requested split")
    for j, key in enumerate(keys):
        split = "test" if j < n_test else ("val" if j < n_test + n_val else "train")
        for entry in buckets[key]:
            entry["split"] = split
    return entries


def create_manifest(input_dir, target_dir=None, val_fraction=0.1, test_fraction=0.1, seed=42, groups=None):
    entries = paired_entries(input_dir, target_dir, groups=groups)
    if target_dir is None:
        return entries
    return assign_splits(entries, val_fraction, test_fraction, seed)


def write_manifest(path, entries, root, metadata=None):
    path = Path(path).resolve()
    root = Path(root).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    portable = []
    for item in entries:
        row = dict(item)
        for key in ("input", "target", "depth"):
            if key in row:
                row[key] = Path(row[key]).resolve().relative_to(root).as_posix()
        portable.append(row)
    payload = {"version": 1, "root": Path(os.path.relpath(root, path.parent)).as_posix(),
               "metadata": metadata or {}, "entries": portable}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = (path.parent / payload["root"]).resolve()
    return payload, root


def audit_manifest(path, verify_hashes=True):
    payload, root = read_manifest(path)
    counts, seen_pixels, seen_groups, seen_ids = {}, {}, {}, set()
    for row in payload["entries"]:
        split = row["split"]
        if split not in ("train", "val", "calibration", "test"):
            raise ValueError(f"Invalid split: {split}")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        seen_ids.add(row["id"])
        counts[split] = counts.get(split, 0) + 1
        if "group" in row:
            old = seen_groups.setdefault(row["group"], split)
            if old != split:
                raise ValueError(f"Scene leakage across {old}/{split}: {row['group']}")
        for key in ("input", "target"):
            if key not in row:
                continue
            file = (root / row[key]).resolve()
            if file != root and root not in file.parents:
                raise ValueError(f"Manifest path escapes data root: {row[key]}")
            if not file.is_file():
                raise ValueError(f"Missing image: {file}")
            if verify_hashes and sha256_file(file) != row[key + "_sha256"]:
                raise ValueError(f"Changed data hash: {file}")
            signature = row[key + "_pixels"]
            old = seen_pixels.setdefault(signature, split)
            if old != split:
                raise ValueError(f"Pixel leakage across {old}/{split}: {row['id']}")
        if "depth" in row:
            depth_path = (root / row["depth"]).resolve()
            if (depth_path != root and root not in depth_path.parents) or not depth_path.is_file():
                raise ValueError(f"Missing or unsafe depth map: {row['depth']}")
            if verify_hashes and sha256_file(depth_path) != row.get("depth_sha256"):
                raise ValueError(f"Changed depth hash: {depth_path}")
            depth = np.load(depth_path, allow_pickle=False)
            with Image.open(root / row["input"]) as image:
                shape = (image.height, image.width)
            if depth.shape != shape or not np.isfinite(depth).all():
                raise ValueError(f"Invalid aligned depth map: {row['id']}")
    if not counts:
        raise ValueError("Empty manifest")
    return {"counts": counts, "sha256": sha256_file(path), "root": str(root),
            "note": "Exact pixel and optional scene checks; near-duplicates require a separate audit."}


def audit_cross_manifests(source, target):
    source_data, _ = read_manifest(source)
    target_data, _ = read_manifest(target)
    used = set()
    for e in source_data["entries"]:
        if e["split"] in ("train", "val", "calibration"):
            used.update(filter(None, (e["input_pixels"], e.get("target_pixels"))))
    matches = [e["id"] for e in target_data["entries"] if e["split"] == "test"
               and any(s in used for s in (e["input_pixels"], e.get("target_pixels")) if s)]
    if matches:
        raise ValueError(f"Cross-dataset leakage: {len(matches)} test images, e.g. {matches[:5]}")
    return {"exact_overlap": 0, "near_duplicates_checked": False}


class ImageDataset(Dataset):
    def __init__(self, manifest, split, size=256, training=False, seed=42):
        self.payload, self.root = read_manifest(manifest)
        self.entries = [e for e in self.payload["entries"] if e["split"] == split]
        if not self.entries:
            raise ValueError(f"No {split} samples in {manifest}")
        self.size, self.training, self.seed, self.epoch = size, training, seed, 0
        if training and size < 16:
            raise ValueError("Training patch size must be >=16")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        row = self.entries[index]
        with Image.open(self.root / row["input"]) as f:
            image = f.convert("RGB")
        target = None
        if "target" in row:
            with Image.open(self.root / row["target"]) as f:
                target = f.convert("RGB")
            if target.size != image.size:
                if not self.payload["metadata"].get("allow_target_resize", False):
                    raise ValueError(f"Unaligned pair {row['id']}; explicitly enable target resizing for super-resolution datasets")
                target = target.resize(image.size, Image.Resampling.BICUBIC)
        depth = None
        if "depth" in row:
            depth_array = np.load(self.root / row["depth"], allow_pickle=False).astype(np.float32)
            if depth_array.shape != (image.height, image.width) or not np.isfinite(depth_array).all():
                raise ValueError(f"Invalid aligned depth map: {row['id']}")
            depth = Image.fromarray(depth_array)
        items = [image, target, depth]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        if self.size:
            if self.training:
                w, h = image.size
                ratio = max(1.0, self.size / min(w, h))
                new_size = (max(self.size, round(w * ratio)), max(self.size, round(h * ratio)))
                items = [v.resize(new_size, Image.Resampling.BILINEAR) if v is not None else None for v in items]
                left = rng.randrange(new_size[0] - self.size + 1)
                top = rng.randrange(new_size[1] - self.size + 1)
                box = (left, top, left + self.size, top + self.size)
                items = [v.crop(box) if v is not None else None for v in items]
                for transform in (Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE):
                    if rng.random() < 0.5:
                        items = [v.transpose(transform) if v is not None else None for v in items]
            else:
                items = [v.resize((self.size, self.size), Image.Resampling.BILINEAR) if v is not None else None for v in items]
        sample = {"id": row["id"]}
        for key, im in zip(("input", "target", "depth"), items):
            if im is not None:
                arr = np.array(im, dtype=np.float32, copy=True)
                if key == "depth":
                    sample[key] = torch.from_numpy(arr).unsqueeze(0)
                else:
                    sample[key] = torch.from_numpy(arr / 255).permute(2, 0, 1)
        return sample
