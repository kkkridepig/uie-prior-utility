import json
from pathlib import Path

from .data import (create_manifest, paired_entries, assign_splits, write_manifest,
                   audit_manifest, sha256_file)


def prepare(args):
    metadata = {"dataset": args.dataset, "seed": args.seed,
                "allow_target_resize": args.allow_target_resize}
    groups = json.loads(Path(args.groups).read_text(encoding="utf-8")) if args.groups else None
    if args.layout == "wwe":
        root = Path(args.root).resolve()
        candidates = [p for p in (root, *root.rglob(args.dataset)) if p.name == args.dataset and (p / "train").is_dir()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {args.dataset}/train directory under {root}; found {candidates}")
        dataset = candidates[0]
        train = paired_entries(dataset / "train/input", dataset / "train/GT", "train", "train/", groups)
        test = paired_entries(dataset / "test/input", dataset / "test/GT", "test", "test/", groups)
        if (dataset / "val/input").is_dir() and not args.validation_from_train:
            val = paired_entries(dataset / "val/input", dataset / "val/GT", "val", "val/", groups)
        else:
            train = assign_splits(train, args.val_fraction, 0, args.seed)
            val = [e for e in train if e["split"] == "val"]
            train = [e for e in train if e["split"] == "train"]
        entries = train + val + test
        metadata["protocol"] = "WWE author test split preserved; validation source recorded below; not a reproduction of original training recipe"
        metadata["validation_from_train"] = args.validation_from_train or not (dataset / "val/input").is_dir()
    else:
        if not args.input:
            raise ValueError("--input is required for pairs/nonref layouts")
        if args.layout == "pairs" and not args.target:
            raise ValueError("--target is required for the pairs layout")
        root = Path(args.root).resolve()
        entries = create_manifest(args.input, args.target if args.layout == "pairs" else None,
                                  args.val_fraction, args.test_fraction, args.seed, groups)
        metadata["protocol"] = "Custom seeded group split; NOT the MPA-Diff U97 or standard U90 split" if args.layout == "pairs" else "Unpaired test only"
    if args.calibration_fraction and args.layout != "nonref":
        train = [e for e in entries if e["split"] == "train"]
        assign_splits(train, args.calibration_fraction, 0, args.seed+1)
        for e in train:
            if e["split"] == "val":
                e["split"] = "calibration"
        metadata["calibration_fraction_of_remaining_train"] = args.calibration_fraction
    if args.depth:
        depth_root = Path(args.depth).resolve()
        for e in entries:
            # Optional maps mirror manifest IDs, including train/test prefixes.
            path = depth_root / (e["id"] + ".npy")
            if not path.exists():
                raise ValueError(f"Missing precomputed depth map: {path}")
            e["depth"] = str(path)
            e["depth_sha256"] = sha256_file(path)
        metadata["depth_convention"] = "Farther=larger; aligned float32 NPY; per-image relative normalization"
    if Path(args.output).exists():
        raise ValueError("Manifest exists; choose a new output to preserve experimental splits")
    write_manifest(args.output, entries, root, metadata)
    report = audit_manifest(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
