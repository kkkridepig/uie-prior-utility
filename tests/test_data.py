import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from uie.data import create_manifest, audit_manifest, ImageDataset, write_manifest
from uie.download import safe_extract


def make_images(root, count=10):
    for folder in ("input", "GT"):
        (root / folder).mkdir(parents=True)
    for i in range(count):
        image = np.random.default_rng(i).integers(0, 255, (19, 27, 3), dtype=np.uint8)
        Image.fromarray(image).save(root / "input" / f"{i:02d}.png")
        Image.fromarray(255 - image).save(root / "GT" / f"{i:02d}.png")


def test_manifest_pairing_determinism_and_portability(tmp_path):
    make_images(tmp_path)
    a = create_manifest(tmp_path / "input", tmp_path / "GT", val_fraction=0.2, test_fraction=0.2)
    b = create_manifest(tmp_path / "input", tmp_path / "GT", val_fraction=0.2, test_fraction=0.2)
    assert a == b
    path = tmp_path / "manifest.json"
    write_manifest(path, a, tmp_path)
    report = audit_manifest(path)
    assert report["counts"] == {"train": 6, "val": 2, "test": 2}
    sample = ImageDataset(path, "train", size=16, training=True)[0]
    assert sample["input"].shape == (3, 16, 16)
    assert sample["target"].shape == sample["input"].shape
    assert json.loads(path.read_text())["root"] == "."


def test_missing_pair_is_not_silently_zipped(tmp_path):
    make_images(tmp_path)
    (tmp_path / "GT" / "03.png").unlink()
    with pytest.raises(ValueError, match="pair"):
        create_manifest(tmp_path / "input", tmp_path / "GT")


def test_duplicate_across_splits_rejected(tmp_path):
    make_images(tmp_path)
    entries = create_manifest(tmp_path / "input", tmp_path / "GT")
    entries[0]["split"] = "train"
    duplicate = dict(entries[0], split="test", id="duplicate")
    path = tmp_path / "manifest.json"
    write_manifest(path, entries + [duplicate], tmp_path)
    with pytest.raises(ValueError, match="leakage"):
        audit_manifest(path)


def test_archive_path_traversal_rejected(tmp_path):
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("../escaped.txt", "no")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_pairs_require_reference_directory(tmp_path):
    from uie.cli import main
    make_images(tmp_path)
    with pytest.raises(ValueError, match="--target"):
        main(["prepare", "--layout", "pairs", "--dataset", "fixture", "--root", str(tmp_path),
              "--input", str(tmp_path / "input"), "--output", str(tmp_path / "manifest.json")])


def test_depth_hash_and_synchronized_augmentation(tmp_path):
    from uie.cli import main
    make_images(tmp_path)
    (tmp_path / "depth").mkdir()
    for image_path in (tmp_path / "input").glob("*.png"):
        red = np.asarray(Image.open(image_path), dtype=np.float32)[..., 0] / 255
        np.save(tmp_path / "depth" / (image_path.stem + ".npy"), red)
    manifest = tmp_path / "manifest.json"
    main(["prepare", "--layout", "pairs", "--dataset", "fixture", "--root", str(tmp_path),
          "--input", str(tmp_path / "input"), "--target", str(tmp_path / "GT"),
          "--depth", str(tmp_path / "depth"), "--output", str(manifest)])
    sample = ImageDataset(manifest, "train", size=16, training=True)[0]
    np.testing.assert_allclose(sample["input"][0].numpy(), sample["depth"][0].numpy(), atol=1/255)
    np.save(tmp_path / "depth/00.npy", np.zeros((19, 27), dtype=np.float32))
    with pytest.raises(ValueError, match="Changed depth hash"):
        audit_manifest(manifest)
