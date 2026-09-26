import torch
import pytest

from uie.engine import configuration, train
from uie.runtime import load_checkpoint
from uie.data import paired_entries, write_manifest
from test_data import make_images


def fixture_manifest(tmp_path):
    make_images(tmp_path, count=12)
    entries = paired_entries(tmp_path / "input", tmp_path / "GT")
    for i, row in enumerate(entries):
        row["split"] = "train" if i < 6 else ("val" if i < 8 else ("calibration" if i < 10 else "test"))
    path = tmp_path / "manifest.json"
    write_manifest(path, entries, tmp_path)
    return path


def test_resume_matches_uninterrupted_cpu(tmp_path):
    manifest = fixture_manifest(tmp_path)
    common = dict(manifest=str(manifest), width=8, size=16, workers=0, batch_size=2,
                  cpu_threads=1, steps=4, schedule_steps=4, warmup=0, validate_every=2,
                  save_every=1, log_every=4, device="cpu", seed=7)
    full = train(configuration(overrides=dict(common, output=str(tmp_path / "full"))))
    partial = train(configuration(overrides=dict(common, steps=2, output=str(tmp_path / "resumed"))))
    resumed = train(configuration(overrides=dict(common, resume=str(partial), output=str(tmp_path / "resumed"))))
    a, b = load_checkpoint(full), load_checkpoint(resumed)
    assert a["step"] == b["step"] == 4
    assert a["data_state"] == b["data_state"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)


@pytest.mark.parametrize("sampler", ["flow", "ddim"])
def test_training_freezes_teacher_and_flow_during_utility(tmp_path, sampler):
    manifest = fixture_manifest(tmp_path)
    common = dict(manifest=str(manifest), width=8, size=16, workers=0, batch_size=2,
                  cpu_threads=1, steps=1, schedule_steps=2, warmup=0, validate_every=1,
                  save_every=1, log_every=1, device="cpu", sampler=sampler, nfe=2)
    base = train(configuration(overrides=dict(common, stage="baseline", output=str(tmp_path / "base"))))
    flow = train(configuration(overrides=dict(common, stage="flow", init=str(base), output=str(tmp_path / "flow"))))
    utility = train(configuration(overrides=dict(common, stage="utility", init=str(flow), output=str(tmp_path / "utility"))))
    a, b, c = [load_checkpoint(p)["model"] for p in (base, flow, utility)]
    for k in a:
        if k.startswith("coarse."):
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
        if not k.startswith("utility."):
            torch.testing.assert_close(b[k], c[k], rtol=0, atol=0)
    assert any(not torch.equal(b[k], c[k]) for k in b if k.startswith("utility."))
