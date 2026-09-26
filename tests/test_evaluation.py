import math

import pytest
import torch

from uie.data import sha256_file
from uie.engine import configuration
from uie.evaluate import evaluate
from uie.metrics import paired_metrics
from uie.models import RestorationSystem
from uie.runtime import atomic_checkpoint
from test_training import fixture_manifest


def test_known_metric_values():
    reference = torch.zeros(3, 16, 16)
    identity = paired_metrics(reference, reference)
    assert identity["ssim"] == pytest.approx(1)
    assert identity["mae"] == 0
    assert paired_metrics(torch.full_like(reference, 0.1), reference)["psnr"] == pytest.approx(20, abs=1e-5)


def test_untrained_gate_is_not_reported_as_calibrated(tmp_path):
    manifest = fixture_manifest(tmp_path)
    model = RestorationSystem(width=8)
    checkpoint = tmp_path / "flow.pt"
    atomic_checkpoint(checkpoint, {"model": model.state_dict(), "model_spec": model.spec,
                      "stage": "flow", "config": configuration(),
                      "manifest_sha256": sha256_file(manifest)})
    with pytest.raises(ValueError, match="no trained"):
        evaluate(checkpoint, manifest, tmp_path / "bad", size=16, gate_mode="learned")
    result = evaluate(checkpoint, manifest, tmp_path / "valid", size=0, steps=1,
                      diagnostics=True, warmup=0)
    assert result["count"] == 2
    assert "calibration" not in result
    assert "brier" not in result["means"]
    assert math.isfinite(result["means"]["off_psnr"])
