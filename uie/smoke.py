from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from .data import paired_entries, write_manifest
from .engine import configuration, train
from .evaluate import evaluate, calibrate
from .runtime import atomic_json, environment_report


def smoke(output, device="cpu", precision="fp32", sampler="flow"):
    root = Path(output).resolve()
    if (root / "smoke_report.json").exists():
        raise ValueError("Use a fresh smoke output directory")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for split, count in [("train", 6), ("val", 2), ("calibration", 2), ("test", 2)]:
        for folder in ("input", "GT"):
            (root / "fixture" / split / folder).mkdir(parents=True, exist_ok=True)
        for i in range(count):
            rng = np.random.default_rng(sum(map(ord, split))*100 + i)
            yy, xx = np.mgrid[:24, :32]
            clean = np.clip(rng.uniform(0.1, 0.9, (1, 1, 3)) + 0.15*np.sin(xx[..., None]/4) +
                            0.1*np.cos(yy[..., None]/3), 0, 1)
            degraded = np.clip(clean * np.array([0.45, 0.7, 0.9]) + np.array([0.08, 0.13, 0.15]), 0, 1)
            for folder, array in (("input", degraded), ("GT", clean)):
                Image.fromarray(np.round(array*255).astype(np.uint8)).save(root / "fixture" / split / folder / f"{i}.png")
        entries.extend(paired_entries(root / "fixture" / split / "input", root / "fixture" / split / "GT",
                                      split, split+"/"))
    manifest = root / "manifest.json"
    write_manifest(manifest, entries, root, {"synthetic_smoke_only": True})
    common = dict(manifest=str(manifest), width=8, size=16, batch_size=2, workers=0,
                  steps=1, schedule_steps=4, warmup=0, validate_every=1, save_every=1,
                  log_every=1, nfe=2, device=device, precision=precision, cpu_threads=1, sampler=sampler)
    baseline = train(configuration(overrides=dict(common, stage="baseline", output=str(root / "baseline"))))
    # Exercise resume with a fixed LR schedule and RNG/data-state restoration.
    baseline = train(configuration(overrides=dict(common, steps=2, stage="baseline",
                                                 resume=str(baseline), output=str(root / "baseline"))))
    flow = train(configuration(overrides=dict(common, stage="flow", init=str(baseline), output=str(root / "flow"))))
    utility = train(configuration(overrides=dict(common, stage="utility", init=str(flow), output=str(root / "utility"))))
    calibrated = root / "utility/calibrated.pt"
    calibration = calibrate(utility, manifest, calibrated, device_name=device, size=16, steps=2, precision=precision)
    result = evaluate(calibrated, manifest, root / "evaluation", device_name=device, precision=precision,
                      size=16, steps=2, diagnostics=True, save_images=True)
    report = {"passed": True, "sampler": sampler, "precision": precision,
              "environment": environment_report(), "calibration": calibration,
              "evaluation_count": result["count"],
              "exercised": ["baseline forward/backward", "resume", sampler + " field update",
                            "frozen counterfactual utility update", "held-out calibration",
                            "test metrics", "image export", "checkpoint strict loading"],
              "scientific_result": False,
              "note": "Synthetic plumbing test only. This is not benchmark performance or PPU validation when device=cpu."}
    atomic_json(root / "smoke_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
