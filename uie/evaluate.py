import csv
import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .data import ImageDataset, audit_manifest, audit_cross_manifests, sha256_file
from .metrics import paired_metrics, nonreference_metrics, as_image, calibration_metrics
from .objectives import utility_target
from .physics import corrupt_prior
from .runtime import (device_for, load_system, atomic_json, environment_report,
                      autocast, atomic_checkpoint)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(checkpoint, manifest, output, split="test", device_name="auto", precision="fp32",
             size=256, steps=4, gate_mode="auto", corruption="clean", seed=42,
             fixed_gate=0.5, diagnostics=False, save_images=False, source_manifest=None,
             limit=0, warmup=2):
    if size < 0 or limit < 0 or steps < 1 or warmup < 0:
        raise ValueError("size/limit/warmup must be nonnegative and steps must be positive")
    if not 0 <= fixed_gate <= 1:
        raise ValueError("fixed_gate must be in [0,1]")
    device = device_for(device_name)
    model, ckpt = load_system(checkpoint, device)
    audit_manifest(manifest)
    if source_manifest:
        if sha256_file(source_manifest) != ckpt["manifest_sha256"]:
            raise ValueError("Source manifest must match the checkpoint training manifest")
        audit_manifest(source_manifest)
        audit_cross_manifests(source_manifest, manifest)
    if gate_mode == "auto":
        gate_mode = "utility" if ckpt["stage"] == "utility" else (
            "learned" if ckpt["config"]["flow_gate"] == "learned" else "all")
    if ckpt["stage"] == "flow" and gate_mode in ("utility", "learned") and ckpt["config"]["flow_gate"] != "learned":
        raise ValueError("This checkpoint has no trained utility head; use all/none/fixed")
    dataset = ImageDataset(manifest, split, size=size)
    if diagnostics and (ckpt["stage"] == "baseline" or any("target" not in e for e in dataset.entries)):
        raise ValueError("Utility diagnostics require a flow/utility checkpoint and paired references")
    calibration = ckpt.get("calibration")
    calibration_applicable = bool(calibration and calibration["nfe"] == steps
                                  and calibration["size"] == size and corruption == "clean"
                                  and calibration["manifest_sha256"] == sha256_file(manifest)
                                  and calibration.get("precision", "fp32") == precision)
    if calibration and not calibration_applicable:
        warnings.warn("Calibration dataset/NFE/size/precision/clean-prior scope differs from evaluation; "
                      "temperature is retained but calibration does not transfer automatically.", stacklevel=2)
    outdir = Path(output)
    if (outdir / "summary.json").exists():
        raise ValueError("Evaluation output exists; choose a new directory")
    outdir.mkdir(parents=True, exist_ok=True)
    rows, probabilities, labels, image_brier = [], [], [], []
    count = min(len(dataset), limit) if limit else len(dataset)
    peak_inference_bytes = 0
    for i in range(count):
        sample = dataset[i]
        x = sample["input"].unsqueeze(0).to(device)
        depth = sample.get("depth")
        depth = depth.unsqueeze(0).to(device) if depth is not None else None

        def infer():
            with autocast(device, precision):
                if ckpt["stage"] == "baseline":
                    return model.coarse(x), None
                return model.restore(x, steps, gate_mode, seed+i, corruption, fixed_gate, depth)
        if i == 0:
            for _ in range(warmup):
                infer()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        prediction, gate = infer()
        prediction = prediction.float().clamp(0, 1)
        synchronize(device)
        elapsed = time.perf_counter() - started
        if device.type == "cuda":
            peak_inference_bytes = max(peak_inference_bytes, torch.cuda.max_memory_allocated(device))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite prediction: {sample['id']}")
        row = {"id": sample["id"], "seconds": elapsed, **nonreference_metrics(prediction[0])}
        row.update({"raw_" + key: value for key, value in nonreference_metrics(sample["input"]).items()})
        if "target" in sample:
            row.update(paired_metrics(prediction[0], sample["target"]))
            row["raw_psnr"] = paired_metrics(sample["input"], sample["target"])["psnr"]
            row["harm_vs_raw"] = float(row["psnr"] < row["raw_psnr"])
        if gate is not None:
            row["gate_mean"] = float(gate.mean())
        if diagnostics:
            target = sample["target"].unsqueeze(0).to(device)
            with autocast(device, precision):
                base = model.coarse(x).clamp(0, 1)
                prior = corrupt_prior(model.prior(x, depth), corruption)
                initial = model.noise(x, seed+i)
                features = model.condition(x)
                off = model.integrate(x, base, prior, initial, 0.0, steps, features)
                on = model.integrate(x, base, prior, initial, 1.0, steps, features)
                if ckpt["stage"] == "utility":
                    delta = ckpt["config"]["utility_delta"]
                    _, label = utility_target(off, on, target, delta, ckpt["config"]["utility_window"])
                    probability = torch.sigmoid(model.utility(x, base, initial, prior).float() / model.temperature)
            row["off_psnr"] = paired_metrics(off[0], sample["target"])["psnr"]
            row["on_psnr"] = paired_metrics(on[0], sample["target"])["psnr"]
            row["harm_vs_off"] = float(row["psnr"] < row["off_psnr"])
            row["delta_psnr_vs_off"] = row["psnr"] - row["off_psnr"]
            if ckpt["stage"] == "utility":
                row["brier"] = float((probability-label).square().mean())
                image_brier.append(row["brier"])
                # Deterministic subsampling only for the reliability diagram.
                stride = max(1, probability.numel() // 4096)
                probabilities.append(probability.cpu().numpy().reshape(-1)[::stride])
                labels.append(label.cpu().numpy().reshape(-1)[::stride])
                del probability, label
            del target, base, prior, initial, features, off, on
        if save_images:
            unique = hashlib.sha256(sample["id"].encode()).hexdigest()[:10]
            filename = f"{Path(sample['id']).name}_{unique}.png"
            images = outdir / "images"
            images.mkdir(exist_ok=True)
            Image.fromarray(np.round(as_image(prediction[0])*255).astype(np.uint8)).save(images / filename)
            if gate is not None:
                gates = outdir / "gates"
                gates.mkdir(exist_ok=True)
                Image.fromarray(np.round(gate[0, 0].float().cpu().numpy()*255).astype(np.uint8)).save(gates / filename)
        rows.append(row)
        del prediction, gate, x, depth
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (outdir / "per_image.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    means = {key: float(np.mean([row[key] for row in rows if key in row])) for key in fields if key != "id"}
    latencies = [r["seconds"] for r in rows]
    summary = {
        "count": count, "means": means,
        "latency_p50_ms": float(np.percentile(latencies, 50)*1000),
        "latency_p95_ms": float(np.percentile(latencies, 95)*1000),
        "timing_scope": "batch=1; includes coarse/prior/CNN-patch/gate/all solver steps/clamp; excludes disk I/O, CPU metrics and host-to-device copy; diagnostics not timed",
        "peak_allocated_gib": peak_inference_bytes/1024**3 if device.type == "cuda" else None,
        "memory_scope": "Peak allocated during timed inference, including resident model; excludes diagnostic rollouts",
        "protocol": {"size": size, "nfe": 0 if ckpt["stage"] == "baseline" else steps,
                     "sampler": model.sampler,
                     "precision": precision, "gate": gate_mode, "fixed_gate": fixed_gate,
                     "corruption": corruption, "seed": seed, "split": split,
                     "ssim": "RGB float [0,1], Gaussian sigma=1.5, 11x11 (smaller for tiny images), population covariance",
                     "uciqe": "uciqe_lab_v1: CIELab/100, chroma std + L percentiles(99-1) + C/sqrt(C²+L²)"},
        "checkpoint_sha256": sha256_file(checkpoint), "manifest_sha256": sha256_file(manifest),
        "training_manifest_sha256": ckpt["manifest_sha256"],
        "environment": environment_report(),
        "cross_dataset_exact_audit": bool(source_manifest),
        "temperature_calibration": calibration,
        "calibration_scope_matches": calibration_applicable,
        "limitations": "No near-duplicate audit, LPIPS/UIQM/URanker or downstream task scores are inferred.",
    }
    if probabilities:
        summary["calibration"] = calibration_metrics(probabilities, labels)
        summary["calibration"]["mean_image_brier"] = float(np.mean(image_brier))
    atomic_json(outdir / "summary.json", summary)
    print(f"Evaluation: {count} images -> {outdir / 'summary.json'}", flush=True)
    return summary


@torch.no_grad()
def calibrate(checkpoint, manifest, output, device_name="auto", size=256, steps=4, seed=42, precision="fp32"):
    device = device_for(device_name)
    model, ckpt = load_system(checkpoint, device)
    if ckpt["stage"] != "utility":
        raise ValueError("Temperature calibration requires a utility checkpoint")
    if ckpt["manifest_sha256"] != sha256_file(manifest):
        raise ValueError("Use the same audited manifest with its held-out calibration split")
    if Path(output).exists():
        raise ValueError("Calibration output already exists")
    audit_manifest(manifest)
    dataset = ImageDataset(manifest, "calibration", size=size)
    logits, labels = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        x, target = sample["input"].unsqueeze(0).to(device), sample["target"].unsqueeze(0).to(device)
        depth = sample.get("depth")
        depth = depth.unsqueeze(0).to(device) if depth is not None else None
        with autocast(device, precision):
            base, prior = model.coarse(x).clamp(0, 1), model.prior(x, depth)
            initial = model.noise(x, seed+i)
            features = model.condition(x)
            off = model.integrate(x, base, prior, initial, 0., steps, features)
            on = model.integrate(x, base, prior, initial, 1., steps, features)
            _, label = utility_target(off, on, target, ckpt["config"]["utility_delta"], ckpt["config"]["utility_window"])
            logit = model.utility(x, base, initial, prior).float()
        stride = max(1, logit.numel() // 4096)
        logits.append(logit.cpu().flatten()[::stride])
        labels.append(label.cpu().flatten()[::stride])
    z, y = torch.cat(logits), torch.cat(labels)
    candidates = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    scores = {t: float(F.binary_cross_entropy_with_logits(z/t, y)) for t in candidates}
    temperature = min(scores, key=scores.get)
    ckpt["model"]["temperature"] = torch.tensor(temperature)
    ckpt["calibration"] = {"temperature": temperature, "split": "calibration", "seed": seed,
                           "nfe": steps, "size": size, "precision": precision, "nll": scores,
                           "manifest_sha256": sha256_file(manifest)}
    atomic_checkpoint(output, ckpt)
    atomic_json(Path(output).with_suffix(".calibration.json"), ckpt["calibration"])
    return ckpt["calibration"]
