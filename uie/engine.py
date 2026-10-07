import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from .data import ImageDataset, audit_manifest
from .models import RestorationSystem
from .physics import corrupt_prior
from .objectives import reconstruction_loss, utility_target
from .metrics import paired_metrics
from .runtime import (device_for, seed_all, autocast, environment_report, rng_state,
                      restore_rng, atomic_json, atomic_checkpoint, load_checkpoint)

DEFAULTS = {
    "stage": "baseline", "manifest": "data/manifests/UIEB.json", "output": "outputs/baseline",
    "device": "auto", "precision": "fp32", "seed": 42, "width": 32, "patch_mode": "dense",
    "sigma": 0.05, "sampler": "flow", "size": 256, "batch_size": 8, "grad_accum": 1, "workers": 4,
    "steps": 20000, "schedule_steps": 20000, "warmup": 200, "lr": 0.0002,
    "weight_decay": 0.000001, "clip_grad": 1.0, "validate_every": 500, "save_every": 500,
    "log_every": 20, "nfe": 4, "prior_dropout": 0.25, "terminal_weight": 0.1,
    "utility_delta": 0.001, "utility_window": 9, "flow_gate": "dropout",
    "corruptions": ["clean", "clean", "color", "shift", "missing"],
    "init": None, "resume": None, "cpu_threads": 4,
}


def configuration(path=None, overrides=None):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    values = values or {}
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    cfg = dict(DEFAULTS, **values)
    if cfg["stage"] not in ("baseline", "flow", "utility"):
        raise ValueError("stage must be baseline/flow/utility")
    for key in ("steps", "schedule_steps", "batch_size", "grad_accum", "validate_every",
                "save_every", "log_every", "nfe"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if cfg["steps"] > cfg["schedule_steps"]:
        raise ValueError("steps cannot exceed schedule_steps; keep the schedule fixed on resume")
    if not 0 <= cfg["prior_dropout"] <= 1:
        raise ValueError("prior_dropout must be in [0,1]")
    if cfg["flow_gate"] not in ("dropout", "learned"):
        raise ValueError("flow_gate must be dropout/learned")
    return cfg


class BatchStream:
    """Reconstructible sample order and per-sample augmentation across resume."""
    def __init__(self, dataset, cfg, state=None):
        self.dataset, self.cfg = dataset, cfg
        self.epoch = (state or {}).get("epoch", 0)
        self.batch = (state or {}).get("batch", 0)
        self.iterator = None

    def __next__(self):
        total = math.ceil(len(self.dataset) / self.cfg["batch_size"])
        if self.batch >= total:
            self.epoch += 1
            self.batch = 0
            self.iterator = None
        if self.iterator is None:
            self.dataset.epoch = self.epoch
            generator = torch.Generator().manual_seed(self.cfg["seed"] + self.epoch)
            order = torch.randperm(len(self.dataset), generator=generator).tolist()
            order = order[self.batch * self.cfg["batch_size"]:]
            loader = DataLoader(self.dataset, batch_size=self.cfg["batch_size"], sampler=order,
                                num_workers=self.cfg["workers"], pin_memory=False,
                                generator=torch.Generator().manual_seed(self.cfg["seed"] + self.epoch))
            self.iterator = iter(loader)
        value = next(self.iterator)
        self.batch += 1
        return value

    def state(self):
        return {"epoch": self.epoch, "batch": self.batch}


def stage_parameters(model, cfg):
    model.requires_grad_(False)
    if cfg["stage"] == "baseline":
        model.coarse.requires_grad_(True)
    elif cfg["stage"] == "flow":
        model.flow.requires_grad_(True)
        model.condition.requires_grad_(True)
        if cfg["flow_gate"] == "learned":
            model.utility.requires_grad_(True)
    else:
        model.utility.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def training_loss(model, batch, cfg, device):
    image, target = batch["input"].to(device), batch["target"].to(device)
    depth = batch.get("depth")
    depth = depth.to(device) if depth is not None else None
    if cfg["stage"] == "baseline":
        return reconstruction_loss(model.coarse(image), target)
    with torch.no_grad():
        base = model.coarse(image).clamp(0, 1)
        corruption = random.choice(cfg["corruptions"])
        prior = corrupt_prior(model.prior(image, depth), corruption)
        initial = model.noise(image)
    if cfg["stage"] == "flow":
        t = torch.rand(image.shape[0], device=device)
        interpolation = t[:, None, None, None]
        residual = target - base
        if model.sampler == "flow":
            state = (1 - interpolation) * initial + interpolation * residual
            field_target = residual - initial
        else:
            alpha = model.alpha_bar(t)[:, None, None, None]
            state = alpha.sqrt() * residual + (1-alpha).sqrt() * initial
            field_target = residual
        if cfg["flow_gate"] == "learned":
            gate = torch.sigmoid(model.utility(image, base, initial, prior))
        else:
            keep = (torch.rand(image.shape[0], 1, 1, 1, device=device) >= cfg["prior_dropout"]).float()
            if random.random() < 0.5:
                # Smooth local dropout makes spatial control familiar during field training.
                grid = (torch.rand(image.shape[0], 1, 4, 4, device=device) >= cfg["prior_dropout"]).float()
                keep = keep * F.interpolate(grid, image.shape[-2:], mode="bilinear", align_corners=False)
            gate = keep
        velocity = model.velocity(image, base, state, t, prior, gate)
        loss = F.mse_loss(velocity.float(), field_target.float())
        endpoint = base + state + (1-interpolation) * velocity if model.sampler == "flow" else base + velocity
        return loss + cfg["terminal_weight"] * reconstruction_loss(endpoint, target)
    # Actual frozen solver endpoints with identical starting noise, not x1 proxies.
    with torch.no_grad():
        features = model.condition(image)
        off = model.integrate(image, base, prior, initial, 0.0, cfg["nfe"], features)
        on = model.integrate(image, base, prior, initial, 1.0, cfg["nfe"], features)
        _, label = utility_target(off, on, target, cfg["utility_delta"], cfg["utility_window"])
    logits = model.utility(image, base, initial, prior)
    return F.binary_cross_entropy_with_logits(logits.float(), label)


@torch.no_grad()
def validate(model, dataset, cfg, device):
    model.eval()
    score = 0.0
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["input"].unsqueeze(0).to(device)
        depth = sample.get("depth")
        depth = depth.unsqueeze(0).to(device) if depth is not None else None
        with autocast(device, cfg["precision"]):
            if cfg["stage"] == "baseline":
                prediction = model.coarse(image)
            else:
                gate = "utility" if cfg["stage"] == "utility" or cfg["flow_gate"] == "learned" else "all"
                prediction, _ = model.restore(image, steps=cfg["nfe"], gate_mode=gate,
                                              seed=cfg["seed"] + i, depth=depth)
        score += paired_metrics(prediction[0], sample["target"])["psnr"]
    return score / len(dataset)


def train(cfg):
    torch.set_num_threads(cfg["cpu_threads"])
    seed_all(cfg["seed"])
    device = device_for(cfg["device"])
    report = audit_manifest(cfg["manifest"])
    trainset = ImageDataset(cfg["manifest"], "train", cfg["size"], True, cfg["seed"])
    valset = ImageDataset(cfg["manifest"], "val", cfg["size"], False, cfg["seed"])
    if any("target" not in e for e in trainset.entries + valset.entries):
        raise ValueError("Training and validation require paired references")
    output = Path(cfg["output"])
    if (output / "last.pt").exists() and not cfg["resume"]:
        raise ValueError("Output already contains a run; use --resume or a different --output")
    output.mkdir(parents=True, exist_ok=True)
    resume = load_checkpoint(cfg["resume"]) if cfg["resume"] else None
    initial = load_checkpoint(cfg["init"]) if cfg["init"] and not resume else None
    if cfg["stage"] != "baseline" and not (resume or initial):
        raise ValueError("flow requires a baseline checkpoint; utility requires a flow checkpoint")
    model = RestorationSystem(cfg["width"], cfg["patch_mode"], cfg["sigma"], cfg["sampler"])
    if resume:
        if resume["stage"] != cfg["stage"] or resume["manifest_sha256"] != report["sha256"]:
            raise ValueError("Resume stage or manifest hash mismatch")
        strict_keys = ["width", "patch_mode", "sigma", "sampler", "seed", "size", "batch_size", "grad_accum",
                       "schedule_steps", "warmup", "lr", "weight_decay", "nfe", "flow_gate",
                       "prior_dropout", "terminal_weight", "corruptions", "utility_delta", "utility_window",
                       "precision", "clip_grad"]
        for key in strict_keys:
            if resume["config"][key] != cfg[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        model.load_state_dict(resume["model"], strict=True)
    elif initial:
        expected = "baseline" if cfg["stage"] == "flow" else "flow"
        if initial["stage"] != expected:
            raise ValueError(f"{cfg['stage']} initialization must come from {expected}")
        if initial["manifest_sha256"] != report["sha256"]:
            raise ValueError("Initialization manifest mismatch; explicit transfer training is not supported")
        if cfg["stage"] == "flow":
            coarse = {k[len("coarse."):]: v for k, v in initial["model"].items() if k.startswith("coarse.")}
            model.coarse.load_state_dict(coarse, strict=True)
        else:
            if model.spec != initial["model_spec"]:
                raise ValueError("Utility model must use the flow checkpoint model_spec")
            model.load_state_dict(initial["model"], strict=True)
    model.to(device)
    params = stage_parameters(model, cfg)
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"], foreach=False)
    start, best = 0, -1e9
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        start, best = resume["step"], resume["best_psnr"]
        restore_rng(resume["rng"])
    stream = BatchStream(trainset, cfg, resume["data_state"] if resume else None)
    provenance = {"config": cfg, "environment": environment_report(), "data": report,
                  "trainable_parameters": sum(p.numel() for p in params)}
    atomic_json(output / "run.json", provenance)
    if start >= cfg["steps"]:
        print(f"Checkpoint already reached {start} optimizer steps", flush=True)
        return output / "last.pt"
    for step in range(start + 1, cfg["steps"] + 1):
        model.eval()
        if cfg["stage"] == "baseline":
            model.coarse.train()
        elif cfg["stage"] == "flow":
            model.flow.train()
            model.condition.train()
        else:
            model.utility.train()
        if step <= cfg["warmup"]:
            factor = step / max(1, cfg["warmup"])
        else:
            progress = (step-cfg["warmup"]) / max(1, cfg["schedule_steps"]-cfg["warmup"])
            factor = 0.5 * (1 + math.cos(math.pi * min(1, progress)))
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * max(factor, 0.01)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            batch = next(stream)
            with autocast(device, cfg["precision"]):
                loss = training_loss(model, batch, cfg, device) / cfg["grad_accum"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}; last checkpoint is preserved")
            loss.backward()
            total_loss += loss.detach().float().item()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg["clip_grad"], error_if_nonfinite=True, foreach=False)
        optimizer.step()
        record = {"step": step, "loss": total_loss, "lr": optimizer.param_groups[0]["lr"],
                  "grad_norm": float(norm), **stream.state()}
        improved = False
        if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
            score = validate(model, valset, cfg, device)
            record["val_psnr"] = score
            improved = score > best
            best = max(best, score)
        if step % cfg["log_every"] == 0 or step == cfg["steps"] or improved:
            print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
        if improved or step % cfg["save_every"] == 0 or step == cfg["steps"]:
            checkpoint = {"format_version": 1, "stage": cfg["stage"], "step": step,
                          "model_spec": model.spec, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "config": cfg,
                          "best_psnr": best, "rng": rng_state(), "data_state": stream.state(),
                          "manifest_sha256": report["sha256"], "environment": provenance["environment"]}
            atomic_checkpoint(output / "last.pt", checkpoint)
            if improved:
                atomic_checkpoint(output / "best.pt", checkpoint)
    return output / "last.pt"
