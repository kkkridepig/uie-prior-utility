import contextlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible device unavailable; check the system PPU torch/runtime")
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError("Only fp32/bf16 are supported")
    if device.type != "cuda":
        raise ValueError("bf16 is enabled only after CUDA-compatible PPU/GPU validation")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This runtime does not report BF16 support; use fp32")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def environment_report():
    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "torch_path": torch.__file__, "cuda_interface": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
              "numpy": np.__version__}
    result["devices"] = [
        {"name": torch.cuda.get_device_name(i),
         "memory_gib": torch.cuda.get_device_properties(i).total_memory / 1024**3}
        for i in range(torch.cuda.device_count())
    ]
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    return result


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    # Only load your own trusted checkpoints; optimizer/RNG data needs pickle.
    return torch.load(path, map_location=device, weights_only=False)


def load_system(path, device):
    from .models import RestorationSystem
    checkpoint = load_checkpoint(path)
    model = RestorationSystem(**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint
