"""Explicit metric conventions; do not mix with upstream HSV 'UCIQE'."""
import math

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity


def as_image(tensor):
    return tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def paired_metrics(output, reference):
    x, y = as_image(output), as_image(reference)
    mse = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    # Cap only the exactly identical case so JSON remains finite.
    psnr = -10 * math.log10(max(mse, 1e-12))
    window = min(11, min(x.shape[:2]))
    window -= 1 - window % 2
    if window < 3:
        raise ValueError("SSIM requires image dimensions >=3")
    ssim = structural_similarity(x, y, channel_axis=-1, data_range=1.0,
                                gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                win_size=window)
    return {"psnr": psnr, "ssim": float(ssim), "mae": float(np.mean(np.abs(x-y)))}


def nonreference_metrics(output):
    x = as_image(output)
    lab = rgb2lab(x)
    # Explicit normalized CIELab convention, recorded in every evaluation report.
    l, a, b = lab[..., 0] / 100.0, lab[..., 1] / 100.0, lab[..., 2] / 100.0
    chroma = np.sqrt(a*a + b*b)
    saturation = chroma / np.sqrt(chroma*chroma + l*l + 1e-12)
    contrast = np.percentile(l, 99) - np.percentile(l, 1)
    score = 0.4680 * np.std(chroma) + 0.2745 * contrast + 0.2576 * np.mean(saturation)
    return {"uciqe_lab_v1": float(score),
            "clipped_fraction": float(np.mean((x <= 0) | (x >= 1)))}


def calibration_metrics(probabilities, labels, bins=10):
    p = np.concatenate([np.asarray(x).reshape(-1) for x in probabilities])
    y = np.concatenate([np.asarray(x).reshape(-1) for x in labels])
    reliability, ece = [], 0.0
    for i in range(bins):
        mask = (p >= i / bins) & ((p < (i+1) / bins) if i < bins-1 else (p <= 1))
        if mask.any():
            confidence, accuracy = float(p[mask].mean()), float(y[mask].mean())
            count = int(mask.sum())
            ece += count / len(p) * abs(confidence - accuracy)
            reliability.append({"bin": i, "count": count, "probability": confidence, "frequency": accuracy})
    return {"brier": float(np.mean((p-y)**2)), "ece_10bin": ece, "reliability": reliability,
            "note": "Descriptive pixel statistics; pixels are not independent samples."}
