"""
evaluation/metrics.py — Official Evaluation Suite & Grayscale LPIPS Adapter.

Includes:
  - Quality Metrics: PSNR, SSIM, MAE, RMSE
  - Documented Grayscale LPIPS Adapter: Repeats 1-channel grayscale tensor 3 times to (N, 3, H, W) for LPIPS.
"""

import os
import sys
import math
import numpy as np
import cv2
import torch

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in os.path.sys.path:
    os.path.sys.path.append(sys_path)

from metrics import compute_psnr, compute_ssim

try:
    import lpips
    HAS_LPIPS = True
    _LPIPS_MODEL = None
except ImportError:
    HAS_LPIPS = False
    _LPIPS_MODEL = None


def get_lpips_model(device="cuda"):
    global _LPIPS_MODEL
    if HAS_LPIPS and _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).to(device)
    return _LPIPS_MODEL


def compute_grayscale_lpips(pred_np: np.ndarray, gt_np: np.ndarray, device="cuda") -> float:
    """
    Grayscale LPIPS Adapter:
    Converts 1-channel float32 array [H, W] in [0, 1] to 3-channel tensor [1, 3, H, W] in [-1, 1]
    to evaluate perceptual distance without changing internal 1-channel model representation.
    """
    if not HAS_LPIPS:
        return 0.0

    lpips_fn = get_lpips_model(device)
    if lpips_fn is None:
        return 0.0

    p_norm = np.clip(pred_np, 0.0, 1.0)
    g_norm = np.clip(gt_np, 0.0, 1.0)

    # Convert to tensor [1, 1, H, W] and repeat 3 channels
    p_t = torch.from_numpy(p_norm)[None, None].repeat(1, 3, 1, 1).to(device)
    g_t = torch.from_numpy(g_norm)[None, None].repeat(1, 3, 1, 1).to(device)

    # Scale to [-1, 1] for LPIPS
    with torch.no_grad():
        dist = lpips_fn(p_t * 2.0 - 1.0, g_t * 2.0 - 1.0)
        return float(dist.item())


def evaluate_metrics_full(pred_np: np.ndarray, gt_np: np.ndarray, device="cuda") -> dict:
    """Computes comprehensive evaluation metrics: PSNR, SSIM, MAE, RMSE, LPIPS."""
    p_norm = np.clip(pred_np, 0.0, 1.0)
    g_norm = np.clip(gt_np, 0.0, 1.0)

    psnr = compute_psnr(p_norm, g_norm)
    ssim = compute_ssim(p_norm, g_norm)
    mae = float(np.mean(np.abs(p_norm - g_norm)))
    rmse = float(np.sqrt(np.mean((p_norm - g_norm) ** 2)))
    lpips_val = compute_grayscale_lpips(p_norm, g_norm, device=device)

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
        "mae": float(mae),
        "rmse": float(rmse),
        "lpips": float(lpips_val)
    }


if __name__ == "__main__":
    p = np.random.rand(256, 256).astype(np.float32)
    g = np.random.rand(256, 256).astype(np.float32)
    m = evaluate_metrics_full(p, g, device="cpu")
    print("Grayscale Evaluation Metrics Output:", m)
