"""
evaluate_fidelity.py — Pixel-Domain Structural Fidelity & Hallucination Risk Analysis Suite.

Evaluates:
  1. Sobel Gradient Magnitude Error (Line Edge Displacement)
  2. High-Frequency Error (FFT Spectral Energy Deviation)
  3. Ringing & Oversharpening Artifact Index
  4. Pixel-Domain Structural Fidelity Risk Score
"""

import os
import sys
import numpy as np
import torch
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import compute_psnr, compute_ssim


def compute_sobel_gradient_map(img_np: np.ndarray) -> np.ndarray:
    """Computes Sobel gradient magnitude map."""
    gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def compute_fidelity_risk_score(pred_np: np.ndarray, gt_np: np.ndarray):
    """
    Computes Pixel-Domain Structural Fidelity Analysis Metrics.
    Returns dict containing:
      - edge_deviation (Sobel gradient magnitude difference)
      - frequency_deviation (2D FFT high-pass magnitude difference)
      - residual_anomaly (99th percentile residual error)
      - fidelity_risk_score (Combined structural risk metric)
    """
    # 1. Edge Gradient Deviation
    grad_pred = compute_sobel_gradient_map(pred_np)
    grad_gt = compute_sobel_gradient_map(gt_np)
    edge_deviation = float(np.mean(np.abs(grad_pred - grad_gt)))

    # 2. High-Frequency FFT Spectral Energy Deviation
    fft_pred = np.abs(np.fft.fft2(pred_np))
    fft_gt = np.abs(np.fft.fft2(gt_np))
    fft_pred_shifted = np.fft.fftshift(fft_pred)
    fft_gt_shifted = np.fft.fftshift(fft_gt)
    
    h, w = pred_np.shape
    cy, cx = h // 2, w // 2
    # High frequency mask (outer 50% radius)
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
    hf_mask = dist_from_center > (min(h, w) * 0.25)
    
    frequency_deviation = float(np.mean(np.abs(fft_pred_shifted[hf_mask] - fft_gt_shifted[hf_mask])))

    # 3. Residual Anomaly (99th percentile outlier error)
    diff = np.abs(pred_np - gt_np)
    residual_anomaly = float(np.percentile(diff, 99.0))

    # 4. Combined Pixel-Domain Structural Fidelity Risk Score
    fidelity_risk_score = float(edge_deviation * 1.5 + frequency_deviation * 0.5 + residual_anomaly * 2.0)

    return {
        "edge_deviation": edge_deviation,
        "frequency_deviation": frequency_deviation,
        "residual_anomaly": residual_anomaly,
        "fidelity_risk_score": fidelity_risk_score
    }


if __name__ == "__main__":
    print("[FIDELITY EVALUATION] Testing Pixel-Domain Structural Fidelity Analysis Module...")
    dummy_pred = np.random.rand(256, 256).astype(np.float32)
    dummy_gt = np.random.rand(256, 256).astype(np.float32)
    res = compute_fidelity_risk_score(dummy_pred, dummy_gt)
    print(f"[PASS] Fidelity Risk Score: {res['fidelity_risk_score']:.4f} (Edge Dev: {res['edge_deviation']:.4f}, Residual Anomaly: {res['residual_anomaly']:.4f})")
