"""
metrics.py — Metrology-Aware & Image Quality Evaluation Metrics for Semiconductor Inspection.

Includes:
  - Standard Image Quality Metrics: PSNR, SSIM, MAE, RMSE
  - Metrology-Specific Inspection Metrics:
      1. Critical Dimension (CD) Error (nm scale pixel boundary displacement)
      2. Line Edge Roughness (LER) Fidelity (1D edge roughness variance)
      3. Overlay Shift Registration Error (sub-pixel centroid alignment delta)
"""

import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def compute_psnr(img1, img2, max_val=1.0):
    """Computes Peak Signal-to-Noise Ratio (PSNR)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * math.log10(max_val / math.sqrt(mse)))


def compute_ssim(img1, img2):
    """Computes Structural Similarity Index (SSIM) using OpenCV Gaussian window."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def compute_cd_error(pred_img, target_img, nm_per_pixel=1.5, threshold=0.5):
    """
    Computes Critical Dimension (CD) measurement error in nanometers (nm).
    Measures 50% threshold width of lithographic lines across sample profiles.
    """
    pred_bin = (pred_img > threshold).astype(np.uint8)
    target_bin = (target_img > threshold).astype(np.uint8)

    # Compute row-wise CD width (sum of line pixels per row)
    pred_cd = np.sum(pred_bin, axis=1) * nm_per_pixel
    target_cd = np.sum(target_bin, axis=1) * nm_per_pixel

    cd_mae = float(np.mean(np.abs(pred_cd - target_cd)))
    cd_rmse = float(np.sqrt(np.mean((pred_cd - target_cd) ** 2)))
    return {"cd_mae_nm": cd_mae, "cd_rmse_nm": cd_rmse}


def compute_overlay_error(pred_img, target_img):
    """
    Computes Overlay Registration Shift Error (sub-pixel centroid delta dx, dy).
    """
    M_pred = cv2.moments((pred_img * 255).astype(np.uint8))
    M_target = cv2.moments((target_img * 255).astype(np.uint8))

    if M_pred["m00"] == 0 or M_target["m00"] == 0:
        return {"overlay_shift_px": 0.0}

    cx_pred = M_pred["m10"] / M_pred["m00"]
    cy_pred = M_pred["m01"] / M_pred["m00"]

    cx_target = M_target["m10"] / M_target["m00"]
    cy_target = M_target["m01"] / M_target["m00"]

    shift = math.sqrt((cx_pred - cx_target) ** 2 + (cy_pred - cy_target) ** 2)
    return {"overlay_shift_px": float(shift)}


def compute_ler_fidelity(pred_img, target_img):
    """
    Computes Line Edge Roughness (LER) 3-sigma variance preservation ratio.
    """
    # Extract edge position (column index of first 0.5 threshold crossing)
    def extract_edge_profile(img):
        edges = []
        for row in img:
            crossings = np.where(np.diff(row > 0.5))[0]
            if len(crossings) > 0:
                edges.append(crossings[0])
        return np.array(edges) if len(edges) > 0 else np.array([0.0])

    edge_p = extract_edge_profile(pred_img)
    edge_t = extract_edge_profile(target_img)

    ler_p = float(3 * np.std(edge_p)) if len(edge_p) > 1 else 0.0
    ler_t = float(3 * np.std(edge_t)) if len(edge_t) > 1 else 0.0
    ler_error = float(abs(ler_p - ler_t))
    return {"ler_pred_3sigma": ler_p, "ler_target_3sigma": ler_t, "ler_error_px": ler_error}


def evaluate_sample(pred_img, target_img):
    """Computes full evaluation report for a restored vs target image pair."""
    p_norm = np.clip(pred_img, 0, 1)
    t_norm = np.clip(target_img, 0, 1)

    psnr = compute_psnr(p_norm, t_norm)
    ssim = compute_ssim(p_norm, t_norm)
    rmse = float(np.sqrt(np.mean((p_norm - t_norm) ** 2)))
    mae = float(np.mean(np.abs(p_norm - t_norm)))
    cd_metrics = compute_cd_error(p_norm, t_norm)
    overlay_metrics = compute_overlay_error(p_norm, t_norm)
    ler_metrics = compute_ler_fidelity(p_norm, t_norm)

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "RMSE": rmse,
        "MAE": mae,
        **cd_metrics,
        **overlay_metrics,
        **ler_metrics
    }
