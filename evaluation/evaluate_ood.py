"""
evaluate_ood.py — OOD-Like Synthetic Robustness Evaluation Suite for SemiconDaAIR-v3.

Evaluates model performance under controlled synthetic distribution shifts:
  1. Stronger Speckle Noise
  2. Mixed Noise (Gaussian + Speckle)
  3. Dynamic Range & Contrast Shifts
  4. Extreme Outlier Values
"""

import os
import sys
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path


def apply_ood_perturbation(gt_np: np.ndarray, perturbation_type: str) -> np.ndarray:
    """Applies controlled synthetic perturbation to GT image (256x256 -> 128x128 LQ)."""
    h, w = gt_np.shape
    # 2x Downsample
    lq = cv2_downsample(gt_np, scale=2)

    if perturbation_type == "strong_speckle":
        noise = np.random.normal(0, 0.25, lq.shape).astype(np.float32)
        lq = lq + lq * noise
    elif perturbation_type == "mixed_noise":
        gauss = np.random.normal(0, 0.08, lq.shape).astype(np.float32)
        speckle = np.random.normal(0, 0.12, lq.shape).astype(np.float32)
        lq = lq + gauss + lq * speckle
    elif perturbation_type == "contrast_shift":
        lq = (lq - 0.5) * 0.5 + 0.5
        noise = np.random.normal(0, 0.05, lq.shape).astype(np.float32)
        lq = lq + noise
    elif perturbation_type == "intensity_scale":
        lq = lq * 1.5
        speckle = np.random.normal(0, 0.10, lq.shape).astype(np.float32)
        lq = lq + lq * speckle

    return lq


def cv2_downsample(img, scale=2):
    import cv2
    h, w = img.shape
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)


def evaluate_model_ood_robustness(model, gt_paths, device="cuda"):
    """Evaluates model across 4 controlled synthetic OOD perturbations."""
    model.eval()
    results = {}

    perturbations = ["strong_speckle", "mixed_noise", "contrast_shift", "intensity_scale"]

    with torch.inference_mode():
        for ptype in perturbations:
            psnr_list, ssim_list = [], []

            for path in gt_paths[:50]:  # 50 sample evaluation
                if path.endswith(".npy"):
                    gt_np = np.load(path).astype(np.float32)
                else:
                    import cv2
                    gt_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

                lq_np = apply_ood_perturbation(gt_np, ptype)

                tensor_lq = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)
                out_tensor = model(tensor_lq)
                if isinstance(out_tensor, tuple):
                    out_tensor = out_tensor[0]

                out_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()

                p = compute_psnr(out_np, gt_np)
                s = compute_ssim(out_np, gt_np)

                psnr_list.append(p)
                ssim_list.append(s)

            results[ptype] = {
                "psnr": float(np.mean(psnr_list)),
                "ssim": float(np.mean(ssim_list))
            }

    overall_psnr = float(np.mean([results[k]["psnr"] for k in results]))
    overall_ssim = float(np.mean([results[k]["ssim"] for k in results]))

    return overall_psnr, overall_ssim, results


if __name__ == "__main__":
    print("[OOD EVALUATION] Module loaded.")
