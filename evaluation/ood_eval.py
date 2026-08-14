"""
evaluation/ood_eval.py — In-Distribution (ID) vs Out-of-Distribution (OOD) Benchmark Evaluator.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in os.path.sys.path:
    os.path.sys.path.append(sys_path)

from evaluation.metrics import evaluate_metrics_full
from utils.test_protection import assert_not_hidden_test_path
from dataset import RealPairedSemiconductorDataset
from utils.ood_validator import OODSemiconductorDataset


def run_ood_evaluation(model, device, gt_dir, lq_dir, val_split_file):
    """
    Evaluates model performance on both ID and OOD datasets.
    Computes PSNR, SSIM, MAE, LPIPS for both splits and reports the Generalization Gap.
    """
    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    model.eval()

    id_dataset = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=val_split_file, augment=False)
    ood_dataset = OODSemiconductorDataset(gt_dir, lq_dir, split_file=val_split_file, shift_type="intensity_noise")

    id_loader = DataLoader(id_dataset, batch_size=1, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=1, shuffle=False)

    # 1. ID Evaluation
    id_psnr, id_ssim, id_mae, id_lpips = [], [], [], []
    with torch.no_grad():
        for lq, gt in id_loader:
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            p_np = pred.squeeze().cpu().numpy()
            g_np = gt.squeeze().cpu().numpy()
            m = evaluate_metrics_full(p_np, g_np, device=device)
            id_psnr.append(m["psnr"])
            id_ssim.append(m["ssim"])
            id_mae.append(m["mae"])
            id_lpips.append(m["lpips"])

    # 2. OOD Evaluation
    ood_psnr, ood_ssim, ood_mae, ood_lpips = [], [], [], []
    with torch.no_grad():
        for lq, gt in ood_loader:
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            p_np = pred.squeeze().cpu().numpy()
            g_np = gt.squeeze().cpu().numpy()
            m = evaluate_metrics_full(p_np, g_np, device=device)
            ood_psnr.append(m["psnr"])
            ood_ssim.append(m["ssim"])
            ood_mae.append(m["mae"])
            ood_lpips.append(m["lpips"])

    mean_id_psnr = float(np.mean(id_psnr))
    mean_id_ssim = float(np.mean(id_ssim))
    mean_id_mae = float(np.mean(id_mae))
    mean_id_lpips = float(np.mean(id_lpips))

    mean_ood_psnr = float(np.mean(ood_psnr))
    mean_ood_ssim = float(np.mean(ood_ssim))
    mean_ood_mae = float(np.mean(ood_mae))
    mean_ood_lpips = float(np.mean(ood_lpips))

    report = {
        "psnr_id": mean_id_psnr,
        "ssim_id": mean_id_ssim,
        "mae_id": mean_id_mae,
        "lpips_id": mean_id_lpips,
        "psnr_ood": mean_ood_psnr,
        "ssim_ood": mean_ood_ssim,
        "mae_ood": mean_ood_mae,
        "lpips_ood": mean_ood_lpips,
        "ood_gap_psnr": mean_id_psnr - mean_ood_psnr,
        "ood_gap_ssim": mean_id_ssim - mean_ood_ssim
    }
    return report


if __name__ == "__main__":
    from models.semicon_daair_v6 import build_semicon_daair_v6
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_semicon_daair_v6(scale=2).to(dev)
    rep = run_ood_evaluation(
        m, dev,
        r"C:\Users\HP\Downloads\dataset\train\train\GT",
        r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR",
        "splits/val.txt"
    )
    print("OOD Benchmark Summary:", json.dumps(rep, indent=2))
