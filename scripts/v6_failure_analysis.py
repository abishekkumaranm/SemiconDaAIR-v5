"""
scripts/v6_failure_analysis.py — Failure Analysis & Top 10 Hard Sample Extraction Script.

Identifies:
  - Top 10 lowest PSNR samples
  - Top 10 lowest SSIM samples
  - Top 10 highest LPIPS samples
  - Top 10 highest MAE samples
Generates 4-panel visual diagnostic comparison heatmaps under results/visualizations/
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

from models.semicon_daair_v6 import build_semicon_daair_v6
from dataset import RealPairedSemiconductorDataset
from evaluation.metrics import evaluate_metrics_full
from evaluation.visualization import generate_visual_comparison_panel, compute_hallucination_risk_indicator
from utils.test_protection import assert_not_hidden_test_path


def run_failure_analysis():
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    val_split = "splits/val.txt"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=val_split, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_semicon_daair_v6(scale=2).to(dev)

    ckpt_path = "checkpoints/v6/semicon_daair_v6_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/final/semicon_daair_v5_candidate.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=dev)
        st = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(st, strict=False)

    model.eval()

    sample_metrics = []
    with torch.no_grad():
        for idx, (lq, gt) in enumerate(val_loader):
            lq, gt = lq.to(dev), gt.to(dev)
            pred = model(lq)
            p_np = pred.squeeze().cpu().numpy()
            g_np = gt.squeeze().cpu().numpy()
            lq_np = lq.squeeze().cpu().numpy()

            m = evaluate_metrics_full(p_np, g_np, device=dev)
            risk = compute_hallucination_risk_indicator(p_np, g_np)

            fname = val_ds.gt_paths[idx] if hasattr(val_ds, "gt_paths") else f"sample_{idx:04d}.npy"
            base_fname = os.path.basename(fname)

            m["filename"] = base_fname
            m["index"] = idx
            m.update(risk)
            sample_metrics.append(m)

            # Generate visual comparison panels for first 10 samples
            if idx < 10:
                out_path = f"results/visualizations/panel_{idx:04d}_{base_fname}.png"
                generate_visual_comparison_panel(lq_np, p_np, g_np, out_path)

    # Sort failures
    worst_psnr = sorted(sample_metrics, key=lambda x: x["psnr"])[:10]
    worst_ssim = sorted(sample_metrics, key=lambda x: x["ssim"])[:10]
    worst_lpips = sorted(sample_metrics, key=lambda x: x["lpips"], reverse=True)[:10]
    worst_mae = sorted(sample_metrics, key=lambda x: x["mae"], reverse=True)[:10]

    failure_report = {
        "total_evaluated": len(sample_metrics),
        "worst_10_psnr": worst_psnr,
        "worst_10_ssim": worst_ssim,
        "worst_10_lpips": worst_lpips,
        "worst_10_mae": worst_mae
    }

    os.makedirs("results", exist_ok=True)
    with open("results/failure_analysis_report.json", "w") as f:
        json.dump(failure_report, f, indent=2)

    print("=" * 80)
    print("      [FAILURE ANALYSIS COMPLETE] - REPORT SAVED TO results/failure_analysis_report.json      ")
    print("=" * 80)
    print(f"Worst PSNR Sample : {worst_psnr[0]['filename']} ({worst_psnr[0]['psnr']:.2f} dB)")
    print(f"Worst SSIM Sample : {worst_ssim[0]['filename']} ({worst_ssim[0]['ssim']:.4f})")
    print(f"Worst LPIPS Sample: {worst_lpips[0]['filename']} ({worst_lpips[0]['lpips']:.4f})")


if __name__ == "__main__":
    run_failure_analysis()
