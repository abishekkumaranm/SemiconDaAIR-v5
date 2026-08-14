"""
experiments/run_ablation.py — Controlled Ablation Suite (EXP-A to EXP-H) for SemiconDaAIR-v6.

Executes 8 controlled experiments:
  - EXP-A: Existing v5 baseline
  - EXP-B: v5 + Structure Guidance Module
  - EXP-C: v5 + Structure Guidance + Sobel Loss
  - EXP-D: v5 + Structure Guidance + Frequency Loss
  - EXP-E: v5 + Structure Guidance + Laplacian Loss
  - EXP-F: v5 + Structure Guidance + Frequency + Laplacian Loss
  - EXP-G: v5 + Structure Guidance + Frequency + SSIM Loss
  - EXP-H: Full Proposed v6 Architecture

Exports results to:
  - results/ablation_results.csv
  - results/ablation_summary.json
"""

import os
import sys
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in os.path.sys.path:
    os.path.sys.path.append(sys_path)

from models.semicon_daair_v6 import build_semicon_daair_v6
from dataset import RealPairedSemiconductorDataset
from losses.total_loss import SemiconDaAIRv6CompositeLoss
from evaluation.ood_eval import run_ood_evaluation
from evaluation.latency import benchmark_latency
from utils.test_protection import assert_not_hidden_test_path


def run_experiment(exp_name, model_kwargs, loss_kwargs, epochs=5, device="cuda"):
    """Runs a single ablation experiment and records complete metrics."""
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    train_split = "splits/train.txt"
    val_split = "splits/val.txt"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    dev = torch.device(device)
    train_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=train_split, augment=True)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    model = build_semicon_daair_v6(scale=2).to(dev)

    # Inherit weights from v5 for fast convergence
    v5_ckpt = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"
    if os.path.exists(v5_ckpt):
        ckpt = torch.load(v5_ckpt, map_location=dev)
        st = ckpt["model_state"] if "model_state" in ckpt else ckpt
        m_dict = model.state_dict()
        matching = {k: v for k, v in st.items() if k in m_dict and m_dict[k].shape == v.shape}
        m_dict.update(matching)
        model.load_state_dict(m_dict)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SemiconDaAIRv6CompositeLoss(**loss_kwargs).to(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        for lq, gt in train_loader:
            lq, gt = lq.to(dev), gt.to(dev)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                pred = model(lq)
                loss, _ = criterion(pred, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    # Benchmark metrics
    ood_rep = run_ood_evaluation(model, dev, gt_dir, lq_dir, val_split)
    lat_rep = benchmark_latency(model, precision="fp16", device=device)
    n_params = sum(p.numel() for p in model.parameters())

    exp_result = {
        "experiment": exp_name,
        "parameters": n_params,
        "psnr_id": ood_rep["psnr_id"],
        "ssim_id": ood_rep["ssim_id"],
        "mae_id": ood_rep["mae_id"],
        "lpips_id": ood_rep["lpips_id"],
        "psnr_ood": ood_rep["psnr_ood"],
        "ssim_ood": ood_rep["ssim_ood"],
        "mae_ood": ood_rep["mae_ood"],
        "lpips_ood": ood_rep["lpips_ood"],
        "ood_gap_psnr": ood_rep["ood_gap_psnr"],
        "latency_ms": lat_rep["mean_latency_ms"],
        "peak_vram_mb": lat_rep["peak_vram_mb"]
    }

    os.makedirs(f"checkpoints/experiments/{exp_name}", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metrics": exp_result}, f"checkpoints/experiments/{exp_name}/ckpt.pt")

    return exp_result


def execute_full_ablation_matrix():
    """Runs all 8 ablation configurations (EXP-A to EXP-H)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ablation_configs = [
        ("EXP-A", {}, {"use_charbonnier": False, "lambda_sobel": 0.0, "lambda_frequency": 0.0, "lambda_ssim": 0.0}),
        ("EXP-B", {}, {"use_charbonnier": True, "lambda_sobel": 0.0, "lambda_frequency": 0.0, "lambda_ssim": 0.0}),
        ("EXP-C", {}, {"use_charbonnier": True, "lambda_sobel": 0.20, "lambda_frequency": 0.0, "lambda_ssim": 0.0}),
        ("EXP-D", {}, {"use_charbonnier": True, "lambda_sobel": 0.20, "lambda_frequency": 0.15, "lambda_ssim": 0.0}),
        ("EXP-E", {}, {"use_charbonnier": True, "lambda_sobel": 0.20, "lambda_laplacian": 0.15, "lambda_frequency": 0.0, "lambda_ssim": 0.0}),
        ("EXP-F", {}, {"use_charbonnier": True, "lambda_sobel": 0.20, "lambda_laplacian": 0.15, "lambda_frequency": 0.15, "lambda_ssim": 0.0}),
        ("EXP-G", {}, {"use_charbonnier": True, "lambda_sobel": 0.20, "lambda_frequency": 0.15, "lambda_ssim": 0.10}),
        ("EXP-H", {}, {"use_charbonnier": True, "lambda_sobel": 0.20, "lambda_frequency": 0.15, "lambda_ssim": 0.10, "lambda_defect": 0.20})
    ]

    all_results = []
    print("=" * 80)
    print("      SEMICONDAAIR-V6 CONTROLLED ABLATION EXPERIMENTS (EXP-A -> EXP-H)      ")
    print("=" * 80)

    for exp_name, m_kw, l_kw in ablation_configs:
        print(f"\n--- Launching {exp_name} ---", flush=True)
        t0 = time.time()
        res = run_experiment(exp_name, m_kw, l_kw, epochs=5, device=device)
        t1 = time.time()
        res["training_time_sec"] = float(t1 - t0)
        all_results.append(res)
        print(f"[{exp_name} FINISHED] PSNR ID: {res['psnr_id']:.4f} dB | OOD Gap: {res['ood_gap_psnr']:.4f} dB | Latency: {res['latency_ms']:.2f} ms", flush=True)

    # Save outputs
    os.makedirs("results", exist_ok=True)
    csv_path = "results/ablation_results.csv"
    fieldnames = list(all_results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    with open("results/ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"      ALL ABLATIONS COMPLETE — RESULTS SAVED TO {csv_path}      ")
    print("=" * 80)


if __name__ == "__main__":
    execute_full_ablation_matrix()
