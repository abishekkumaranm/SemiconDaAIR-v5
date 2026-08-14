"""
train_v6_ablation.py — Training & Controlled Loss Ablation Suite for SemiconDaAIR-v6.

Runs 25-Epoch Training with FP16 AMP, Cosine Annealing, reproducible seed, and complete metric evaluation:
  - In-Distribution (ID) Val PSNR, SSIM, LPIPS, MAE
  - Out-of-Distribution (OOD) Val PSNR, SSIM, LPIPS, MAE, Generalization Gap
  - Loss Ablation Study (Configurations A, B, C, D, E, F) -> Exports ablation_results.csv
  - Synchronized CUDA Latency & Parameter Count
  - Checkpoint Promotion: Saves best candidate to checkpoints/final/semicon_daair_v6_candidate.pt
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v6 import build_semicon_daair_v6
from dataset import RealPairedSemiconductorDataset
from losses.v6_composite_loss import V6AblationLoss
from utils.ood_validator import evaluate_id_vs_ood
from metrics import compute_psnr, compute_ssim

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def train_single_ablation(config_name="F", epochs=25, lr=2e-4, batch_size=16, device="cuda"):
    """Trains SemiconDaAIR-v6 with a specific loss configuration."""
    print("=" * 75)
    print(f"   [TRAINING V6 CANDIDATE - LOSS CONFIGURATION {config_name}] ({device})   ")
    print("=" * 75)

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    train_split = "splits/train.txt"
    val_split = "splits/val.txt"

    train_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=train_split, augment=True)
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=val_split, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_semicon_daair_v6(scale=2).to(device)

    # Inherit matching backbone weights from v5 for stable convergence
    v5_ckpt_path = "checkpoints/final/semicon_daair_v5_candidate.pt"
    if os.path.exists(v5_ckpt_path):
        ckpt_v5 = torch.load(v5_ckpt_path, map_location=device)
        state_v5 = ckpt_v5["model_state"] if "model_state" in ckpt_v5 else ckpt_v5
        model_dict = model.state_dict()
        matching = {k: v for k, v in state_v5.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matching)
        model.load_state_dict(model_dict)
        print(f"[WEIGHT INHERITANCE] Inherited {len(matching)} matching tensor weights from {v5_ckpt_path}.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = V6AblationLoss(config_name=config_name).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_psnr = 0.0
    best_ssim = 0.0
    best_mae = 1.0

    os.makedirs("checkpoints/final", exist_ok=True)
    ckpt_save_path = "checkpoints/final/semicon_daair_v6_candidate.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for lq, gt in train_loader:
            lq, gt = lq.to(device), gt.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(lq)
                loss, _ = criterion(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        psnr_list, ssim_list, mae_list = [], [], []
        with torch.no_grad():
            for lq, gt in val_loader:
                lq, gt = lq.to(device), gt.to(device)
                pred = model(lq)
                p_np = pred.squeeze().cpu().numpy()
                g_np = gt.squeeze().cpu().numpy()

                psnr_list.append(compute_psnr(p_np, g_np))
                ssim_list.append(compute_ssim(p_np, g_np))
                mae_list.append(float(np.mean(np.abs(p_np - g_np))))

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        val_mae = float(np.mean(mae_list))

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] (LR: {current_lr:.6f}) - Loss: {avg_loss:.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae
            }, ckpt_save_path)
            print(f"  [SAVED BEST V6] {ckpt_save_path} (PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f})")

    # Final Benchmark & Latency Measurement
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128, device=device)
    for _ in range(10): _ = model(dummy_input)
    if device.type == "cuda": torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(100): _ = model(dummy_input)
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    latency_ms = float(((t1 - t0) / 100.0) * 1000.0)

    # Evaluate ID vs OOD
    ood_metrics = evaluate_id_vs_ood(model, device, gt_dir, lq_dir, val_split)

    n_params = sum(p.numel() for p in model.parameters())

    summary = {
        "config": config_name,
        "parameters": n_params,
        "best_psnr_id": best_psnr,
        "best_ssim_id": best_ssim,
        "best_mae_id": best_mae,
        "psnr_ood": ood_metrics["psnr_ood"],
        "ssim_ood": ood_metrics["ssim_ood"],
        "lpips_id": ood_metrics["lpips_id"],
        "lpips_ood": ood_metrics["lpips_ood"],
        "psnr_gap": ood_metrics["psnr_gap"],
        "latency_ms": latency_ms
    }

    return summary


def run_full_v6_ablation_suite():
    """Runs controlled loss ablation across configurations A, B, C, D, E, F."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # Fast 5-epoch ablation probe for A-E to compare loss contributions
    for cfg in ["A", "B", "C", "D", "E"]:
        res = train_single_ablation(config_name=cfg, epochs=5, device=device)
        results.append(res)

    # Full 25-epoch competition run for Configuration F (Full Composite Loss)
    res_f = train_single_ablation(config_name="F", epochs=25, device=device)
    results.append(res_f)

    # Save ablation_results.csv
    csv_path = "ablation_results.csv"
    fieldnames = ["config", "parameters", "best_psnr_id", "best_ssim_id", "best_mae_id", "psnr_ood", "ssim_ood", "lpips_id", "lpips_ood", "psnr_gap", "latency_ms"]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n" + "=" * 80)
    print("      [FULL V6 ABLATION STUDY COMPLETE] - CSV SAVED TO ablation_results.csv      ")
    print("=" * 80)
    for r in results:
        print(f"Config {r['config']} | ID PSNR: {r['best_psnr_id']:.4f} dB | OOD PSNR: {r['psnr_ood']:.4f} dB | Gap: {r['psnr_gap']:.4f} dB | Latency: {r['latency_ms']:.2f} ms")


if __name__ == "__main__":
    run_full_v6_ablation_suite()
