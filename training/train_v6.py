"""
train_v6.py — Production Training Script for SemiconDaAIR-v6 (with OOD Domain Perturbation Augmentation).

Features:
  - OOD Domain Perturbation Augmentation: Dynamic intensity scaling (0.5x to 2.0x), speckle variance shift, detector spikes.
  - Multi-Structural MoE Gating Loss
  - FP16 AMP Acceleration
  - Cosine Annealing Learning Rate Scheduler
  - Checkpoint Management: Saves best candidate to checkpoints/v6/semicon_daair_v6_best.pt
"""

import os
import sys
import time
import json
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
from dataset_ood_aug import OODDomainPerturbationDataset
from losses.total_loss import SemiconDaAIRv6CompositeLoss
from evaluation.metrics import evaluate_metrics_full
from utils.test_protection import assert_not_hidden_test_path


def train_production_v6(epochs=25, lr=2e-4, batch_size=16, device="cuda"):
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    train_split = "splits/train.txt"
    val_split = "splits/val.txt"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    dev = torch.device(device)
    print("=" * 80, flush=True)
    print(f"      [DATASET SAFETY AUDIT - SEMICONDAAIR-V6 STRUCTURAL-MoE PRODUCTION]      ", flush=True)
    print("=" * 80, flush=True)
    print(f"Train GT Directory  : {gt_dir}", flush=True)
    print(f"Train LQ Directory  : {lq_dir}", flush=True)

    # Use OOD Domain Perturbation Dataset wrapper for train set
    train_ds = OODDomainPerturbationDataset(gt_dir, lq_dir, split_file=train_split, augment=True)
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=val_split, augment=False)

    print(f"Train Samples Count : {len(train_ds)} (OOD Perturbation Active)", flush=True)
    print(f"Val Samples Count   : {len(val_ds)}", flush=True)
    print("=" * 80, flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_semicon_daair_v6(scale=2).to(dev)

    # Inherit matching backbone weights from v5 candidate if exists
    v5_ckpt = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"
    if os.path.exists(v5_ckpt):
        ckpt = torch.load(v5_ckpt, map_location=dev)
        st = ckpt["model_state"] if "model_state" in ckpt else ckpt
        m_dict = model.state_dict()
        matching = {k: v for k, v in st.items() if k in m_dict and m_dict[k].shape == v.shape}
        m_dict.update(matching)
        model.load_state_dict(m_dict, strict=False)
        print(f"[WEIGHT INHERITANCE] Inherited {len(matching)} matching tensor weights from v5 backup.", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SemiconDaAIRv6CompositeLoss().to(dev)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    best_psnr = 0.0
    best_ssim = 0.0
    best_lpips = 1.0

    os.makedirs("checkpoints/v6", exist_ok=True)
    ckpt_path = "checkpoints/v6/semicon_daair_v6_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for lq, gt in train_loader:
            lq, gt = lq.to(dev), gt.to(dev)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(dev.type == "cuda")):
                pred, extras = model(lq, return_extras=True)
                loss, _ = criterion(pred, gt, moe_gates=extras.get("moe_gates"))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            if not torch.isnan(loss):
                total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        psnr_list, ssim_list, mae_list, lpips_list = [], [], [], []
        with torch.no_grad():
            for lq, gt in val_loader:
                lq, gt = lq.to(dev), gt.to(dev)
                pred = model(lq)
                p_np = pred.squeeze().cpu().numpy()
                g_np = gt.squeeze().cpu().numpy()
                m = evaluate_metrics_full(p_np, g_np, device=device)

                psnr_list.append(m["psnr"])
                ssim_list.append(m["ssim"])
                mae_list.append(m["mae"])
                lpips_list.append(m["lpips"])

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        val_mae = float(np.mean(mae_list))
        val_lpips = float(np.mean(lpips_list))

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] (LR: {current_lr:.6f}) - Loss: {avg_loss:.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f} | LPIPS: {val_lpips:.4f}", flush=True)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_lpips = val_lpips
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "val_lpips": val_lpips
            }, ckpt_path)
            print(f"  [SAVED BEST V6 MoE] {ckpt_path} (PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f})", flush=True)

    print("\n[PRODUCTION TRAINING COMPLETE] Best model saved to:", ckpt_path, flush=True)


if __name__ == "__main__":
    train_production_v6()
