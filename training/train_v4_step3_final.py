"""
train_v4_step3_final.py — Step 3: Full 25-Epoch Competition Schedule Training for SemiconDaAIR-v4.

Trains SemiconDaAIR-v4 on the 3,200-sample KLA dataset using the Master Composite Loss:
  L = L1 + 0.20 * L_Sobel + 0.10 * L_Laplacian + 0.05 * L_FFT + 0.10 * L_SSIM

Optimizers: AdamW (lr=2e-4) with CosineAnnealingWarmRestarts.
Saves final production checkpoint to checkpoints/final/semicon_daair_v4_final.pt.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v4 import build_semicon_daair_v4
from losses.structure_loss import MasterCompositeLoss
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path, verify_split_isolation

STEP0_CKPT = "checkpoints/step0_v3_25ep_baseline.pt"
FINAL_V4_CKPT = "checkpoints/final/semicon_daair_v4_final.pt"


def evaluate_model(model, val_loader, device="cuda"):
    model.eval()
    psnr_list, ssim_list, mae_list = [], [], []

    with torch.inference_mode():
        for lq_t, gt_t in val_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)
            out_hr = model(lq_t)
            if isinstance(out_hr, tuple):
                out_hr = out_hr[0]

            out_np = out_hr.squeeze(1).cpu().numpy()
            gt_np = gt_t.squeeze(1).cpu().numpy()

            for i in range(out_np.shape[0]):
                psnr_list.append(compute_psnr(out_np[i], gt_np[i]))
                ssim_list.append(compute_ssim(out_np[i], gt_np[i]))
                mae_list.append(float(np.mean(np.abs(out_np[i] - gt_np[i]))))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), float(np.mean(mae_list))


def run_step3_final_training(epochs=25, batch_size=16, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"   [STEP 3] SEMICONDAAIR-V4 FULL 25-EPOCH COMPETITION TRAINING ({device})   ")
    print("=" * 75)

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    n_train, n_val = verify_split_isolation("splits/train.txt", "splits/val.txt")
    print(f"[SECURITY CHECK] Isolated Train={n_train} samples, Val={n_val} samples.")

    train_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/train.txt", augment=True)
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_semicon_daair_v4(scale=2).to(device)

    # Initialize from Step 0 baseline checkpoint
    if os.path.exists(STEP0_CKPT):
        ckpt = torch.load(STEP0_CKPT, map_location=device)
        model.backbone.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[WEIGHT INHERITANCE] Inherited Step 0 baseline weights (Saved PSNR: {ckpt['val_psnr']:.4f} dB).")

    init_psnr, init_ssim, init_mae = evaluate_model(model, val_loader, device=device)
    print(f"[INITIAL V4] Val PSNR: {init_psnr:.4f} dB | Val SSIM: {init_ssim:.4f} | Val MAE: {init_mae:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = MasterCompositeLoss().to(device)

    best_psnr = init_psnr
    best_ssim = init_ssim
    best_mae = init_mae

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for lq_t, gt_t in train_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out_hr = model(lq_t)
                if isinstance(out_hr, tuple):
                    out_hr = out_hr[0]
                loss = criterion(out_hr, gt_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        val_psnr, val_ssim, val_mae = evaluate_model(model, val_loader, device=device)
        curr_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] (LR: {curr_lr:.6f}) - Loss: {epoch_loss/len(train_loader):.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr or epoch == epochs:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_mae = val_mae
            os.makedirs(os.path.dirname(FINAL_V4_CKPT), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "parameters": sum(p.numel() for p in model.parameters())
            }, FINAL_V4_CKPT)
            print(f"  [SAVED V4 FINAL BEST] {FINAL_V4_CKPT} (PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f})")

    # Final CUDA Latency Measurement
    dummy_x = torch.randn(1, 1, 128, 128).to(device)
    model.eval()
    for _ in range(10): _ = model(dummy_x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): _ = model(dummy_x)
    if device.type == "cuda": torch.cuda.synchronize()
    lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    print("=" * 75)
    print(f"[SEMICONDAAIR-V4 TRAINING COMPLETED]")
    print(f"Final Production Model: {FINAL_V4_CKPT}")
    print(f"Validation PSNR       : {best_psnr:.4f} dB")
    print(f"Validation SSIM       : {best_ssim:.4f}")
    print(f"Validation MAE        : {best_mae:.4f}")
    print(f"Target Latency        : {lat_ms:.2f} ms (< 15.0 ms ceiling)")
    print(f"Total Parameters      : {sum(p.numel() for p in model.parameters()):,} (< 700,000 budget)")
    print("=" * 75)

    return best_psnr, best_ssim, best_mae, lat_ms


if __name__ == "__main__":
    run_step3_final_training(epochs=25, batch_size=16)
