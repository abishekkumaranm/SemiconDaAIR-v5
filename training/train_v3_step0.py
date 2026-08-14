"""
train_v3_step0.py — Step 0: Baseline Sanity Run for SemiconDaAIR-v3.

Runs original SemiconDaAIR-v3 as-is for 25 epochs with CosineAnnealingWarmRestarts
and standard flip/rotation data augmentation on the 3,200-sample KLA dataset.

Establishes the true baseline for v3 to verify if undertraining (12 epochs)
was the primary bottleneck.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v3 import build_semicon_daair_v3
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path, verify_split_isolation

V2_FINAL_CKPT = "checkpoints/final/semicon_daair_v2_final.pt"
STEP0_CKPT = "checkpoints/step0_v3_25ep_baseline.pt"


class SobelL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        px = F.conv2d(pred, self.sobel_x, padding=1)
        py = F.conv2d(pred, self.sobel_y, padding=1)
        tx = F.conv2d(target, self.sobel_x, padding=1)
        ty = F.conv2d(target, self.sobel_y, padding=1)
        grad_loss = self.l1(torch.sqrt(px**2 + py**2 + 1e-6), torch.sqrt(tx**2 + ty**2 + 1e-6))
        return l1_loss + 0.10 * grad_loss


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


def run_step0_baseline(epochs=25, batch_size=16, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"   [STEP 0] SEMICONDAAIR-V3 BASELINE 25-EPOCH RUN ({device})   ")
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

    model = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True).to(device)

    # Inherit pre-trained v2 weights
    if os.path.exists(V2_FINAL_CKPT):
        ckpt = torch.load(V2_FINAL_CKPT, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[WEIGHT INHERITANCE] Inherited v2 baseline weights (Missing keys: {len(missing)}).")

    init_psnr, init_ssim, init_mae = evaluate_model(model, val_loader, device=device)
    print(f"[INITIAL] Val PSNR: {init_psnr:.4f} dB | Val SSIM: {init_ssim:.4f} | Val MAE: {init_mae:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    criterion = SobelL1Loss().to(device)

    best_psnr = init_psnr
    best_ssim = init_ssim
    best_mae = init_mae

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for lq_t, gt_t in train_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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
            os.makedirs(os.path.dirname(STEP0_CKPT), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "parameters": sum(p.numel() for p in model.parameters())
            }, STEP0_CKPT)
            print(f"  [SAVED STEP 0 BEST] {STEP0_CKPT} (PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f})")

    print("=" * 75)
    print(f"[STEP 0 COMPLETED] True v3 Baseline: PSNR = {best_psnr:.4f} dB | SSIM = {best_ssim:.4f}")
    print("=" * 75)
    return best_psnr, best_ssim, best_mae


if __name__ == "__main__":
    run_step0_baseline(epochs=25, batch_size=16)
