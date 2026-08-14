"""
train_v3_full.py — Dedicated Full Training Pipeline for SemiconDaAIR-v3 Production Model.

Trains full SemiconDaAIR-v3 (FidelityGatedHead + Fingerprint + SSM Global Context + RCAB)
with composite L1 + Sobel Edge + FFT Frequency loss to outperform SemiconDaAIR-v2 baseline (> 28.0 dB PSNR).

Saves candidate checkpoint to checkpoints/final/semicon_daair_v3_candidate.pt.
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
V3_CANDIDATE_CKPT = "checkpoints/final/semicon_daair_v3_candidate.pt"


class CompositeRestorationLoss(nn.Module):
    """
    Composite Loss for Semiconductor Restoration:
      L_total = L1 + 0.1 * L_sobel_edge + 0.05 * L_fft_frequency
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

        # Sobel Kernels for edge loss
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def edge_loss(self, pred, target):
        px = F.conv2d(pred, self.sobel_x, padding=1)
        py = F.conv2d(pred, self.sobel_y, padding=1)
        tx = F.conv2d(target, self.sobel_x, padding=1)
        ty = F.conv2d(target, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(px**2 + py**2 + 1e-6)
        target_grad = torch.sqrt(tx**2 + ty**2 + 1e-6)
        return self.l1(pred_grad, target_grad)

    def frequency_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        return self.l1(torch.abs(pred_fft), torch.abs(target_fft))

    def forward(self, pred, target):
        l1_val = self.l1(pred, target)
        edge_val = self.edge_loss(pred, target)
        freq_val = self.frequency_loss(pred, target)
        return l1_val + 0.10 * edge_val + 0.05 * freq_val


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
                p = compute_psnr(out_np[i], gt_np[i])
                s = compute_ssim(out_np[i], gt_np[i])
                m = float(np.mean(np.abs(out_np[i] - gt_np[i])))
                psnr_list.append(p)
                ssim_list.append(s)
                mae_list.append(m)

    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), float(np.mean(mae_list))


def run_full_v3_training(epochs=15, lr=2e-4, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"   STARTING FULL SEMICONDAAIR-V3 PRODUCTION TRAINING ({device})   ")
    print("=" * 70)

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    n_train, n_val = verify_split_isolation("splits/train.txt", "splits/val.txt")
    print(f"[SECURITY CHECK] Isolated Train={n_train} samples, Val={n_val} samples.")

    # Datasets
    train_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/train.txt", augment=True)
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize Full SemiconDaAIR-v3 (Fidelity Gate + Fingerprint + SSM)
    model = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, fingerprint_dim=16, use_ssm=True).to(device)

    # Inherit matching pre-trained v2 weights
    if os.path.exists(V2_FINAL_CKPT):
        ckpt = torch.load(V2_FINAL_CKPT, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[WEIGHT INHERITANCE] Loaded pre-trained v2 baseline weights. (Missing: {len(missing)})")

    init_psnr, init_ssim, init_mae = evaluate_model(model, val_loader, device=device)
    print(f"[INITIAL BASELINE] PSNR: {init_psnr:.4f} dB | SSIM: {init_ssim:.4f} | MAE: {init_mae:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CompositeRestorationLoss().to(device)

    best_psnr = init_psnr
    best_ssim = init_ssim

    print(f"\n[TRAINING START] Training {epochs} epochs with Composite Loss...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for lq_t, gt_t in train_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)

            optimizer.zero_grad()
            out_hr = model(lq_t)
            if isinstance(out_hr, tuple):
                out_hr = out_hr[0]

            loss = criterion(out_hr, gt_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        val_psnr, val_ssim, val_mae = evaluate_model(model, val_loader, device=device)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] (LR: {current_lr:.6f}) - Loss: {epoch_loss/len(train_loader):.6f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr or epoch == epochs:
            best_psnr = val_psnr
            best_ssim = val_ssim
            os.makedirs(os.path.dirname(V3_CANDIDATE_CKPT), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "architecture": "SemiconDaAIR-v3 (Fidelity-Gated SSM)"
            }, V3_CANDIDATE_CKPT)
            print(f"  [SAVED V3 CANDIDATE] Saved candidate checkpoint to {V3_CANDIDATE_CKPT} (PSNR: {val_psnr:.4f} dB)")

    print("=" * 70)
    print(f"[V3 TRAINING COMPLETE] Final Best PSNR: {best_psnr:.4f} dB | Best SSIM: {best_ssim:.4f}")
    print("=" * 70)

    return best_psnr, best_ssim


if __name__ == "__main__":
    run_full_v3_training(epochs=12, batch_size=16)
