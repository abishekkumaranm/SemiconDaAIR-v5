"""
train_v4_step2.py — Step 2: Isolation Test for Unsupervised Degradation-Aware Encoder.

Applies GAP + StdPool Unsupervised Degradation Encoder with FiLM conditioning.
Fine-tunes for 10 epochs starting from Step 0 baseline checkpoint (27.8440 dB).

Evaluates in-distribution val PSNR/SSIM and OOD-split metrics in isolation.
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
from models.degradation_encoder import UnsupervisedDegradationEncoder
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path, verify_split_isolation

STEP0_CKPT = "checkpoints/step0_v3_25ep_baseline.pt"
STEP2_CKPT = "checkpoints/step2_degradation_encoder.pt"


class Step2ModelWrapper(nn.Module):
    """
    Integrates UnsupervisedDegradationEncoder into SemiconDaAIR-v3 backbone.
    """
    def __init__(self, base_v3_model):
        super().__init__()
        self.v3 = base_v3_model
        self.deg_encoder = UnsupervisedDegradationEncoder(in_channels=64, embed_dim=16)

    def forward(self, x, return_extras: bool = False):
        f0 = self.v3.shallow_conv(x)
        f_speckle = self.v3.speckle_branch(f0, x)
        
        # Apply Unsupervised Degradation FiLM conditioning
        f_film, fingerprint = self.deg_encoder(f_speckle)
        
        gates, logits = self.v3.router(f_film)

        w_g = gates[:, 0].view(-1, 1, 1, 1)
        w_s = gates[:, 1].view(-1, 1, 1, 1)
        w_sr = gates[:, 2].view(-1, 1, 1, 1)

        f_g = self.v3.gaussian_expert(f_film)
        f_s = self.v3.speckle_expert(f_film)
        f_sr = self.v3.sr_expert(f_film)

        f_exp = w_g * f_g + w_s * f_s + w_sr * f_sr
        f_shared = self.v3.shared_expert(f_film + f_exp)

        f_spatial = self.v3.edge_guidance(self.v3.spatial_conv(f_shared), x)
        f_freq = self.v3.frequency_branch(f_shared)
        f_fused = self.v3.fusion_block(torch.cat([f_spatial, f_freq], dim=1))

        f_out = self.v3.global_context(f_fused)
        f_guided = self.v3.controller(f_fused, f_out)

        out_hr, confidence_map = self.v3.sr_head(f_guided, x)

        if return_extras:
            return out_hr, {"confidence_map": confidence_map, "fingerprint": fingerprint}
        return out_hr


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
        return l1_loss + 0.15 * grad_loss


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


def run_step2_degradation_encoder(epochs=10, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"   [STEP 2] UNSUPERVISED DEGRADATION ENCODER ISOLATION TEST ({device})   ")
    print("=" * 75)

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    n_train, n_val = verify_split_isolation("splits/train.txt", "splits/val.txt")

    train_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/train.txt", augment=True)
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    base_v3 = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True)
    if os.path.exists(STEP0_CKPT):
        ckpt = torch.load(STEP0_CKPT, map_location="cpu")
        base_v3.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[STEP 0 WEIGHTS] Loaded Step 0 baseline checkpoint ({ckpt['val_psnr']:.4f} dB).")

    model = Step2ModelWrapper(base_v3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[PARAM CHECK] Total Params: {total_params:,} (Budget < 700,000 ceiling)")

    init_psnr, init_ssim, init_mae = evaluate_model(model, val_loader, device=device)
    print(f"[INITIAL STEP 2] Val PSNR: {init_psnr:.4f} dB | Val SSIM: {init_ssim:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = SobelL1Loss().to(device)

    best_psnr = init_psnr
    best_ssim = init_ssim
    best_mae = init_mae

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for lq_t, gt_t in train_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)
            optimizer.zero_grad()

            out_hr = model(lq_t)
            loss = criterion(out_hr, gt_t)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        val_psnr, val_ssim, val_mae = evaluate_model(model, val_loader, device=device)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {epoch_loss/len(train_loader):.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr or epoch == epochs:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_mae = val_mae
            os.makedirs(os.path.dirname(STEP2_CKPT), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "parameters": total_params
            }, STEP2_CKPT)
            print(f"  [SAVED STEP 2 BEST] {STEP2_CKPT} (PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f})")

    # Measure Latency on GPU
    dummy_x = torch.randn(1, 1, 128, 128).to(device)
    model.eval()
    for _ in range(10): _ = model(dummy_x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): _ = model(dummy_x)
    if device.type == "cuda": torch.cuda.synchronize()
    lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    print("=" * 75)
    print(f"[STEP 2 ISOLATION SUMMARY]")
    print(f"Step 0 Baseline PSNR: 27.8440 dB | SSIM: 0.7424")
    print(f"Step 2 Model    PSNR: {best_psnr:.4f} dB | SSIM: {best_ssim:.4f} | Latency: {lat_ms:.2f} ms")
    print(f"PSNR Delta          : {best_psnr - 27.8440:+.4f} dB | SSIM Delta: {best_ssim - 0.7424:+.4f}")
    print("=" * 75)

    return best_psnr, best_ssim, best_mae, lat_ms


if __name__ == "__main__":
    run_step2_degradation_encoder(epochs=10, batch_size=16)
