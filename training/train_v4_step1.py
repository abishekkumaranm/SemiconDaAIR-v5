"""
train_v4_step1.py — Step 1: Isolation Test for HFEdgeRefinement Module.

Adds a single lightweight HFEdgeRefinement module (<45K params) with a single
learned scalar gate (initialized to 0.1) after FidelityGatedHead.

Fine-tunes for 10 epochs starting from Step 0 checkpoint (27.8440 dB).
Measures PSNR/SSIM delta and latency in isolation.
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
from models.hf_edge_refinement import HFEdgeRefinement
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path, verify_split_isolation

STEP0_CKPT = "checkpoints/step0_v3_25ep_baseline.pt"
STEP1_CKPT = "checkpoints/step1_hf_edge_refined.pt"


class Step1ModelWrapper(nn.Module):
    """
    Wraps SemiconDaAIR-v3 and appends HFEdgeRefinement with a single learned scalar gate.
    """
    def __init__(self, base_v3_model):
        super().__init__()
        self.v3 = base_v3_model
        self.hf_refinement = HFEdgeRefinement(in_channels=64, out_channels=1, hidden_channels=32)
        # Single learned scalar gate initialized to 0.1
        self.scalar_gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x, return_extras: bool = False):
        if hasattr(self.v3, "forward") and return_extras:
            y_base, extras = self.v3(x, return_extras=True)
            f_guided = extras.get("f_guided", None)
        else:
            y_base = self.v3(x)
            f_guided = None

        # If f_guided not returned in extras, extract features
        if f_guided is None:
            f0 = self.v3.shallow_conv(x)
            f_speckle = self.v3.speckle_branch(f0, x)
            f_film, _ = self.v3.fingerprint_encoder(f_speckle)
            gates, _ = self.v3.router(f_film)
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
            # Upsample f_guided to match y_base spatial resolution
            f_guided = F.interpolate(f_guided, scale_factor=self.v3.scale, mode="bilinear", align_corners=False)

        y_hf, g_hf, r_hf = self.hf_refinement(y_base, f_guided)
        y_final = y_base + self.scalar_gate * (g_hf * r_hf)

        if return_extras:
            return y_final, {"scalar_gate": self.scalar_gate.item(), "g_hf": g_hf}
        return y_final


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


def run_step1_hf_refinement(epochs=10, batch_size=16, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"   [STEP 1] HFEdgeRefinement ISOLATION TEST ({device})   ")
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

    model = Step1ModelWrapper(base_v3).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    hf_params = sum(p.numel() for p in model.hf_refinement.parameters()) + 1
    print(f"[PARAM CHECK] Total Params: {total_params:,} | HFEdgeRefinement Params: {hf_params:,} (Budget < 700,000 total)")

    init_psnr, init_ssim, init_mae = evaluate_model(model, val_loader, device=device)
    print(f"[INITIAL STEP 1] Val PSNR: {init_psnr:.4f} dB | Val SSIM: {init_ssim:.4f}")

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
        sg_val = model.scalar_gate.item()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] (ScalarGate: {sg_val:.4f}) - Loss: {epoch_loss/len(train_loader):.5f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr or epoch == epochs:
            best_psnr = val_psnr
            best_ssim = val_ssim
            best_mae = val_mae
            os.makedirs(os.path.dirname(STEP1_CKPT), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "parameters": total_params
            }, STEP1_CKPT)
            print(f"  [SAVED STEP 1 BEST] {STEP1_CKPT} (PSNR: {val_psnr:.4f} dB, SSIM: {val_ssim:.4f})")

    # Measure Latency on GPU
    dummy_x = torch.randn(1, 1, 128, 128).to(device)
    model.eval()
    for _ in range(10): _ = model(dummy_x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): _ = model(dummy_x)
    torch.cuda.synchronize()
    latency_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    print("=" * 75)
    print(f"[STEP 1 ISOLATION SUMMARY]")
    print(f"Step 0 Baseline PSNR: 27.8440 dB | SSIM: 0.7424")
    print(f"Step 1 Model    PSNR: {best_psnr:.4f} dB | SSIM: {best_ssim:.4f} | Latency: {latency_ms:.2f} ms")
    print(f"PSNR Delta: {best_psnr - 27.8440:+.4f} dB | SSIM Delta: {best_ssim - 0.7424:+.4f}")
    print("=" * 75)

    return best_psnr, best_ssim, best_mae, latency_ms


if __name__ == "__main__":
    run_step1_hf_refinement(epochs=10, batch_size=16)
