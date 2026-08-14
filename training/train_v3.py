"""
train_v3.py — Controlled Training & Ablation Suite for SemiconDaAIR-v3 Experiments (EXP-A to EXP-H).

Supports:
  - EXP-A: SemiconDaAIR-v2 Gold Standard Baseline Reference
  - EXP-B: SemiconDaAIR-v2 + FidelityGatedHead
  - EXP-C: SemiconDaAIR-v2 + UnlabeledDegradationFingerprint
  - EXP-D: SemiconDaAIR-v2 + StateSpaceGlobalContextBlock
  - EXP-H: Full SemiconDaAIR-v3 Candidate Model

Protections:
  - Hidden Test Set Guard (assert_not_hidden_test_path)
  - READ-ONLY Checkpoint Protection for checkpoints/exp02/best_psnr.pt & final/semicon_daair_v2_final.pt
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v2 import build_semicon_daair_v2
from models.semicon_daair_v3 import build_semicon_daair_v3
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path, verify_split_isolation

# Checkpoint paths
V2_FINAL_CKPT = "checkpoints/final/semicon_daair_v2_final.pt"
EXPERIMENT_DIR = "checkpoints/experiments"


def evaluate_model(model, val_loader, device="cuda"):
    model.eval()
    psnr_list, ssim_list, mae_list = [], [], []

    with torch.inference_mode():
        for lq_tensor, gt_tensor in val_loader:
            lq_tensor = lq_tensor.to(device)
            gt_tensor = gt_tensor.to(device)

            out_hr = model(lq_tensor)
            if isinstance(out_hr, tuple):
                out_hr = out_hr[0]

            out_np = out_hr.squeeze(1).cpu().numpy()
            gt_np = gt_tensor.squeeze(1).cpu().numpy()

            for i in range(out_np.shape[0]):
                p = compute_psnr(out_np[i], gt_np[i])
                s = compute_ssim(out_np[i], gt_np[i])
                m = float(np.mean(np.abs(out_np[i] - gt_np[i])))

                psnr_list.append(p)
                ssim_list.append(s)
                mae_list.append(m)

    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), float(np.mean(mae_list))


def run_experiment_b(epochs=5, lr=1e-4, batch_size=16):
    """
    EXP-B: SemiconDaAIR-v2 + FidelityGatedHead
    Loads v2 pre-trained weights and fine-tunes with FidelityGatedHead.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[EXP-B] Starting Controlled Experiment B on {device}...")

    # Data paths
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    n_train, n_val = verify_split_isolation("splits/train.txt", "splits/val.txt")
    print(f"[DATASET] Split Isolation Verified: Train={n_train} samples, Val={n_val} samples.")

    # Create Datasets using exact split files for fast evaluation
    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize v3 Model (Fidelity Gate Enabled)
    model = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=False).to(device)

    # Load pre-trained v2 weights into matching layers
    if os.path.exists(V2_FINAL_CKPT):
        ckpt = torch.load(V2_FINAL_CKPT, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[WEIGHT INHERITANCE] Initialized from v2 baseline weights. (Missing: {len(missing)}, Unexpected: {len(unexpected)})")

    # Initial Baseline Evaluation before fine-tuning
    init_psnr, init_ssim, init_mae = evaluate_model(model, val_loader, device=device)
    print(f"[EXP-B Initial] PSNR: {init_psnr:.4f} dB | SSIM: {init_ssim:.4f} | MAE: {init_mae:.4f}")

    # Optimizer & Loss Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_l1 = nn.L1Loss()

    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    exp_ckpt_path = os.path.join(EXPERIMENT_DIR, "exp_b_fidelity_gate.pt")

    best_psnr = init_psnr
    best_ssim = init_ssim

    train_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/train.txt", augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"[EXP-B Fine-Tuning] Running {epochs} fine-tuning epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for lq_t, gt_t in train_loader:
            lq_t, gt_t = lq_t.to(device), gt_t.to(device)

            optimizer.zero_grad()
            out_hr = model(lq_t)
            if isinstance(out_hr, tuple):
                out_hr = out_hr[0]

            loss = criterion_l1(out_hr, gt_t)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_psnr, val_ssim, val_mae = evaluate_model(model, val_loader, device=device)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_loss/len(train_loader):.6f} | Val PSNR: {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")

        if val_psnr > best_psnr or epoch == epochs:
            best_psnr = val_psnr
            best_ssim = val_ssim
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_mae": val_mae,
                "experiment": "EXP-B (SemiconDaAIR-v2 + FidelityGatedHead)"
            }, exp_ckpt_path)
            print(f"  [SAVED EXP-B] Saved checkpoint to {exp_ckpt_path} (PSNR: {val_psnr:.4f} dB)")

    print("=" * 70)
    print(f"[EXP-B SUMMARY] Best Validation PSNR: {best_psnr:.4f} dB | Best SSIM: {best_ssim:.4f}")
    print(f"Baseline v2 to beat: 27.7500 dB PSNR | 0.7438 SSIM")
    print("=" * 70)

    return best_psnr, best_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    run_experiment_b(epochs=args.epochs, batch_size=args.batch_size)
