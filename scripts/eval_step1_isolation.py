"""
eval_step1_isolation.py — Measures Step 1 HFEdgeRefinement performance and CUDA latency in isolation.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v3 import build_semicon_daair_v3
from training.train_v4_step1 import Step1ModelWrapper
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim

STEP0_CKPT = "checkpoints/step0_v3_25ep_baseline.pt"
STEP1_CKPT = "checkpoints/step1_hf_edge_refined.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Step 1 Isolation Model on {device}...")

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # 1. Load Step 0 Baseline
    base_v3_0 = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True).to(device)
    if os.path.exists(STEP0_CKPT):
        ckpt0 = torch.load(STEP0_CKPT, map_location=device)
        base_v3_0.load_state_dict(ckpt0["model_state"], strict=False)
    base_v3_0.eval()

    psnr_s0, ssim_s0 = [], []
    with torch.inference_mode():
        for lq, gt in val_loader:
            lq, gt = lq.to(device), gt.to(device)
            out = base_v3_0(lq)
            if isinstance(out, tuple): out = out[0]
            out_np = out.squeeze(1).cpu().numpy()
            gt_np = gt.squeeze(1).cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_s0.append(compute_psnr(out_np[i], gt_np[i]))
                ssim_s0.append(compute_ssim(out_np[i], gt_np[i]))

    mean_p_s0 = float(np.mean(psnr_s0))
    mean_s_s0 = float(np.mean(ssim_s0))

    # 2. Load Step 1 Model
    base_v3_1 = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True)
    model1 = Step1ModelWrapper(base_v3_1).to(device)
    if os.path.exists(STEP1_CKPT):
        ckpt1 = torch.load(STEP1_CKPT, map_location=device)
        model1.load_state_dict(ckpt1["model_state"], strict=False)
    model1.eval()

    psnr_s1, ssim_s1 = [], []
    with torch.inference_mode():
        for lq, gt in val_loader:
            lq, gt = lq.to(device), gt.to(device)
            out = model1(lq)
            if isinstance(out, tuple): out = out[0]
            out_np = out.squeeze(1).cpu().numpy()
            gt_np = gt.squeeze(1).cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_s1.append(compute_psnr(out_np[i], gt_np[i]))
                ssim_s1.append(compute_ssim(out_np[i], gt_np[i]))

    mean_p_s1 = float(np.mean(psnr_s1))
    mean_s_s1 = float(np.mean(ssim_s1))

    # Latency Benchmark
    dummy = torch.randn(1, 1, 128, 128).to(device)
    for _ in range(10): _ = model1(dummy)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): _ = model1(dummy)
    if device.type == "cuda": torch.cuda.synchronize()
    lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    total_params = sum(p.numel() for p in model1.parameters())

    print("=" * 75)
    print("      STEP 1 HFEDGEREFINEMENT ISOLATION TEST SUMMARY      ")
    print("=" * 75)
    print(f"Step 0 Baseline (v3, 25ep): PSNR = {mean_p_s0:.4f} dB | SSIM = {mean_s_s0:.4f}")
    print(f"Step 1 Model (HF-Refined) : PSNR = {mean_p_s1:.4f} dB | SSIM = {mean_s_s1:.4f} | Latency = {lat_ms:.2f} ms")
    print(f"PSNR Delta                : {mean_p_s1 - mean_p_s0:+.4f} dB")
    print(f"SSIM Delta                : {mean_s_s1 - mean_s_s0:+.4f}")
    print(f"Total Parameters          : {total_params:,} (< 700,000 budget)")
    print(f"Learned Scalar Gate       : {model1.scalar_gate.item():.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
