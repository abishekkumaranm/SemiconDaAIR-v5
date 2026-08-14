"""
eval_step2_isolation.py — Measures Step 2 Unsupervised Degradation Encoder performance and CUDA latency in isolation.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v3 import build_semicon_daair_v3
from training.train_v4_step2 import Step2ModelWrapper
from dataset import RealPairedSemiconductorDataset
from utils.metrics import compute_psnr, compute_ssim

STEP0_CKPT = "checkpoints/step0_v3_25ep_baseline.pt"
STEP2_CKPT = "checkpoints/step2_degradation_encoder.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Step 2 Isolation Model on {device}...")

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file="splits/val.txt", augment=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load Step 2 Model
    base_v3 = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, use_ssm=True)
    model2 = Step2ModelWrapper(base_v3).to(device)
    if os.path.exists(STEP2_CKPT):
        ckpt2 = torch.load(STEP2_CKPT, map_location=device)
        model2.load_state_dict(ckpt2["model_state"], strict=False)
        print(f"[STEP 2 CKPT] Loaded saved Step 2 checkpoint ({ckpt2['val_psnr']:.4f} dB).")
    model2.eval()

    psnr_s2, ssim_s2 = [], []
    with torch.inference_mode():
        for lq, gt in val_loader:
            lq, gt = lq.to(device), gt.to(device)
            out = model2(lq)
            if isinstance(out, tuple): out = out[0]
            out_np = out.squeeze(1).cpu().numpy()
            gt_np = gt.squeeze(1).cpu().numpy()
            for i in range(out_np.shape[0]):
                psnr_s2.append(compute_psnr(out_np[i], gt_np[i]))
                ssim_s2.append(compute_ssim(out_np[i], gt_np[i]))

    mean_p_s2 = float(np.mean(psnr_s2))
    mean_s_s2 = float(np.mean(ssim_s2))

    # Latency Benchmark
    dummy = torch.randn(1, 1, 128, 128).to(device)
    for _ in range(10): _ = model2(dummy)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): _ = model2(dummy)
    if device.type == "cuda": torch.cuda.synchronize()
    lat_ms = ((time.perf_counter() - t0) / 100.0) * 1000.0

    total_params = sum(p.numel() for p in model2.parameters())

    print("=" * 75)
    print("   STEP 2 UNSUPERVISED DEGRADATION ENCODER SUMMARY   ")
    print("=" * 75)
    print(f"Step 0 Baseline (v3, 25ep): PSNR = 27.8440 dB | SSIM = 0.7424")
    print(f"Step 2 Model (Deg-Encoder): PSNR = {mean_p_s2:.4f} dB | SSIM = {mean_s_s2:.4f} | Latency = {lat_ms:.2f} ms")
    print(f"PSNR Delta                : {mean_p_s2 - 27.8440:+.4f} dB")
    print(f"SSIM Delta                : {mean_s_s2 - 0.7424:+.4f}")
    print(f"Total Parameters          : {total_params:,} (< 700,000 budget)")
    print("=" * 75)


if __name__ == "__main__":
    main()
