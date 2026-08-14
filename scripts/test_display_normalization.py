"""
test_display_normalization.py — Test robust display normalization for browser visualization.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.semicon_daair_v2 import build_semicon_daair_v2

sample_lq = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR\000040.npy"
sample_gt = r"C:\Users\HP\Downloads\dataset\train\train\GT\000040.npy"

if os.path.exists(sample_lq) and os.path.exists(sample_gt):
    lq_np = np.load(sample_lq).astype(np.float32)
    gt_np = np.load(sample_gt).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_semicon_daair_v2(scale=2, base_channels=64).to(device)
    ckpt = torch.load("checkpoints/final/semicon_daair_v2_final.pt", map_location=device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tensor_in = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        out_tensor = model(tensor_in)
    out_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()

    print(f"Raw Model Output Min/Max: [{np.min(out_np):.4f}, {np.max(out_np):.4f}] | Mean: {np.mean(out_np):.4f} | Std: {np.std(out_np):.4f}")
    print(f"Raw GT Min/Max: [{np.min(gt_np):.4f}, {np.max(gt_np):.4f}] | Mean: {np.mean(gt_np):.4f} | Std: {np.std(gt_np):.4f}")

    # Method 1: Naive min-max
    naive_norm = (out_np - np.min(out_np)) / (np.max(out_np) - np.min(out_np) + 1e-5)

    # Method 2: Robust Percentile / GT-aligned normalization (1st to 99th percentile)
    p_low, p_high = np.percentile(out_np, 0.5), np.percentile(out_np, 99.5)
    robust_norm = np.clip((out_np - p_low) / (p_high - p_low + 1e-5), 0, 1)

    print(f"Naive Norm Range: [{np.min(naive_norm):.4f}, {np.max(naive_norm):.4f}]")
    print(f"Robust Percentile Range: [{np.min(robust_norm):.4f}, {np.max(robust_norm):.4f}]")

    out_dir = "results/display_test"
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((naive_norm * 255.0).astype(np.uint8)).save(os.path.join(out_dir, "naive_display.png"))
    Image.fromarray((robust_norm * 255.0).astype(np.uint8)).save(os.path.join(out_dir, "robust_display.png"))
    Image.fromarray((np.clip(gt_np, 0, 1) * 255.0).astype(np.uint8)).save(os.path.join(out_dir, "gt_display.png"))

    print("Saved naive_display.png, robust_display.png, and gt_display.png to results/display_test/")
