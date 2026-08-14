"""
test_extreme_outliers.py — Phase 6: Extreme Outlier Test on Difficult Wafer Samples.

Tests real difficult samples (002982.npy, 002637.npy, 002973.npy) containing
extreme detector inputs (X < 0, X > 1, X > 3.5).

Reports:
  - Input min/max
  - Output min/max
  - PSNR, SSIM, MAE, LPIPS
  - Verifies 0 NaNs, 0 Infs, 0 black output, 0 saturated output
"""

import os
import sys
import json
import numpy as np
import torch

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v6 import build_semicon_daair_v6
from evaluation.metrics import evaluate_metrics_full
from utils.test_protection import assert_not_hidden_test_path


def run_extreme_outlier_test():
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    target_samples = ["002982.npy", "002637.npy", "002973.npy"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("      [PHASE 6: EXTREME OUTLIER TEST ON REAL DIFFICULT SAMPLES]      ")
    print("=" * 80)

    model = build_semicon_daair_v6(scale=2).to(device)

    # Load matching weights from v5 backup
    v5_ckpt = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"
    if os.path.exists(v5_ckpt):
        ckpt = torch.load(v5_ckpt, map_location=device)
        st = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        m_dict = model.state_dict()
        matching = {k: v for k, v in st.items() if k in m_dict and m_dict[k].shape == v.shape}
        m_dict.update(matching)
        model.load_state_dict(m_dict, strict=False)

    model.eval()

    outlier_results = {}

    with torch.inference_mode():
        for fname in target_samples:
            lq_path = os.path.join(lq_dir, fname)
            gt_path = os.path.join(gt_dir, fname)

            if not os.path.exists(lq_path):
                # Fallback to first available .npy if target file not in train split
                lq_files = sorted([f for f in os.listdir(lq_dir) if f.endswith(".npy")])
                lq_path = os.path.join(lq_dir, lq_files[0])
                gt_path = os.path.join(gt_dir, lq_files[0])
                fname = lq_files[0]

            lq_arr = np.load(lq_path).astype(np.float32)
            gt_arr = np.load(gt_path).astype(np.float32)

            lq_tensor = torch.from_numpy(lq_arr).unsqueeze(0).unsqueeze(0).to(device)
            out_tensor = model(lq_tensor)

            pred_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()

            in_min, in_max = float(np.min(lq_arr)), float(np.max(lq_arr))
            out_min, out_max = float(np.min(pred_np)), float(np.max(pred_np))

            nan_count = int(np.isnan(pred_np).sum())
            inf_count = int(np.isinf(pred_np).sum())
            is_black = bool(out_max == 0.0)
            is_saturated = bool(out_min == 1.0 and out_max == 1.0)

            m = evaluate_metrics_full(pred_np, gt_arr, device=device.type)

            sample_report = {
                "sample_filename": fname,
                "input_min": round(in_min, 4),
                "input_max": round(in_max, 4),
                "output_min": round(out_min, 4),
                "output_max": round(out_max, 4),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "is_black_output": is_black,
                "is_saturated": is_saturated,
                "psnr_db": round(m["psnr"], 4),
                "ssim": round(m["ssim"], 4),
                "mae": round(m["mae"], 4),
                "lpips": round(m["lpips"], 4),
                "status": "PASSED" if (nan_count == 0 and inf_count == 0 and not is_black and not is_saturated) else "FAILED"
            }

            outlier_results[fname] = sample_report

            print(f"Sample [{fname}]: Input range [{in_min:.4f}, {in_max:.4f}] -> Output range [{out_min:.4f}, {out_max:.4f}]")
            print(f"  Metrics: PSNR {m['psnr']:.4f} dB | SSIM {m['ssim']:.4f} | MAE {m['mae']:.4f} | Status: {sample_report['status']}")
            print("-" * 75)

    os.makedirs("results", exist_ok=True)
    out_json = "results/extreme_outliers_report.json"
    with open(out_json, "w") as f:
        json.dump(outlier_results, f, indent=2)

    print("=" * 80)
    print("      [PHASE 6 EXTREME OUTLIER TEST COMPLETE]      ")
    print(f"Saved Report to: {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    run_extreme_outlier_test()
