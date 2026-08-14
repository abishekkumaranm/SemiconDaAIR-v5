"""
benchmark_v5_baseline.py — Phase 1: Frozen v5 Baseline Benchmark.

Evaluates the protected champion checkpoint (checkpoints/v5_backup/semicon_daair_v5_candidate.pt)
across all 640 validation samples and saves exact benchmark results to results/baseline_v5.json.
This is the mandatory baseline metric threshold for all v6 experiments.
"""

import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v5 import build_semicon_daair_v5
from dataset import RealPairedSemiconductorDataset
from evaluation.metrics import evaluate_metrics_full
from utils.test_protection import assert_not_hidden_test_path


def benchmark_v5():
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    val_split = "splits/val.txt"
    ckpt_path = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)
    assert os.path.exists(ckpt_path), f"CRITICAL ERROR: Protected v5 checkpoint not found at {ckpt_path}!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      [PHASE 1: FROZEN SEMICONDAAIR-V5 BASELINE BENCHMARK]      ")
    print("=" * 80)
    print(f"Checkpoint Path : {ckpt_path}")
    print(f"Hardware Device : {device}")

    model = build_semicon_daair_v5(scale=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    st = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(st, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    val_ds = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=val_split, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    psnr_list, ssim_list, mae_list, rmse_list, lpips_list = [], [], [], [], []
    latencies = []

    with torch.inference_mode():
        for lq, gt in val_loader:
            lq, gt = lq.to(device), gt.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(lq)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            p_np = pred.squeeze().cpu().numpy()
            g_np = gt.squeeze().cpu().numpy()
            m = evaluate_metrics_full(p_np, g_np, device=device.type)

            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])
            mae_list.append(m["mae"])
            rmse_list.append(m["rmse"])
            lpips_list.append(m["lpips"])

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_mae = float(np.mean(mae_list))
    mean_rmse = float(np.mean(rmse_list))
    mean_lpips = float(np.mean(lpips_list))
    mean_lat = float(np.mean(latencies[10:])) if len(latencies) > 10 else float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    baseline_results = {
        "model_version": "SemiconDaAIR-v5 (Protected Champion Baseline)",
        "checkpoint": ckpt_path,
        "parameters": n_params,
        "samples_evaluated": len(val_ds),
        "mean_psnr_db": round(mean_psnr, 4),
        "mean_ssim": round(mean_ssim, 4),
        "mean_mae": round(mean_mae, 4),
        "mean_rmse": round(mean_rmse, 4),
        "mean_lpips": round(mean_lpips, 4),
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "device": str(device)
    }

    os.makedirs("results", exist_ok=True)
    out_json = "results/baseline_v5.json"
    with open(out_json, "w") as f:
        json.dump(baseline_results, f, indent=2)

    print("\n" + "=" * 80)
    print("      [PHASE 1 BASELINE BENCHMARK COMPLETE]      ")
    print("=" * 80)
    print(f"Model Parameters   : {n_params:,}")
    print(f"Validation PSNR    : {mean_psnr:.4f} dB")
    print(f"Validation SSIM    : {mean_ssim:.4f}")
    print(f"Validation MAE     : {mean_mae:.4f}")
    print(f"Validation LPIPS   : {mean_lpips:.4f}")
    print(f"Mean GPU Latency   : {mean_lat:.2f} ms")
    print(f"Saved Results to   : {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_v5()
