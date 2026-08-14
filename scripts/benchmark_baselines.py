"""
benchmark_baselines.py — Clean End-to-End Evaluation & Benchmark Suite for Competitor Baselines.

Evaluates on the exact 640 validation split (splits/val.txt):
  1. Bicubic Interpolation
  2. SRCNN
  3. EDSR-Light
  4. SwinIR-Light
  5. SemiconDaAIR

Timing strictly includes:
  disk loading (.npy) -> numpy preprocessing -> CPU-to-GPU -> forward pass -> GPU-to-CPU -> output saving (.npy)
"""

import os
import sys
import time
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.baselines import BicubicBaseline, SRCNN, EDSRLight, SwinIRLight
from models.semicon_daair import build_semicon_daair
from utils.metrics import compute_psnr, compute_ssim


def evaluate_end_to_end(model, model_name, val_files, sample_limit=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    out_dir = f"results/baseline_outputs/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())

    psnr_list, ssim_list = [], []
    latencies_ms = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Limit to sample_limit for quick reproducible baseline comparison
    eval_files = val_files[:sample_limit]

    with torch.no_grad():
        for fname in eval_files:
            t_start = time.time()

            # 1. Disk Loading (.npy)
            lq_path = os.path.join(lq_dir, fname)
            gt_path = os.path.join(gt_dir, fname)
            
            lq_np = np.load(lq_path).astype(np.float32)
            gt_np = np.load(gt_path).astype(np.float32)

            # 2. CPU -> GPU Transfer
            lq_tensor = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)

            # 3. Model Inference
            out_tensor = model(lq_tensor)

            # 4. GPU -> CPU Transfer
            pred_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()

            # 5. Output Saving (.npy)
            save_path = os.path.join(out_dir, fname)
            np.save(save_path, pred_np)

            t_end = time.time()
            latencies_ms.append((t_end - t_start) * 1000.0)

            # Quality metrics
            psnr_list.append(compute_psnr(pred_np, gt_np))
            ssim_list.append(compute_ssim(pred_np, gt_np))

    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_lat = float(np.mean(latencies_ms))
    fps = 1000.0 / max(mean_lat, 1e-5)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0

    return {
        "model_name": model_name,
        "parameters": n_params,
        "psnr_db": mean_psnr,
        "ssim": mean_ssim,
        "end_to_end_latency_ms": mean_lat,
        "fps": fps,
        "peak_vram_mb": peak_vram_mb
    }


def run_baseline_benchmark():
    val_split_file = "splits/val.txt"
    if not os.path.exists(val_split_file):
        raise RuntimeError("Validation split file splits/val.txt not found. Run scripts/create_splits.py first.")

    with open(val_split_file, "r") as f:
        val_files = [line.strip() for line in f if line.strip()]

    print("=" * 80)
    print(f"       BENCHMARKING BASELINE MODELS ON {len(val_files)} VALIDATION SAMPLES       ")
    print("=" * 80)

    models_to_test = [
        ("Bicubic", BicubicBaseline(scale=2)),
        ("SRCNN", SRCNN(scale=2)),
        ("EDSR-Light", EDSRLight(scale=2, num_blocks=4, num_feats=32)),
        ("SwinIR-Light", SwinIRLight(scale=2, embed_dim=32, num_blocks=2)),
        ("SemiconDaAIR", build_semicon_daair(scale=2, base_channels=64, num_blocks=4))
    ]

    results = []
    for name, m in models_to_test:
        print(f"\n---> Evaluating {name}...")
        res = evaluate_end_to_end(m, name, val_files, sample_limit=50)
        results.append(res)
        print(f"     PSNR: {res['psnr_db']:.2f} dB | SSIM: {res['ssim']:.4f} | End-to-End Latency: {res['end_to_end_latency_ms']:.2f} ms ({res['fps']:.1f} FPS)")

    # 1. Export CSV
    os.makedirs("results", exist_ok=True)
    csv_path = "results/baseline_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "parameters", "psnr_db", "ssim", "end_to_end_latency_ms", "fps", "peak_vram_mb"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV comparison to: {csv_path}")

    # 2. Export JSON
    json_path = "results/baseline_comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved JSON comparison to: {json_path}")

    # 3. Export PNG Comparison Chart
    plt.figure(figsize=(10, 5))
    names = [r["model_name"] for r in results]
    psnrs = [r["psnr_db"] for r in results]
    ssims = [r["ssim"] for r in results]

    plt.subplot(1, 2, 1)
    plt.bar(names, psnrs, color="skyblue")
    plt.title("Validation PSNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.xticks(rotation=30)

    plt.subplot(1, 2, 2)
    plt.bar(names, ssims, color="lightgreen")
    plt.title("Validation SSIM")
    plt.ylabel("SSIM Score")
    plt.xticks(rotation=30)

    plt.tight_layout()
    png_path = "results/baseline_comparison.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    print(f"Saved PNG chart plot to: {png_path}")

    print("\n" + "=" * 80)
    print("                      BASELINE BENCHMARK SUMMARY TABLE                    ")
    print("=" * 80)
    print(f"{'Model Name':<15} | {'Params':<10} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'Latency':<10} | {'FPS':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_name']:<15} | {r['parameters']:<10,} | {r['psnr_db']:<10.2f} | {r['ssim']:<8.4f} | {r['end_to_end_latency_ms']:<8.2f} ms | {r['fps']:<8.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_benchmark()
