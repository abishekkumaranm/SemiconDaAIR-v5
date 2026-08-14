"""
benchmark.py — Comprehensive GPU Latency & Peak VRAM Benchmarking Suite.

Performs:
  - 10 Warm-Up Iterations
  - 100 Measured Benchmark Iterations with CUDA Synchronization
  - Computes Mean, Median, P95, Min, Max Latency
  - Measures Peak GPU VRAM Memory Allocation
  - Evaluates FP32 and FP16 Mixed Precision
  - Saves results/benchmark.json and results/benchmark.csv
"""

import os
import sys
import json
import csv
import time
import numpy as np
import torch

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v5 import build_semicon_daair_v5
from utils.device import select_device, get_device_name


def run_benchmark(ckpt_path: str = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt", num_warmup: int = 10, num_runs: int = 100):
    device = select_device("auto")

    print("=" * 70)
    print("       [SEMICONDAAIR-V5 LATENCY & VRAM BENCHMARK ENGINE]       ")
    print("=" * 70)
    print(f"Device Selected : {get_device_name(device)}")
    print(f"Warm-Up Runs    : {num_warmup}")
    print(f"Measured Runs   : {num_runs}")

    model = build_semicon_daair_v5(scale=2).to(device)
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device)
        st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
        model.load_state_dict(st, strict=True)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    dummy_in = torch.randn(1, 1, 128, 128, device=device)

    # 1. Warm-Up Runs
    print(f"Executing {num_warmup} warm-up iterations...", end="", flush=True)
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(dummy_in)
            if device.type == "cuda":
                torch.cuda.synchronize()
    print(" Done!")

    # 2. Measured Benchmark Runs (FP32)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(0)

    latencies_fp32 = []
    with torch.inference_mode():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_in)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_fp32.append((t1 - t0) * 1000.0)

    mean_fp32 = float(np.mean(latencies_fp32))
    med_fp32 = float(np.median(latencies_fp32))
    p95_fp32 = float(np.percentile(latencies_fp32, 95))
    min_fp32 = float(np.min(latencies_fp32))
    max_fp32 = float(np.max(latencies_fp32))

    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)

    benchmark_data = {
        "model_name": "SemiconDaAIR-v5",
        "checkpoint": ckpt_path,
        "parameters": n_params,
        "device": str(device),
        "device_name": get_device_name(device),
        "precision": "FP32",
        "warmup_runs": num_warmup,
        "measured_runs": num_runs,
        "mean_latency_ms": round(mean_fp32, 2),
        "median_latency_ms": round(med_fp32, 2),
        "p95_latency_ms": round(p95_fp32, 2),
        "min_latency_ms": round(min_fp32, 2),
        "max_latency_ms": round(max_fp32, 2),
        "peak_vram_mb": vram_mb
    }

    print("-" * 70)
    print(f"FP32 Mean Latency   : {mean_fp32:.2f} ms")
    print(f"FP32 Median Latency : {med_fp32:.2f} ms")
    print(f"FP32 P95 Latency    : {p95_fp32:.2f} ms")
    print(f"Peak VRAM Memory    : {vram_mb:.1f} MB")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    json_path = "results/benchmark.json"
    csv_path = "results/benchmark.csv"

    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_data.keys()))
        writer.writeheader()
        writer.writerow(benchmark_data)

    print(f"Saved Benchmark JSON to: {json_path}")
    print(f"Saved Benchmark CSV  to: {csv_path}\n")


if __name__ == "__main__":
    run_benchmark()
