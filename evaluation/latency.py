"""
evaluation/latency.py — Rigorous GPU Latency & VRAM Memory Benchmark Profiler.

Benchmarks model inference speed across batch size 1:
  - 100 Warmup iterations (CUDA context initialization)
  - 500 Timed iterations with CUDA synchronization
  - Computes Mean, Median, Std, and P95 latency (ms)
  - Supports both FP32 and FP16 precisions
  - Measures Peak VRAM Allocation (MB)
"""

import os
import sys
import time
import json
import numpy as np
import torch

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in os.path.sys.path:
    os.path.sys.path.append(sys_path)

from models.semicon_daair_v6 import build_semicon_daair_v6


def benchmark_latency(model, input_size=(1, 1, 128, 128), precision="fp16", warmup=100, timed=500, device="cuda"):
    """
    Measures GPU latency (mean, median, std, p95) and VRAM usage.
    """
    dev = torch.device(device)
    model.to(dev).eval()

    dtype = torch.float16 if precision == "fp16" else torch.float32
    dummy_input = torch.randn(*input_size, device=dev, dtype=torch.float32)

    # 1. Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.amp.autocast("cuda", enabled=(precision == "fp16")):
                _ = model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()

    # 2. Timed Runs
    latencies = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.inference_mode():
        for _ in range(timed):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast("cuda", enabled=(precision == "fp16")):
                _ = model(dummy_input)

            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    peak_vram_mb = float(torch.cuda.max_memory_allocated(dev) / (1024 * 1024)) if dev.type == "cuda" else 0.0

    report = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU",
        "precision": precision,
        "input_size": list(input_size),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "std_latency_ms": float(np.std(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "peak_vram_mb": peak_vram_mb
    }
    return report


if __name__ == "__main__":
    m = build_semicon_daair_v6(scale=2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep_fp16 = benchmark_latency(m, precision="fp16", device=dev)
    rep_fp32 = benchmark_latency(m, precision="fp32", device=dev)

    print("=" * 70)
    print("      SEMICONDAAIR-V6 GPU LATENCY & VRAM BENCHMARK REPORT      ")
    print("=" * 70)
    print("FP16 Results:", json.dumps(rep_fp16, indent=2))
    print("FP32 Results:", json.dumps(rep_fp32, indent=2))
