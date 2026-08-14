"""
benchmark.py — Comprehensive Latency & Throughput Benchmark Suite for SemiconDaAIR-v2.

Measures:
  1. Warmup runs
  2. CUDA Synchronization (torch.cuda.synchronize())
  3. Model-only Forward Latency (mean, median, p95, min, max)
  4. End-to-End Pipeline Latency (Load .npy -> Preprocess -> GPU Model -> Write .npy)
  5. Throughput (FPS)
  6. Peak VRAM Memory Consumption
"""

import os
import sys
import time
import json
import shutil
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.semicon_daair_v2 import build_semicon_daair_v2


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("      SEMICONDAAIR-V2 RIGOROUS LATENCY & THROUGHPUT BENCHMARK      ")
    print("=" * 70)
    print(f"Target Hardware Device : {device}")
    if device.type == "cuda":
        print(f"GPU Name               : {torch.cuda.get_device_name(0)}")

    # Load Model
    model = build_semicon_daair_v2(scale=2, base_channels=64).to(device)
    ckpt_path = "checkpoints/final/semicon_daair_v2_final.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 1, 128, 128, device=device, dtype=torch.float32)

    # 1. Warmup Runs
    print("\nExecuting 50 Warmup Iterations...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # 2. Model-Only Forward Latency Benchmark (1000 Iterations)
    print("Executing 1,000 Model-Only Forward Passes...")
    model_latencies_ms = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(1000):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            model_latencies_ms.append((t_end - t_start) * 1000.0)

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0

    # Statistics Calculation
    mean_model_lat = float(np.mean(model_latencies_ms))
    median_model_lat = float(np.median(model_latencies_ms))
    p95_model_lat = float(np.percentile(model_latencies_ms, 95))
    min_model_lat = float(np.min(model_latencies_ms))
    max_model_lat = float(np.max(model_latencies_ms))
    fps_model = 1000.0 / max(mean_model_lat, 1e-5)

    print("\n--- MODEL-ONLY LATENCY RESULTS ---")
    print(f"Mean Latency   : {mean_model_lat:.4f} ms")
    print(f"Median Latency : {median_model_lat:.4f} ms")
    print(f"p95 Latency    : {p95_model_lat:.4f} ms")
    print(f"Min Latency    : {min_model_lat:.4f} ms")
    print(f"Max Latency    : {max_model_lat:.4f} ms")
    print(f"Throughput     : {fps_model:.1f} FPS")
    print(f"Peak VRAM      : {peak_vram_mb:.2f} MB")

    # 3. End-to-End Pipeline Latency Benchmark (Load .npy -> Preprocess -> Model -> Save .npy)
    temp_in_dir = "results/benchmark_temp"
    os.makedirs(temp_in_dir, exist_ok=True)
    temp_file = os.path.join(temp_in_dir, "sample.npy")
    np.save(temp_file, np.random.randn(128, 128).astype(np.float32))

    print("\nExecuting 200 End-to-End File I/O + Inference Passes...")
    e2e_latencies_ms = []

    with torch.no_grad():
        for _ in range(200):
            t_start = time.perf_counter()

            # I/O Load
            arr = np.load(temp_file).astype(np.float32)
            tensor_in = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

            # Model Forward
            out_tensor = model(tensor_in)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # I/O Save
            out_arr = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
            np.save(os.path.join(temp_in_dir, "out_sample.npy"), out_arr)

            t_end = time.perf_counter()
            e2e_latencies_ms.append((t_end - t_start) * 1000.0)

    mean_e2e_lat = float(np.mean(e2e_latencies_ms))
    median_e2e_lat = float(np.median(e2e_latencies_ms))
    p95_e2e_lat = float(np.percentile(e2e_latencies_ms, 95))
    fps_e2e = 1000.0 / max(mean_e2e_lat, 1e-5)

    print("\n--- END-TO-END PIPELINE LATENCY RESULTS ---")
    print(f"Mean E2E Latency   : {mean_e2e_lat:.4f} ms")
    print(f"Median E2E Latency : {median_e2e_lat:.4f} ms")
    print(f"p95 E2E Latency    : {p95_e2e_lat:.4f} ms")
    print(f"End-to-End FPS     : {fps_e2e:.1f} FPS")

    results = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "model_only_latency_ms": {
            "mean": mean_model_lat,
            "median": median_model_lat,
            "p95": p95_model_lat,
            "min": min_model_lat,
            "max": max_model_lat
        },
        "fps_model_only": fps_model,
        "end_to_end_latency_ms": {
            "mean": mean_e2e_lat,
            "median": median_e2e_lat,
            "p95": p95_e2e_lat
        },
        "fps_end_to_end": fps_e2e,
        "peak_vram_mb": peak_vram_mb
    }

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved benchmark metrics JSON to: results/benchmark_results.json")

    # Cleanup temp
    if os.path.exists(temp_in_dir):
        shutil.rmtree(temp_in_dir, ignore_errors=True)


if __name__ == "__main__":
    run_benchmark()
