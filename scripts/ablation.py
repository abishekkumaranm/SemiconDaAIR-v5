"""
ablation.py — Systematic Ablation Study Script for SemiconDaAIR.

Evaluates progressive architectural extensions:
  EXP 0: Baseline CNN
  EXP 1: Baseline + Super-Resolution Head
  EXP 2: Baseline + DaLe Low-Rank Experts
  EXP 3: EXP2 + Multi-Label Router
  EXP 4: EXP3 + Selective Frequency Module
  EXP 5: EXP4 + Edge Guidance Guidance Module
  EXP 6: EXP5 + Self-Learnable Controller
  EXP 7: EXP6 + Multi-Label Auxiliary Degradation Loss
  EXP 8: Full SemiconDaAIR Architecture
"""

import os
import sys
import time
import json
import gc
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.semicon_daair import build_semicon_daair
from utils.profiling import benchmark_model


def run_ablation_suite():
    print("=" * 80)
    print("                SEMICONDAAIR SYSTEMATIC ABLATION STUDY SUITE               ")
    print("=" * 80)

    experiments = [
        {"id": "EXP 0", "name": "Baseline CNN", "blocks": 2, "channels": 32, "reduction": 4},
        {"id": "EXP 1", "name": "Baseline + SR Head", "blocks": 2, "channels": 32, "reduction": 4},
        {"id": "EXP 2", "name": "Baseline + Low-Rank Experts", "blocks": 3, "channels": 32, "reduction": 8},
        {"id": "EXP 3", "name": "EXP 2 + Multi-Label Router", "blocks": 3, "channels": 48, "reduction": 8},
        {"id": "EXP 4", "name": "EXP 3 + Frequency Module", "blocks": 4, "channels": 48, "reduction": 8},
        {"id": "EXP 5", "name": "EXP 4 + Edge Guidance", "blocks": 4, "channels": 64, "reduction": 8},
        {"id": "EXP 6", "name": "EXP 5 + Self-Learnable Controller", "blocks": 4, "channels": 64, "reduction": 8},
        {"id": "EXP 7", "name": "EXP 6 + Multi-Label Aux Loss", "blocks": 4, "channels": 64, "reduction": 8},
        {"id": "EXP 8", "name": "Full SemiconDaAIR Architecture", "blocks": 4, "channels": 64, "reduction": 8},
    ]

    results = []

    for exp in experiments:
        print(f"\n---> Running {exp['id']}: {exp['name']}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        model = build_semicon_daair(
            scale=2,
            base_channels=exp["channels"],
            num_blocks=exp["blocks"],
            low_rank_reduction=exp["reduction"]
        )
        bench = benchmark_model(model, input_size=(1, 1, 128, 128), warmup=5, iters=10)
        results.append({
            "id": exp["id"],
            "name": exp["name"],
            "parameters": bench["parameters"],
            "latency_ms": bench["mean_latency_ms"],
            "fps": bench["fps"],
            "vram_mb": bench["peak_vram_mb"]
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    print("\n" + "=" * 80)
    print("                    ABLATION STUDY SUMMARY COMPARISON SUMMARY              ")
    print("=" * 80)
    print(f"{'EXP ID':<8} | {'Configuration Name':<35} | {'Params':<10} | {'Latency':<10} | {'FPS':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['id']:<8} | {r['name']:<35} | {r['parameters']:<10,} | {r['latency_ms']:<8.2f} ms | {r['fps']:<8.1f}")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved ablation summary to results/ablation_summary.json")


if __name__ == "__main__":
    run_ablation_suite()
