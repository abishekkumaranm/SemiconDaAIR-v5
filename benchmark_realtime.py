"""
benchmark_realtime.py — Proves (or disproves) real-time viability.

"Real-time" only means something relative to how fast the inspection tool
produces images. This script measures sustained throughput (not a single
warm-cache inference) and compares it against realistic inspection-tool
frame rates so you can honestly state, in your PPT, whether the model keeps
up or needs batching/pipelining.

Typical wafer inspection tool acquisition rates (order-of-magnitude, varies
by tool generation and inspection mode): ~5-30 images/sec for high-res
optical/e-beam review tools. Use --target_fps to compare against whatever
number is realistic for your assumed tool.
"""
import argparse
import time

import numpy as np
import torch

from model import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="checkpoints/restorenet.pt")
    p.add_argument("--size", type=str, default="base", choices=["tiny", "base"])
    p.add_argument("--input_res", type=int, default=128,
                    help="input (low-res) side length, e.g. 128 for 256->128 pairs")
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--n_measure", type=int, default=100)
    p.add_argument("--target_fps", type=float, default=10.0,
                    help="Assumed inspection tool acquisition rate to compare against.")
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(scale=2, size=args.size).to(device)
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    except FileNotFoundError:
        print(f"No weights at {args.weights} -- benchmarking architecture speed "
              f"with random init (valid for timing, not quality).")
    model.eval()

    dummy = torch.randn(1, 1, args.input_res, args.input_res).to(device)

    # Warmup (first few calls include CUDA kernel compilation / cache warmup)
    with torch.no_grad():
        for _ in range(args.n_warmup):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Sustained measurement
    times = []
    with torch.no_grad():
        for _ in range(args.n_measure):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)

    times = np.array(times)
    mean_ms = times.mean() * 1000
    p95_ms = np.percentile(times, 95) * 1000
    achieved_fps = 1000 / mean_ms

    print(f"\n=== Real-time throughput benchmark ({device}) ===")
    print(f"Input resolution: {args.input_res}x{args.input_res}  |  Model size: {args.size}")
    print(f"Mean latency:  {mean_ms:.2f} ms/image")
    print(f"P95 latency:   {p95_ms:.2f} ms/image")
    print(f"Sustained throughput: {achieved_fps:.1f} FPS")
    print(f"Target (assumed tool acquisition rate): {args.target_fps:.1f} FPS")

    if achieved_fps >= args.target_fps:
        margin = achieved_fps / args.target_fps
        print(f"RESULT: KEEPS UP -- {margin:.1f}x the required rate, single instance, no batching needed.")
    else:
        needed = args.target_fps / achieved_fps
        print(f"RESULT: TOO SLOW at this rate -- would need ~{needed:.1f}x speedup or "
              f"parallel instances / batched inference to keep pace.")


if __name__ == "__main__":
    main()
