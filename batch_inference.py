"""
batch_inference.py — Batch Directory Inference Engine.

Command:
  python batch_inference.py --input_dir test_images --output_dir outputs/restored --device auto

Supports:
  - PNG, JPG, JPEG, TIFF, TIF, NPY
  - Processes every image in input_dir
  - Preserves filenames exactly
  - Applies exact SemiconDaAIR-v5 restoration and 2x PixelShuffle expansion
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

sys_path = os.path.abspath(os.path.dirname(__file__))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v5 import build_semicon_daair_v5
from utils.device import select_device, get_device_name
from utils.preprocessing import load_image_exact, save_image_exact


def run_batch_inference(input_dir: str, output_dir: str, ckpt_path: str = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt", device_str: str = "auto"):
    device = select_device(device_str)

    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' does not exist!", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    supported_exts = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_exts) and not f.startswith("._")]
    all_files.sort()

    print("=" * 70)
    print("           SemiconDaAIR-v5 BATCH DIRECTORY INFERENCE ENGINE           ")
    print("=" * 70)
    print(f"Input Directory  : {input_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Checkpoint File  : {ckpt_path}")
    print(f"Device Selected  : {get_device_name(device)}")
    print(f"Total Samples    : {len(all_files)}")
    print("-" * 70)

    model = build_semicon_daair_v5(scale=2).to(device)
    if os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device)
        st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
        model.load_state_dict(st, strict=True)
    model.eval()

    latencies = []

    with torch.inference_mode():
        for idx, fname in enumerate(all_files):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            lq_np = load_image_exact(in_path)
            lq_tensor = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            out_tensor = model(lq_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)

            pred_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
            save_image_exact(pred_np, out_path)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(all_files):
                pct = ((idx + 1) / len(all_files)) * 100.0
                curr_mean_lat = float(np.mean(latencies))
                print(f"  [BATCH PROGRESS] {idx + 1}/{len(all_files)} ({pct:.1f}%) | Mean Latency: {curr_mean_lat:.2f} ms/sample", flush=True)

    print("=" * 70)
    print(f"BATCH INFERENCE COMPLETE: Restored {len(all_files)} images.")
    print(f"Restored outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SemiconDaAIR-v5 Batch Directory Inference CLI")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input degraded images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save restored output images directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/v5_backup/semicon_daair_v5_candidate.pt", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    args = parser.parse_args()

    run_batch_inference(args.input_dir, args.output_dir, args.checkpoint, args.device)
