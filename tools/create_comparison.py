"""
create_comparison.py — Visual Side-by-Side Comparison Generator.

Features:
  - Generates outputs/comparison/side_by_side.png
  - If GT exists, creates 3-column panel: Degraded Input | Restored Output | Ground Truth
  - Generates absolute pixel difference map outputs/comparison/difference.png if GT exists.
"""

import os
import sys
import cv2
import numpy as np

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from utils.preprocessing import load_image_exact


def create_comparison(input_path: str, restored_path: str, gt_path: str = None, output_dir: str = "outputs/comparison"):
    os.makedirs(output_dir, exist_ok=True)

    in_np = load_image_exact(input_path)
    res_np = load_image_exact(restored_path)

    # Convert to 8-bit uint8 for visual panel
    def to_uint8(arr):
        if arr.min() < 0.0 or arr.max() > 1.0:
            p_min, p_max = np.percentile(arr, (0.5, 99.5))
            if p_max > p_min:
                norm = np.clip((arr - p_min) / (p_max - p_min), 0.0, 1.0)
            else:
                norm = np.clip(arr, 0.0, 1.0)
        else:
            norm = np.clip(arr, 0.0, 1.0)
        return (norm * 255.0).round().astype(np.uint8)

    in_u8 = to_uint8(in_np)
    res_u8 = to_uint8(res_np)

    # Match height by resizing degraded input for visual comparison
    h_out, w_out = res_u8.shape[:2]
    in_u8_resized = cv2.resize(in_u8, (w_out, h_out), interpolation=cv2.INTER_NEAREST)

    has_gt = bool(gt_path) and os.path.exists(gt_path)

    if has_gt:
        gt_np = load_image_exact(gt_path)
        gt_u8 = to_uint8(gt_np)
        if gt_u8.shape != (h_out, w_out):
            gt_u8 = cv2.resize(gt_u8, (w_out, h_out), interpolation=cv2.INTER_CUBIC)

        panel = np.hstack([in_u8_resized, res_u8, gt_u8])

        # Difference map
        diff = np.abs(res_np.astype(np.float32) - gt_np.astype(np.float32))
        diff_u8 = cv2.applyColorMap((np.clip(diff / (diff.max() + 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

        diff_path = os.path.join(output_dir, "difference.png")
        cv2.imwrite(diff_path, diff_u8)
        print(f"Difference map saved to: {diff_path}")
    else:
        panel = np.hstack([in_u8_resized, res_u8])

    side_path = os.path.join(output_dir, "side_by_side.png")
    cv2.imwrite(side_path, panel)

    print(f"Side-by-side comparison saved to: {side_path}")
    return side_path


if __name__ == "__main__":
    in_file = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR\000000.npy"
    res_file = r"outputs/sample_000000_restored.png"
    gt_file = r"C:\Users\HP\Downloads\dataset\train\train\GT\000000.npy"

    create_comparison(in_file, res_file, gt_file)
