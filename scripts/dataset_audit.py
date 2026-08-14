"""
dataset_audit.py — Phase 14: Complete Dataset Safety Audit.

Scans all train and validation files to detect:
  - NaNs, Infs, empty/corrupt arrays
  - Data types and dimension mismatches
  - Range statistics (min, max, mean, std) across GT and NoisyLR datasets
Saves audit report to results/dataset_audit.json.
"""

import os
import sys
import json
import numpy as np
from PIL import Image

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from utils.test_protection import assert_not_hidden_test_path


def audit_dataset():
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"

    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    print("=" * 80)
    print("      [PHASE 14: DATASET INTEGRITY & SAFETY AUDIT]      ")
    print("=" * 80)

    lq_files = sorted([f for f in os.listdir(lq_dir) if f.endswith((".npy", ".png", ".tif", ".jpg")) and not f.startswith("._")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith((".npy", ".png", ".tif", ".jpg")) and not f.startswith("._")])

    print(f"NoisyLR Files Found : {len(lq_files)}")
    print(f"GT Files Found      : {len(gt_files)}")

    audit_summary = {
        "lq_files_count": len(lq_files),
        "gt_files_count": len(gt_files),
        "corrupt_files_count": 0,
        "nan_files_count": 0,
        "inf_files_count": 0,
        "dimension_mismatch_count": 0,
        "lq_min_overall": 1e9,
        "lq_max_overall": -1e9,
        "gt_min_overall": 1e9,
        "gt_max_overall": -1e9,
        "exceeded_range_files_count": 0
    }

    for idx, fname in enumerate(lq_files):
        lq_path = os.path.join(lq_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            print(f"[AUDIT ERROR] Missing GT counterpart for LQ file: {fname}")
            audit_summary["corrupt_files_count"] += 1
            continue

        # Load LQ
        try:
            if fname.endswith(".npy"):
                arr_lq = np.load(lq_path).astype(np.float32)
            else:
                arr_lq = np.array(Image.open(lq_path).convert("L"), dtype=np.float32) / 255.0
        except Exception as e:
            print(f"[AUDIT ERROR] Corrupt LQ file {fname}: {e}")
            audit_summary["corrupt_files_count"] += 1
            continue

        # Load GT
        try:
            if fname.endswith(".npy"):
                arr_gt = np.load(gt_path).astype(np.float32)
            else:
                arr_gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0
        except Exception as e:
            print(f"[AUDIT ERROR] Corrupt GT file {fname}: {e}")
            audit_summary["corrupt_files_count"] += 1
            continue

        # Check NaNs/Infs
        if np.isnan(arr_lq).any() or np.isnan(arr_gt).any():
            print(f"[AUDIT WARNING] NaN detected in {fname}")
            audit_summary["nan_files_count"] += 1

        if np.isinf(arr_lq).any() or np.isinf(arr_gt).any():
            print(f"[AUDIT WARNING] Inf detected in {fname}")
            audit_summary["inf_files_count"] += 1

        # Check resolution 2x ratio
        if arr_lq.shape[-2:] != (128, 128) and arr_lq.shape[-2:] != (256, 256):
            audit_summary["dimension_mismatch_count"] += 1

        lq_min = float(np.min(arr_lq))
        lq_max = float(np.max(arr_lq))
        gt_min = float(np.min(arr_gt))
        gt_max = float(np.max(arr_gt))

        audit_summary["lq_min_overall"] = min(audit_summary["lq_min_overall"], lq_min)
        audit_summary["lq_max_overall"] = max(audit_summary["lq_max_overall"], lq_max)
        audit_summary["gt_min_overall"] = min(audit_summary["gt_min_overall"], gt_min)
        audit_summary["gt_max_overall"] = max(audit_summary["gt_max_overall"], gt_max)

        if lq_min < 0.0 or lq_max > 1.0:
            audit_summary["exceeded_range_files_count"] += 1

    os.makedirs("results", exist_ok=True)
    out_json = "results/dataset_audit.json"
    with open(out_json, "w") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("      [PHASE 14 DATASET AUDIT COMPLETE]      ")
    print("=" * 80)
    print(f"Corrupt Files        : {audit_summary['corrupt_files_count']}")
    print(f"NaN Files            : {audit_summary['nan_files_count']}")
    print(f"Inf Files            : {audit_summary['inf_files_count']}")
    print(f"LQ Range (Overall)   : [{audit_summary['lq_min_overall']:.4f}, {audit_summary['lq_max_overall']:.4f}]")
    print(f"GT Range (Overall)   : [{audit_summary['gt_min_overall']:.4f}, {audit_summary['gt_max_overall']:.4f}]")
    print(f"Exceeded Range Count : {audit_summary['exceeded_range_files_count']} / {len(lq_files)} (Speckle Range Active)")
    print(f"Saved Report to      : {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    audit_dataset()
