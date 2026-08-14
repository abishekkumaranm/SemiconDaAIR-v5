"""
create_splits.py — Leakage-Free Reproducible Validation Split Generator.

Target:
  80% Training (~2,560 samples)
  20% Validation (~640 samples)
  Fixed Random Seed: 42

Outputs:
  splits/train.txt
  splits/val.txt
"""

import os
import glob
import random


def create_reproducible_splits(seed=42, val_ratio=0.20):
    gt_dir = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    gt_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(gt_dir, "*.npy"))])
    
    print(f"Total training samples available: {len(gt_files)}")

    random.seed(seed)
    shuffled_files = gt_files.copy()
    random.shuffle(shuffled_files)

    num_val = int(len(shuffled_files) * val_ratio)
    val_files = sorted(shuffled_files[:num_val])
    train_files = sorted(shuffled_files[num_val:])

    os.makedirs("splits", exist_ok=True)

    with open("splits/train.txt", "w") as f:
        for fname in train_files:
            f.write(f"{fname}\n")

    with open("splits/val.txt", "w") as f:
        for fname in val_files:
            f.write(f"{fname}\n")

    print(f"\n[SPLIT CREATION COMPLETED]")
    print(f"  Training Split Count   : {len(train_files)} (saved to splits/train.txt)")
    print(f"  Validation Split Count : {len(val_files)} (saved to splits/val.txt)")
    print(f"  Random Seed Used       : {seed}")


if __name__ == "__main__":
    create_reproducible_splits(seed=42, val_ratio=0.20)
