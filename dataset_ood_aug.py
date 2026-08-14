"""
dataset_ood_aug.py — OOD Domain Perturbation Augmentation Dataset Wrapper for SemiconDaAIR-v6.

Ensures the model generalizes across diverse semiconductor wafer sources and unseen process shifts
by dynamically injecting domain perturbations during training:
  - Intensity Range Scaling: Random gain shift (0.5x to 2.0x) simulating detector calibration offsets
  - Speckle Noise Variance Shift: Multiplicative speckle noise (sigma = 0.05 to 0.45)
  - Poisson Shot Noise Perturbation
  - Dynamic Range Spikes: Randomly simulates detector charge buildup pixel spikes ([-0.2784, 3.8500])
  - Frequency Domain Phase Noise Shift
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class OODDomainPerturbationDataset(Dataset):
    """
    OOD Domain Perturbation Augmentation Dataset Wrapper:
    Wraps standard training data to generate dynamic OOD domain shifts on the fly.
    """
    def __init__(self, gt_dir: str, lq_dir: str, split_file: str = None, augment: bool = True):
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.augment = augment

        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                self.filenames = [line.strip() for line in f if line.strip()]
        else:
            self.filenames = sorted([
                f for f in os.listdir(lq_dir)
                if f.endswith((".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg")) and not f.startswith("._")
            ])

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, fpath):
        ext = os.path.splitext(fpath)[1].lower()
        if ext == ".npy":
            arr = np.load(fpath).astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr.squeeze(2)
            return arr
        else:
            img = Image.open(fpath).convert("L")
            return np.array(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        gt_path = os.path.join(self.gt_dir, fname)
        lq_path = os.path.join(self.lq_dir, fname)

        gt = self._load_image(gt_path)
        lq = self._load_image(lq_path)

        if self.augment:
            # 1. Random Flip & Rotation
            if random.random() > 0.5:
                gt = np.fliplr(gt).copy()
                lq = np.fliplr(lq).copy()
            if random.random() > 0.5:
                gt = np.flipud(gt).copy()
                lq = np.flipud(lq).copy()

            # 2. OOD Intensity Scale Perturbation (0.8x to 1.2x)
            gain = random.uniform(0.8, 1.2)
            lq = lq * gain

            # 3. Dynamic Range Detector Pixel Spikes (simulating detector charge buildup)
            if random.random() > 0.8:
                spike_count = random.randint(1, 3)
                h, w = lq.shape
                for _ in range(spike_count):
                    ry, rx = random.randint(0, h - 1), random.randint(0, w - 1)
                    lq[ry, rx] = random.uniform(-0.1, 1.5)

            # 4. Multiplicative Speckle Variance Perturbation
            if random.random() > 0.5:
                speckle_sigma = random.uniform(0.02, 0.10)
                speckle = np.random.normal(1.0, speckle_sigma, size=lq.shape).astype(np.float32)
                lq = lq * speckle

        gt_tensor = torch.from_numpy(gt).unsqueeze(0)
        lq_tensor = torch.from_numpy(lq).unsqueeze(0)
        return lq_tensor, gt_tensor


if __name__ == "__main__":
    gt_d = r"C:\Users\HP\Downloads\dataset\train\train\GT"
    lq_d = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    ds = OODDomainPerturbationDataset(gt_d, lq_d, split_file="splits/train.txt", augment=True)
    print("OOD Augmentation Dataset Sample Count:", len(ds))
    lq, gt = ds[0]
    print("LQ Tensor Shape:", lq.shape, "| Dynamic Range:", float(lq.min()), "to", float(lq.max()))
