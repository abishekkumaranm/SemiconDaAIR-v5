"""
dataset.py — PyTorch Dataset & Data Loading Pipeline for KLA Semiconductor Restoration.

Supports:
  1. Real Paired Dataset Mode (.npy & image files): Loads ground-truth (GT 256x256 float32) and degraded inputs (NoisyLR 128x128 float32).
  2. Split-Filtered Loading: Filter by splits/train.txt (2560 samples) or splits/val.txt (640 samples).
  3. Synthetic Degradation Mode: On-the-fly synthesis of speckle noise + Gaussian noise/blur + 2x downsampling.
  4. Data Augmentations: Random 90/180/270 rotation, horizontal/vertical flips, and patch extraction.
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from degrade import degrade_image
from utils.test_protection import assert_not_hidden_test_path


def find_default_dataset():
    """Locates real KLA dataset paths on local system if available."""
    candidates = [
        r"C:\Users\HP\Downloads\dataset\train\train",
        r"C:\Users\HP\Downloads\dataset\train",
        r"C:\Users\HP\Downloads\dataset",
        "data/dataset/train",
        "data/train"
    ]
    for c in candidates:
        gt_dir = os.path.join(c, "GT")
        lq_dir = os.path.join(c, "NoisyLR")
        if os.path.exists(gt_dir) and os.path.exists(lq_dir):
            return gt_dir, lq_dir
    return None, None


class RealPairedSemiconductorDataset(Dataset):
    """
    Dataset for real paired ground-truth (GT 256x256) and low-quality (NoisyLR 128x128) semiconductor inspection images (.npy / .png).
    Supports split_file filtering for fast validation.
    """
    def __init__(self, gt_dir, lq_dir, split_file=None, patch_size=None, augment=True):
        super().__init__()
        assert_not_hidden_test_path(gt_dir)
        assert_not_hidden_test_path(lq_dir)

        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.patch_size = patch_size
        self.augment = augment

        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                valid_names = [line.strip() for line in f if line.strip()]
            self.gt_paths = [os.path.join(gt_dir, name) for name in valid_names if os.path.exists(os.path.join(gt_dir, name))]
            self.lq_paths = [os.path.join(lq_dir, name) for name in valid_names if os.path.exists(os.path.join(lq_dir, name))]
        else:
            self.gt_paths = sorted([p for p in glob.glob(os.path.join(gt_dir, "*.npy")) if not os.path.basename(p).startswith("._")])
            self.lq_paths = sorted([p for p in glob.glob(os.path.join(lq_dir, "*.npy")) if not os.path.basename(p).startswith("._")])

        if len(self.gt_paths) == 0 or len(self.lq_paths) == 0:
            raise RuntimeError(f"No valid dataset files found in GT: {gt_dir} or NoisyLR: {lq_dir}")

    def __len__(self):
        return min(len(self.gt_paths), len(self.lq_paths))

    def __getitem__(self, idx):
        gt_path = self.gt_paths[idx]
        lq_path = self.lq_paths[idx]

        if gt_path.endswith(".npy"):
            gt = np.load(gt_path).astype(np.float32)
        else:
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        if lq_path.endswith(".npy"):
            lq = np.load(lq_path).astype(np.float32)
        else:
            lq = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Optional augmentations
        if self.augment:
            if np.random.rand() < 0.5:
                gt = np.fliplr(gt).copy()
                lq = np.fliplr(lq).copy()
            if np.random.rand() < 0.5:
                gt = np.flipud(gt).copy()
                lq = np.flipud(lq).copy()
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                gt = np.rot90(gt, k=rot_k).copy()
                lq = np.rot90(lq, k=rot_k).copy()

        tensor_lq = torch.from_numpy(lq).unsqueeze(0)
        tensor_gt = torch.from_numpy(gt).unsqueeze(0)

        return tensor_lq, tensor_gt


class SyntheticSemiconductorDataset(Dataset):
    """
    On-the-fly synthetic degradation dataset using clean reference images.
    """
    def __init__(self, clean_dir, patch_size=128, scale=2, augment=True):
        super().__init__()
        assert_not_hidden_test_path(clean_dir)
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment

        self.image_paths = sorted(
            glob.glob(os.path.join(clean_dir, "*.png")) +
            glob.glob(os.path.join(clean_dir, "*.jpg")) +
            glob.glob(os.path.join(clean_dir, "*.tiff"))
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No clean images found in directory: {clean_dir}")

    def __len__(self):
        return max(len(self.image_paths) * 100, 200)

    def __getitem__(self, idx):
        path = self.image_paths[idx % len(self.image_paths)]
        img_gt_uint8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gt_uint8 is None:
            raise ValueError(f"Failed to read image at {path}")

        h, w = img_gt_uint8.shape
        if self.patch_size and h > self.patch_size and w > self.patch_size:
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            img_gt_uint8 = img_gt_uint8[top:top + self.patch_size, left:left + self.patch_size]

        img_lq_float, _ = degrade_image(img_gt_uint8, scale=self.scale)
        img_gt_float = img_gt_uint8.astype(np.float32) / 255.0

        if self.augment:
            if np.random.rand() < 0.5:
                img_gt_float = np.fliplr(img_gt_float).copy()
                img_lq_float = np.fliplr(img_lq_float).copy()
            if np.random.rand() < 0.5:
                img_gt_float = np.flipud(img_gt_float).copy()
                img_lq_float = np.flipud(img_lq_float).copy()
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img_gt_float = np.rot90(img_gt_float, k=rot_k).copy()
                img_lq_float = np.rot90(img_lq_float, k=rot_k).copy()

        tensor_lq = torch.from_numpy(img_lq_float).unsqueeze(0)
        tensor_gt = torch.from_numpy(img_gt_float).unsqueeze(0)

        return tensor_lq, tensor_gt
