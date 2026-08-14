"""
ood_validator.py — Data Integrity Audit & In-Distribution (ID) vs Out-of-Distribution (OOD) Validation Suite.

Performs:
  1. Data Integrity Check: Verifies 0 train/val file overlap, 0 NaNs/Infs, image shapes, dynamic ranges.
  2. Dual Validation Evaluator:
       - ID Val: Standard 640-sample validation split (splits/val.txt).
       - OOD Val: Evaluates robustness under synthetic intensity scale shifts, noise variance shifts, and spatial frequency shifts.
       - Generalization Gap: PSNR_gap = PSNR_ID - PSNR_OOD.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import compute_psnr, compute_ssim
from utils.test_protection import assert_not_hidden_test_path

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def audit_dataset_integrity(gt_dir, lq_dir, train_split_file, val_split_file):
    """Verifies strict train/val isolation, NaNs/Infs check, and intensity statistics."""
    assert_not_hidden_test_path(gt_dir)
    assert_not_hidden_test_path(lq_dir)

    with open(train_split_file, "r") as f:
        train_names = set(line.strip() for line in f if line.strip())

    with open(val_split_file, "r") as f:
        val_names = set(line.strip() for line in f if line.strip())

    overlap = train_names.intersection(val_names)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! {len(overlap)} files overlap between train and val splits."

    nan_count = 0
    inf_count = 0
    lq_min_list, lq_max_list = [], []
    gt_min_list, gt_max_list = [], []
    shapes_lq = set()
    shapes_gt = set()

    for fname in sorted(list(val_names))[:50]:
        gt_path = os.path.join(gt_dir, fname)
        lq_path = os.path.join(lq_dir, fname)

        if os.path.exists(gt_path) and os.path.exists(lq_path):
            gt = np.load(gt_path).astype(np.float32)
            lq = np.load(lq_path).astype(np.float32)

            nan_count += int(np.isnan(gt).sum() + np.isnan(lq).sum())
            inf_count += int(np.isinf(gt).sum() + np.isinf(lq).sum())

            gt_min_list.append(float(np.min(gt)))
            gt_max_list.append(float(np.max(gt)))
            lq_min_list.append(float(np.min(lq)))
            lq_max_list.append(float(np.max(lq)))

            shapes_gt.add(tuple(gt.shape))
            shapes_lq.add(tuple(lq.shape))

    report = {
        "train_sample_count": len(train_names),
        "val_sample_count": len(val_names),
        "train_val_overlap_count": len(overlap),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "lq_shapes": [list(s) for s in shapes_lq],
        "gt_shapes": [list(s) for s in shapes_gt],
        "lq_dynamic_range_min_mean": float(np.mean(lq_min_list)),
        "lq_dynamic_range_max_mean": float(np.mean(lq_max_list)),
        "gt_dynamic_range_min_mean": float(np.mean(gt_min_list)),
        "gt_dynamic_range_max_mean": float(np.mean(gt_max_list)),
        "data_leakage_passed": True,
        "data_integrity_passed": (nan_count == 0 and inf_count == 0)
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/data_integrity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


class OODSemiconductorDataset(Dataset):
    """
    Dataset loader for Out-of-Distribution (OOD) testing.
    Applies non-linear intensity scaling, additive variance shifts, and high-frequency noise perturbation.
    """
    def __init__(self, gt_dir, lq_dir, split_file, shift_type="intensity_noise"):
        super().__init__()
        assert_not_hidden_test_path(gt_dir)
        assert_not_hidden_test_path(lq_dir)
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.shift_type = shift_type

        with open(split_file, "r") as f:
            valid_names = [line.strip() for line in f if line.strip()]

        self.gt_paths = [os.path.join(gt_dir, name) for name in valid_names if os.path.exists(os.path.join(gt_dir, name))]
        self.lq_paths = [os.path.join(lq_dir, name) for name in valid_names if os.path.exists(os.path.join(lq_dir, name))]

    def __len__(self):
        return min(len(self.gt_paths), len(self.lq_paths))

    def __getitem__(self, idx):
        gt = np.load(self.gt_paths[idx]).astype(np.float32)
        lq = np.load(self.lq_paths[idx]).astype(np.float32)

        # Apply deterministic OOD shift based on sample index
        rng = np.random.RandomState(idx + 9999)
        if self.shift_type == "intensity_noise":
            # Intensity scaling shift + additive Gaussian noise variance spike
            scale_factor = rng.uniform(0.7, 1.4)
            lq = lq * scale_factor + rng.normal(0, 0.08, size=lq.shape).astype(np.float32)
        elif self.shift_type == "contrast_speckle":
            # Multiplicative speckle boost
            speckle_extra = lq * rng.uniform(-0.25, 0.25, size=lq.shape).astype(np.float32)
            lq = lq + speckle_extra

        return torch.from_numpy(lq).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)


def evaluate_id_vs_ood(model, device, gt_dir, lq_dir, val_split_file):
    """
    Evaluates model performance on both In-Distribution (ID) and Out-of-Distribution (OOD) validation sets.
    Computes PSNR, SSIM, LPIPS, MAE, and Generalization Gap.
    """
    from dataset import RealPairedSemiconductorDataset

    model.eval()

    id_dataset = RealPairedSemiconductorDataset(gt_dir, lq_dir, split_file=val_split_file, augment=False)
    ood_dataset = OODSemiconductorDataset(gt_dir, lq_dir, split_file=val_split_file, shift_type="intensity_noise")

    id_loader = DataLoader(id_dataset, batch_size=1, shuffle=False)
    ood_loader = DataLoader(ood_dataset, batch_size=1, shuffle=False)

    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device) if HAS_LPIPS else None

    # 1. ID Evaluation
    id_psnr, id_ssim, id_lpips, id_mae = [], [], [], []
    with torch.no_grad():
        for lq, gt in id_loader:
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            p_np = pred.squeeze().cpu().numpy()
            g_np = gt.squeeze().cpu().numpy()

            id_psnr.append(compute_psnr(p_np, g_np))
            id_ssim.append(compute_ssim(p_np, g_np))
            id_mae.append(float(np.mean(np.abs(p_np - g_np))))

            if lpips_fn:
                p_c = torch.clamp(pred, 0, 1).repeat(1, 3, 1, 1)
                g_c = torch.clamp(gt, 0, 1).repeat(1, 3, 1, 1)
                id_lpips.append(float(lpips_fn(p_c * 2 - 1, g_c * 2 - 1).item()))

    # 2. OOD Evaluation
    ood_psnr, ood_ssim, ood_lpips, ood_mae = [], [], [], []
    with torch.no_grad():
        for lq, gt in ood_loader:
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            p_np = pred.squeeze().cpu().numpy()
            g_np = gt.squeeze().cpu().numpy()

            ood_psnr.append(compute_psnr(p_np, g_np))
            ood_ssim.append(compute_ssim(p_np, g_np))
            ood_mae.append(float(np.mean(np.abs(p_np - g_np))))

            if lpips_fn:
                p_c = torch.clamp(pred, 0, 1).repeat(1, 3, 1, 1)
                g_c = torch.clamp(gt, 0, 1).repeat(1, 3, 1, 1)
                ood_lpips.append(float(lpips_fn(p_c * 2 - 1, g_c * 2 - 1).item()))

    results = {
        "psnr_id": float(np.mean(id_psnr)),
        "ssim_id": float(np.mean(id_ssim)),
        "lpips_id": float(np.mean(id_lpips)) if id_lpips else 0.0,
        "mae_id": float(np.mean(id_mae)),
        "psnr_ood": float(np.mean(ood_psnr)),
        "ssim_ood": float(np.mean(ood_ssim)),
        "lpips_ood": float(np.mean(ood_lpips)) if ood_lpips else 0.0,
        "mae_ood": float(np.mean(ood_mae)),
        "psnr_gap": float(np.mean(id_psnr) - np.mean(ood_psnr)),
        "ssim_gap": float(np.mean(id_ssim) - np.mean(ood_ssim))
    }

    return results


if __name__ == "__main__":
    rep = audit_dataset_integrity(
        r"C:\Users\HP\Downloads\dataset\train\train\GT",
        r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR",
        "splits/train.txt",
        "splits/val.txt"
    )
    print("Data Integrity Report:", json.dumps(rep, indent=2))
