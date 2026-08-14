"""
utils/metrics.py — Alias module importing root metrology & quality metrics.
"""

import os
import sys
import importlib.util

# Safely load root metrics.py
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_root_metrics_path = os.path.join(_root_dir, "metrics.py")

_spec = importlib.util.spec_from_file_location("root_metrics", _root_metrics_path)
_root_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_metrics)

compute_psnr = _root_metrics.compute_psnr
compute_ssim = _root_metrics.compute_ssim
compute_cd_error = _root_metrics.compute_cd_error
compute_overlay_error = _root_metrics.compute_overlay_error
compute_ler_fidelity = _root_metrics.compute_ler_fidelity
evaluate_sample = _root_metrics.evaluate_sample

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_cd_error",
    "compute_overlay_error",
    "compute_ler_fidelity",
    "evaluate_sample",
]
