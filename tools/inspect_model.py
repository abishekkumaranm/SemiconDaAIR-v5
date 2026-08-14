"""
inspect_model.py — Actual Model Inspection & Tensor Diagnostic Tool.

Loads the actual SemiconDaAIR-v5 architecture and checkpoint:
  - Model class & version
  - Measured parameter count & trainable parameter count
  - Checkpoint disk size & key validation
  - Input/output tensor shapes and data types
  - Saves results/model_inspection.json
"""

import os
import sys
import json
import torch

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v5 import build_semicon_daair_v5
from utils.device import select_device, get_device_name


def inspect_actual_model(ckpt_path: str = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt", device_str: str = "auto"):
    device = select_device(device_str)
    print("=" * 65)
    print("           SemiconDaAIR MODEL INSPECTION ENGINE          ")
    print("=" * 65)

    # 1. Instantiate Actual Model
    model = build_semicon_daair_v5(scale=2).to(device)
    model_class = model.__class__.__name__

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. Check Checkpoint
    ckpt_exists = os.path.exists(ckpt_path)
    ckpt_size_mb = round(os.path.getsize(ckpt_path) / (1024 ** 2), 2) if ckpt_exists else 0.0

    missing_keys, unexpected_keys = [], []
    if ckpt_exists:
        st = torch.load(ckpt_path, map_location=device)
        st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
        res = model.load_state_dict(st, strict=False)
        missing_keys = res.missing_keys
        unexpected_keys = res.unexpected_keys
        print(f"Checkpoint File  : {ckpt_path} ({ckpt_size_mb} MB)")
        print(f"State Dict Match : {len(missing_keys)} missing, {len(unexpected_keys)} unexpected keys")
    else:
        print(f"[WARNING] Checkpoint not found at: {ckpt_path}")

    model.eval()

    # 3. Dummy Forward Pass
    dummy_in = torch.randn(1, 1, 128, 128, device=device)
    with torch.no_grad():
        dummy_out = model(dummy_in)

    is_finite = bool(torch.isfinite(dummy_out).all().item())

    inspection = {
        "model_name": "SemiconDaAIR-v5",
        "model_class": model_class,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint_path": ckpt_path,
        "checkpoint_exists": ckpt_exists,
        "checkpoint_size_mb": ckpt_size_mb,
        "missing_keys_count": len(missing_keys),
        "unexpected_keys_count": len(unexpected_keys),
        "expected_input_shape": list(dummy_in.shape),
        "expected_output_shape": list(dummy_out.shape),
        "dtype": str(dummy_out.dtype),
        "device": str(device),
        "dummy_output_finite": is_finite
    }

    print(f"Model Class      : {model_class}")
    print(f"Total Parameters : {total_params:,}")
    print(f"Trainable Params : {trainable_params:,}")
    print(f"Input Tensor     : {list(dummy_in.shape)} ({dummy_in.dtype})")
    print(f"Output Tensor    : {list(dummy_out.shape)} ({dummy_out.dtype})")
    print(f"Device Used      : {get_device_name(device)}")
    print(f"Finite Output    : {is_finite}")
    print("=" * 65)

    os.makedirs("results", exist_ok=True)
    with open("results/model_inspection.json", "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)

    print("Saved Inspection JSON to: results/model_inspection.json\n")
    return inspection


if __name__ == "__main__":
    inspect_actual_model()
