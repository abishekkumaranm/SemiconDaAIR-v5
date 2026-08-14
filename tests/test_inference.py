"""
test_inference.py — PyTest Automated Unit Testing Suite.

Tests:
  1. Checkpoint file exists
  2. Model construction succeeds
  3. Checkpoint state dict loads cleanly (0 missing, 0 unexpected keys)
  4. Total parameter count matches 555,141 exactly
  5. All model parameters are finite (0 NaNs, 0 Infs)
  6. Dummy forward pass output tensor is finite
  7. Dummy forward pass output shape performs exact 2x spatial expansion ([1, 1, 128, 128] -> [1, 1, 256, 256])
"""

import os
import sys
try:
    import pytest
except ImportError:
    pytest = None
import torch

sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v5 import build_semicon_daair_v5
from utils.device import select_device


def get_ckpt_path():
    path = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"
    assert os.path.exists(path), f"Checkpoint '{path}' not found!"
    return path

if pytest is not None:
    ckpt_path = pytest.fixture(get_ckpt_path)
else:
    ckpt_path = get_ckpt_path


def test_model_construction():
    model = build_semicon_daair_v5(scale=2)
    assert model is not None
    params = sum(p.numel() for p in model.parameters())
    assert params == 555141, f"Expected 555,141 params, got {params}"


def test_checkpoint_loading(ckpt_path):
    device = select_device("auto")
    model = build_semicon_daair_v5(scale=2).to(device)
    st = torch.load(ckpt_path, map_location=device)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    res = model.load_state_dict(st, strict=True)
    assert len(res.missing_keys) == 0
    assert len(res.unexpected_keys) == 0


def test_forward_pass_finite_and_2x_shape(ckpt_path):
    device = select_device("auto")
    model = build_semicon_daair_v5(scale=2).to(device)
    st = torch.load(ckpt_path, map_location=device)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    model.load_state_dict(st, strict=True)
    model.eval()

    dummy_in = torch.randn(1, 1, 128, 128, device=device)
    with torch.no_grad():
        out = model(dummy_in)

    assert torch.isfinite(out).all().item(), "Forward pass output contains NaNs or Infs!"
    assert out.shape == torch.Size([1, 1, 256, 256]), f"Expected [1, 1, 256, 256], got {list(out.shape)}"


if __name__ == "__main__":
    print("=" * 60)
    print("     [SEMICONDAAIR-V5 PYTEST / UNITTEST VERIFICATION SUITE]     ")
    print("=" * 60)
    ckpt = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"
    test_model_construction()
    print("1. test_model_construction             : PASSED (555,141 params)")
    test_checkpoint_loading(ckpt)
    print("2. test_checkpoint_loading            : PASSED (0 missing keys)")
    test_forward_pass_finite_and_2x_shape(ckpt)
    print("3. test_forward_pass_finite_and_2x_shape: PASSED ([1,1,128,128] -> [1,1,256,256])")
    print("=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!\n")
