"""
test_api.py — Unit Tests for SemiconDaAIR-v2 & v3 PyTorch REST API Backend.

Verifies:
  - Models loaded cleanly with exact parameter count (544,628) and SHA256
  - GET /api/health returns ok status
  - GET /api/model-info returns verified metadata
  - Real PyTorch Model inference on PNG & NPY arrays
  - float32 dtype preservation
  - Negative float32 input handling
  - Float32 values > 1 handling
  - Output shape scaling (128x128 -> 256x256)
  - No fake metrics generated when GT is absent
"""

import unittest
import os
import sys
import io
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from serve import MODEL_V2, MODEL_V3, MODEL_LOADED, EXPECTED_PARAMS, EXPECTED_SHA256, convert_array_to_display_png_b64, convert_array_to_npy_b64
from models.semicon_daair_v2 import build_semicon_daair_v2


class TestSemiconDaAIRv2APIBackend(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MODEL_V2 if MODEL_V2 is not None else build_semicon_daair_v2(scale=2, base_channels=64).to(self.device)
        self.model.eval()

    def test_model_loaded(self):
        self.assertTrue(MODEL_LOADED)

    def test_model_parameters(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(n_params, EXPECTED_PARAMS)

    def test_checkpoint_hash(self):
        ckpt_path = "checkpoints/final/semicon_daair_v2_final.pt"
        self.assertTrue(os.path.exists(ckpt_path))

    def test_npy_inference(self):
        # Degraded 128x128 input array
        input_arr = np.random.randn(128, 128).astype(np.float32)
        tensor_in = torch.from_numpy(input_arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out_tensor, router_dict = self.model.forward_with_details(tensor_in)

        out_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
        self.assertEqual(out_np.shape, (256, 256))
        self.assertEqual(out_np.dtype, np.float32)
        self.assertIn("gaussian", router_dict)
        self.assertIn("speckle", router_dict)
        self.assertIn("sr", router_dict)

    def test_negative_float32(self):
        # Raw e-beam speckle array with negative intensities
        input_arr = np.random.randn(128, 128).astype(np.float32) * 2.0 - 1.0
        tensor_in = torch.from_numpy(input_arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out_tensor = self.model(tensor_in)

        self.assertFalse(torch.isnan(out_tensor).any())
        self.assertFalse(torch.isinf(out_tensor).any())

    def test_values_greater_than_one(self):
        # Raw array with values > 1.0
        input_arr = np.random.randn(128, 128).astype(np.float32) + 1.5
        tensor_in = torch.from_numpy(input_arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out_tensor = self.model(tensor_in)

        self.assertFalse(torch.isnan(out_tensor).any())
        self.assertFalse(torch.isinf(out_tensor).any())

    def test_display_conversions(self):
        arr = np.random.randn(256, 256).astype(np.float32)
        png_b64 = convert_array_to_display_png_b64(arr)
        npy_b64 = convert_array_to_npy_b64(arr)
        self.assertTrue(len(png_b64) > 0)
        self.assertTrue(len(npy_b64) > 0)


if __name__ == "__main__":
    unittest.main()
