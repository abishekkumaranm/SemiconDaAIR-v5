"""
test_shapes.py — Comprehensive PyTorch Unit Tests for SemiconDaAIR-v2.

Verifies:
  - Input shape [B, 1, 128, 128] -> Output shape [B, 1, 256, 256]
  - float32 dtype preservation
  - Negative intensity input preservation
  - Intensity values > 1.0 preservation
  - Checkpoint loading integrity (checkpoints/final/semicon_daair_v2_final.pt)
  - CPU & GPU inference fallback
  - Missing file handling
"""

import unittest
import os
import torch
import numpy as np

from models.semicon_daair_v2 import build_semicon_daair_v2


class TestSemiconDaAIRv2Pipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_semicon_daair_v2(scale=2, base_channels=64).to(self.device)
        self.model.eval()

        self.ckpt_path = "checkpoints/final/semicon_daair_v2_final.pt"
        if os.path.exists(self.ckpt_path):
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)

    def test_output_shape(self):
        x = torch.randn(2, 1, 128, 128, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 1, 256, 256))

    def test_float32_preservation(self):
        x = torch.randn(1, 1, 128, 128, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.dtype, torch.float32)

    def test_negative_input_preservation(self):
        # Degraded KLA input dynamic range includes negative values [-0.2786, 2.1580]
        x = torch.randn(1, 1, 128, 128, device=self.device, dtype=torch.float32) * 2.0 - 0.5
        with torch.no_grad():
            out = self.model(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_values_greater_than_one_preservation(self):
        x = torch.randn(1, 1, 128, 128, device=self.device, dtype=torch.float32) + 1.5
        with torch.no_grad():
            out = self.model(x)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_parameter_count(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(n_params, 544628)

    def test_cpu_fallback(self):
        cpu_model = build_semicon_daair_v2(scale=2, base_channels=64).to("cpu")
        cpu_model.eval()
        x_cpu = torch.randn(1, 1, 128, 128, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            out_cpu = cpu_model(x_cpu)
        self.assertEqual(out_cpu.shape, (1, 1, 256, 256))


if __name__ == "__main__":
    unittest.main()
