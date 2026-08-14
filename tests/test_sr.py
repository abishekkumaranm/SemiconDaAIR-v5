"""
test_sr.py — Unit Tests for PixelShuffle SR Head & Global Residual Upsampling.
"""

import unittest
import torch
from models.sr_head import PixelShuffleSRHead


class TestSRHead(unittest.TestCase):
    def setUp(self):
        self.sr_head = PixelShuffleSRHead(in_channels=32, out_channels=1, upscale=2)

    def test_sr_upsampling_shape(self):
        feat = torch.randn(2, 32, 64, 64)
        raw = torch.randn(2, 1, 64, 64)
        out = self.sr_head(feat, raw)
        self.assertEqual(out.shape, (2, 1, 128, 128))

    def test_no_checkerboard_nan(self):
        feat = torch.randn(1, 32, 128, 128)
        raw = torch.randn(1, 1, 128, 128)
        out = self.sr_head(feat, raw)
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
