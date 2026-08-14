"""
test_v3.py — PyTorch Unit Test Suite for SemiconDaAIR-v3 Research Architecture.
"""

import unittest
import torch

from models.semicon_daair_v3 import build_semicon_daair_v3, FidelityGatedHead, UnlabeledDegradationFingerprint, StateSpaceGlobalContextBlock


class TestSemiconDaAIRv3(unittest.TestCase):

    def test_fidelity_gated_head_shape(self):
        head = FidelityGatedHead(in_channels=64, out_channels=1, scale=2)
        feat = torch.randn(2, 64, 128, 128)
        lq = torch.randn(2, 1, 128, 128)
        out, conf = head(feat, lq)
        self.assertEqual(out.shape, (2, 1, 256, 256))
        self.assertEqual(conf.shape, (2, 1, 256, 256))
        self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())

    def test_unlabeled_fingerprint_shape(self):
        fp = UnlabeledDegradationFingerprint(in_channels=64, embed_dim=16)
        x = torch.randn(2, 64, 128, 128)
        modulated_x, d = fp(x)
        self.assertEqual(modulated_x.shape, (2, 64, 128, 128))
        self.assertEqual(d.shape, (2, 16))

    def test_ssm_block_shape(self):
        ssm = StateSpaceGlobalContextBlock(channels=64)
        x = torch.randn(2, 64, 128, 128)
        out = ssm(x)
        self.assertEqual(out.shape, (2, 64, 128, 128))

    def test_v3_full_model_forward(self):
        model = build_semicon_daair_v3(scale=2, use_fidelity_gate=True, fingerprint_dim=16, use_ssm=True)
        x = torch.randn(2, 1, 128, 128)
        out, extras = model(x, return_extras=True)
        self.assertEqual(out.shape, (2, 1, 256, 256))
        self.assertEqual(extras["confidence_map"].shape, (2, 1, 256, 256))
        self.assertEqual(extras["lq_reconstructed"].shape, (2, 1, 128, 128))


if __name__ == "__main__":
    unittest.main()
