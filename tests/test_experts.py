"""
test_experts.py — Unit Tests for Shared and Specialized Low-Rank Experts and Fusion.
"""

import unittest
import torch
from models.experts import SharedExpert, SpeckleExpert, GaussianExpert, ResolutionExpert, ExpertFusion


class TestExperts(unittest.TestCase):
    def setUp(self):
        self.channels = 32
        self.reduction = 4
        self.shared = SharedExpert(self.channels, reduction=self.reduction)
        self.speckle = SpeckleExpert(self.channels, reduction=self.reduction)
        self.gaussian = GaussianExpert(self.channels, reduction=self.reduction)
        self.resolution = ResolutionExpert(self.channels, reduction=self.reduction)
        self.fusion = ExpertFusion(self.channels, reduction=self.reduction)

    def test_expert_shapes(self):
        x = torch.randn(2, 32, 16, 16)
        gates = torch.tensor([[0.9, 0.8, 0.7], [0.1, 0.5, 0.2]])
        
        out_shared = self.shared(x)
        out_speckle = self.speckle(x)
        out_gaussian = self.gaussian(x)
        out_res = self.resolution(x)
        out_fused = self.fusion(x, gates)

        self.assertEqual(out_shared.shape, x.shape)
        self.assertEqual(out_speckle.shape, x.shape)
        self.assertEqual(out_gaussian.shape, x.shape)
        self.assertEqual(out_res.shape, x.shape)
        self.assertEqual(out_fused.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
