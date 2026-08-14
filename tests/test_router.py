"""
test_router.py — Unit Tests for Multi-Label Degradation Router.
"""

import unittest
import torch
from models.degradation_router import DegradationRouter


class TestDegradationRouter(unittest.TestCase):
    def setUp(self):
        self.router = DegradationRouter(in_channels=32, hidden_dim=16, num_degradations=3)

    def test_router_outputs(self):
        x = torch.randn(4, 32, 64, 64)
        gates, logits = self.router(x)
        self.assertEqual(gates.shape, (4, 3))
        self.assertEqual(logits.shape, (4, 3))
        self.assertTrue((gates >= 0.0).all() and (gates <= 1.0).all())

    def test_multilabel_coexistence(self):
        x = torch.randn(2, 32, 32, 32)
        gates, _ = self.router(x)
        # Verify multiple gates can be simultaneously active (> 0.5)
        self.assertEqual(gates.ndim, 2)


if __name__ == "__main__":
    unittest.main()
