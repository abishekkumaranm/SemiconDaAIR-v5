"""
hf_edge_refinement.py — High-Frequency Structure Refinement Module for Semiconductor Inspection.

Extracts multi-representation high-frequency features (Sobel X, Sobel Y, Laplacian, Gradient Magnitude),
predicts a small residual correction, applies a confidence gate, and combines it with the base reconstruction.

Target parameter count: < 45,000 parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HFEdgeRefinement(nn.Module):
    """
    Lightweight High-Frequency Structure Refinement Module:
      F_hf = HFEncoder(F_features)
      R_hf = HFResidual(F_hf)
      G_hf = Sigmoid(HFGate(F_hf))
      Y_final = Y_base + G_hf * R_hf
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 1, hidden_channels: int = 32):
        super().__init__()
        
        # Fixed Sobel and Laplacian edge extraction kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

        # Edge feature extraction from base reconstruction & feature backbone
        self.edge_conv = nn.Sequential(
            nn.Conv2d(4, hidden_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(hidden_channels)
        )

        # Feature fusion (Backbone features + Edge features)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size=1),
            nn.PReLU(hidden_channels)
        )

        # High-Frequency Residual prediction
        self.hf_residual = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.PReLU(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )

        # High-Frequency Confidence Gate (Initialized with positive bias)
        self.hf_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.PReLU(hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=3, padding=1)
        )
        # Initialize gate bias to +2.0 (Sigmoid(2.0) ~ 0.88)
        nn.init.constant_(self.hf_gate[-1].bias, 2.0)

    def extract_edge_maps(self, y_base: torch.Tensor) -> torch.Tensor:
        """Extracts Sobel X, Sobel Y, Laplacian, and Gradient Magnitude."""
        gx = F.conv2d(y_base, self.sobel_x, padding=1)
        gy = F.conv2d(y_base, self.sobel_y, padding=1)
        lap = F.conv2d(y_base, self.laplacian, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return torch.cat([gx, gy, lap, grad_mag], dim=1)

    def forward(self, y_base: torch.Tensor, backbone_feat: torch.Tensor):
        """
        y_base: Reconstruction output (B, 1, H, W)
        backbone_feat: Upsampled backbone features (B, C, H, W)
        """
        edge_maps = self.extract_edge_maps(y_base)
        edge_feat = self.edge_conv(edge_maps)
        
        fused = self.fuse_conv(torch.cat([backbone_feat, edge_feat], dim=1))
        
        r_hf = self.hf_residual(fused)
        g_hf = torch.sigmoid(self.hf_gate(fused))
        
        y_final = y_base + g_hf * r_hf
        return y_final, g_hf, r_hf


if __name__ == "__main__":
    m = HFEdgeRefinement(in_channels=64, out_channels=1, hidden_channels=32)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"HFEdgeRefinement Parameter Count: {n_params:,} (Target < 45,000)")
    
    y_dummy = torch.randn(2, 1, 256, 256)
    feat_dummy = torch.randn(2, 64, 256, 256)
    y_out, g_out, r_out = m(y_dummy, feat_dummy)
    print(f"Output Shape: {y_out.shape} | Gate Shape: {g_out.shape} | Residual Shape: {r_out.shape}")
