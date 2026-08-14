"""
model.py — Unified Architecture Gateway for Semiconductor Restoration (SemiconDaAIR-v2, SemiconDaAIR, SemiconRestorNet).

Includes:
  1. SemiconDaAIR-v2: Advanced Degradation-Adapted FiLM MoE Network with Speckle-Aware Signed Log Branch
  2. SemiconDaAIR: Physics-Guided Multi-Label MoE Network
  3. SemiconRestorNet v2.0: Dual-Path Network
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.semicon_daair import SemiconDaAIR, build_semicon_daair
from models.semicon_daair_v2 import SemiconDaAIRv2, build_semicon_daair_v2
from models.degradation_router import DegradationRouter
from models.experts import ExpertFusion
from models.frequency_module import SelectiveFrequencyModule
from models.edge_module import EdgeGuidanceModule
from models.controller import SelfLearnableController
from models.sr_head import PixelShuffleSRHead


# --- Pretrained DnCNN Integration from SEMIHACKTHAN ---
class PretrainedDnCNN(nn.Module):
    """17-layer DnCNN residual noise estimator."""
    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)


def load_pretrained_dncnn():
    dncnn_path = r"C:\Users\HP\OneDrive\Documents\SEMIHACKTHAN\DnCNN-master\DnCNN-master\TrainingCodes\dncnn_pytorch\models\DnCNN_sigma25\model.pth"
    if os.path.exists(dncnn_path):
        try:
            sys.modules['__main__'].DnCNN = PretrainedDnCNN
            model = torch.load(dncnn_path, weights_only=False, map_location="cpu")
            return model
        except Exception:
            pass
    return None


class DynamicNoiseEstimator(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=32, desc_dim=64):
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("high_pass", laplacian[None, None, :, :])
        dncnn_model = load_pretrained_dncnn()
        self.dncnn = dncnn_model if dncnn_model is not None else PretrainedDnCNN(depth=5, n_channels=32)

        self.conv_noise = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.desc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, desc_dim),
            nn.SiLU()
        )

    def forward(self, x):
        hp = F.conv2d(x, self.high_pass.to(x.dtype), padding=1)
        dncnn_residual = self.dncnn(x)
        noise_map = self.conv_noise(torch.cat([hp, dncnn_residual], dim=1))
        desc = self.desc_head(noise_map)
        return noise_map, desc


class FiLM(nn.Module):
    def __init__(self, desc_dim=64, channels=64):
        super().__init__()
        self.fc = nn.Linear(desc_dim, channels * 2)

    def forward(self, x, desc):
        gamma_beta = self.fc(desc)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)


class DirectionalSobelGatedConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x[None, None, :, :])
        self.register_buffer("sobel_y", sobel_y[None, None, :, :])
        self.conv_feat = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_gate = nn.Conv2d(channels + 2, channels, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, raw_input):
        gx = F.conv2d(raw_input, self.sobel_x.to(x.dtype), padding=1)
        gy = F.conv2d(raw_input, self.sobel_y.to(x.dtype), padding=1)
        grad_concat = torch.cat([gx, gy], dim=1)
        if grad_concat.shape[2:] != x.shape[2:]:
            grad_concat = F.interpolate(grad_concat, size=x.shape[2:], mode='bilinear', align_corners=False)
        feat = self.conv_feat(x)
        gate = torch.sigmoid(self.conv_gate(torch.cat([x, grad_concat], dim=1)))
        return self.act(feat * gate)


class DualPathMoEResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.edge_expert = DirectionalSobelGatedConv(channels)
        self.texture_expert = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1)
        )
        self.act = nn.PReLU(channels)

    def forward(self, x, raw_input, expert_weights):
        w_edge = expert_weights[:, 0].view(-1, 1, 1, 1)
        w_tex = expert_weights[:, 1].view(-1, 1, 1, 1)
        edge_feat = self.edge_expert(x, raw_input)
        texture_feat = self.texture_expert(x)
        return x + self.act(w_edge * edge_feat + w_tex * texture_feat)


class SemiconRestorNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_blocks=8, scale=2):
        super().__init__()
        self.scale = scale
        self.noise_estimator = DynamicNoiseEstimator(in_channels=in_channels, hidden_dim=32, desc_dim=64)
        self.head = nn.Conv2d(in_channels + 1, base_channels, kernel_size=3, padding=1)
        self.film = FiLM(desc_dim=64, channels=base_channels)
        self.moe_router = DegradationRouter(in_channels=base_channels, hidden_dim=32, num_degradations=2)
        self.blocks = nn.ModuleList([DualPathMoEResidualBlock(base_channels) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(base_channels)
        )
        self.tail_image = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels, base_channels // 2, 3, padding=1),
            nn.PReLU(base_channels // 2),
            nn.Conv2d(base_channels // 2, in_channels, 3, padding=1)
        )

    def forward(self, x, return_confidence=False):
        up_ref = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        noise_map, desc = self.noise_estimator(x)
        feat = self.film(self.head(torch.cat([x, noise_map], dim=1)), desc)
        gates, _ = self.moe_router(feat)
        res = feat
        for block in self.blocks:
            res = block(res, raw_input=x, expert_weights=gates)
        feat = feat + self.body_tail(res)
        up_feat = self.upsample(feat)
        coarse = self.tail_image(up_feat) + up_ref
        out = coarse + self.refiner(coarse)
        return out


def build_model(scale=2, size="semicon_daair_v2", in_channels=1):
    """
    Unified model factory.
    Sizes: 'semicon_daair_v2' (Advanced FiLM MoE), 'semicon_daair', 'semicon_restornet', 'tiny'
    """
    if size == "semicon_daair_v2":
        return SemiconDaAIRv2(in_channels=in_channels, out_channels=1, base_channels=64, num_blocks=4, scale=scale, low_rank_reduction=8)
    elif size == "semicon_daair":
        return SemiconDaAIR(in_channels=in_channels, out_channels=1, base_channels=64, num_blocks=4, scale=scale, low_rank_reduction=8)
    elif size == "semicon_restornet":
        return SemiconRestorNet(in_channels=in_channels, base_channels=64, num_blocks=8, scale=scale)
    else:
        return SemiconDaAIR(in_channels=in_channels, out_channels=1, base_channels=32, num_blocks=2, scale=scale, low_rank_reduction=4)


if __name__ == "__main__":
    m = build_model(scale=2, size="semicon_daair_v2")
    n_params = sum(p.numel() for p in m.parameters())
    print(f"SemiconDaAIR-v2 Parameter Count: {n_params:,}")
    x = torch.randn(2, 1, 128, 128)
    out = m(x)
    print(f"Forward Pass Test -> Input: {x.shape} -> Output: {out.shape}")
