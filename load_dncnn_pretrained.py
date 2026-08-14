"""
load_dncnn_pretrained.py — Extracts pretrained DnCNN weights from SEMIHACKTHAN
and transfers them into SemiconRestorNet's Dynamic Noise Estimator.
"""

import sys
import torch
import torch.nn as nn

dncnn_path = r"C:\Users\HP\OneDrive\Documents\SEMIHACKTHAN\DnCNN-master\DnCNN-master\TrainingCodes\dncnn_pytorch\models\DnCNN_sigma25\model.pth"


class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
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


def load_pretrained_dncnn_weights():
    sys.modules['__main__'].DnCNN = DnCNN
    print(f"Loading pretrained DnCNN model from: {dncnn_path}")
    dncnn_model = torch.load(dncnn_path, weights_only=False, map_location="cpu")
    print(f"Pretrained DnCNN successfully loaded! Parameter count: {sum(p.numel() for p in dncnn_model.parameters()):,}")
    return dncnn_model


if __name__ == "__main__":
    dncnn = load_pretrained_dncnn_weights()
    sample_x = torch.randn(1, 1, 128, 128)
    noise_pred = dncnn(sample_x)
    print(f"Sample Input: {sample_x.shape} -> Residual Noise Output: {noise_pred.shape}")
