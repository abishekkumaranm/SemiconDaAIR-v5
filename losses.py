"""
losses.py — Physics-Aware Composite Loss Function Suite for Semiconductor Image Restoration.

Composite Loss Formulation:
  L_total = w1 * Charbonnier + w2 * MS-SSIM + w3 * SobelGradient + w4 * FourierFrequency + w5 * DefectPreservation + lambda_deg * BCEWithLogits

Includes FP32 precision clamps for SSIM and 2D FFT to guarantee 100% numerical stability under AMP FP16 training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 smooth variant): sqrt((pred - target)^2 + eps^2)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def gaussian_window(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


class SSIMLoss(nn.Module):
    """Single-scale SSIM loss with FP32 precision clamp to prevent AMP FP16 NaN underflow."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        g1d = gaussian_window(window_size, sigma)
        window = g1d[:, None] @ g1d[None, :]
        self.register_buffer("window", window[None, None, :, :])
        self.window_size = window_size

    def forward(self, pred, target):
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        window = self.window.to(torch.float32)

        mu1 = F.conv2d(p_fp32, window, padding=pad)
        mu2 = F.conv2d(t_fp32, window, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = torch.clamp(F.conv2d(p_fp32 * p_fp32, window, padding=pad) - mu1_sq, min=0.0)
        sigma2_sq = torch.clamp(F.conv2d(t_fp32 * t_fp32, window, padding=pad) - mu2_sq, min=0.0)
        sigma12 = F.conv2d(p_fp32 * t_fp32, window, padding=pad) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-7
        )
        return 1.0 - ssim_map.mean()


class MultiScaleSSIMLoss(nn.Module):
    """Multi-Scale SSIM loss across 3 spatial resolutions."""
    def __init__(self, weights=(0.5, 0.3, 0.2)):
        super().__init__()
        self.weights = weights
        self.ssim = SSIMLoss()

    def forward(self, pred, target):
        loss = 0.0
        p, t = pred, target
        for w in self.weights:
            loss += w * self.ssim(p, t)
            if p.shape[2] > 16 and p.shape[3] > 16:
                p = F.avg_pool2d(p, 2)
                t = F.avg_pool2d(t, 2)
        return loss


class SobelGradientEdgeLoss(nn.Module):
    """Sobel-gradient edge loss to enforce sharp pattern transition boundaries."""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x[None, None, :, :])
        self.register_buffer("sobel_y", sobel_y[None, None, :, :])

    def forward(self, pred, target):
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)
        sobel_x_buf = self.sobel_x.to(torch.float32)
        sobel_y_buf = self.sobel_y.to(torch.float32)

        gx_p = F.conv2d(p_fp32, sobel_x_buf, padding=1)
        gy_p = F.conv2d(p_fp32, sobel_y_buf, padding=1)
        gx_t = F.conv2d(t_fp32, sobel_x_buf, padding=1)
        gy_t = F.conv2d(t_fp32, sobel_y_buf, padding=1)
        
        mag_p = torch.sqrt(gx_p**2 + gy_p**2 + 1e-6)
        mag_t = torch.sqrt(gx_t**2 + gy_t**2 + 1e-6)
        return F.l1_loss(mag_p, mag_t)


class FourierFrequencyLoss(nn.Module):
    """Penalizes high-frequency magnitude spectrum discrepancies in 2D FFT domain."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)

        fft_p = torch.fft.rfft2(p_fp32, norm="ortho")
        fft_t = torch.fft.rfft2(t_fp32, norm="ortho")
        
        mag_p = torch.abs(fft_p)
        mag_t = torch.abs(fft_t)
        return F.l1_loss(mag_p, mag_t)


class DefectPreservationLoss(nn.Module):
    """Applies focused weight penalty on high-contrast feature boundaries where nanoscale defects occur."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        p_fp32 = pred.to(torch.float32)
        t_fp32 = target.to(torch.float32)

        target_mean = F.avg_pool2d(t_fp32, kernel_size=5, stride=1, padding=2)
        target_sq_mean = F.avg_pool2d(t_fp32 ** 2, kernel_size=5, stride=1, padding=2)
        local_var = torch.clamp(target_sq_mean - target_mean ** 2, min=0.0)
        
        weights = 1.0 + 5.0 * (local_var / (local_var.max() + 1e-6))
        diff = torch.abs(p_fp32 - t_fp32)
        return torch.mean(weights * diff)


class MultiLabelDegradationLoss(nn.Module):
    """Auxiliary multi-label BCEWithLogitsLoss for degradation router supervision."""
    def __init__(self, lambda_deg=0.05):
        super().__init__()
        self.lambda_deg = lambda_deg
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target_vector):
        return self.lambda_deg * self.bce(logits.to(torch.float32), target_vector.to(torch.float32))


class RestorationLoss(nn.Module):
    """
    Semiconductor-Specific Composite Loss Function.
    Combines Charbonnier, MS-SSIM, Sobel Edge, Fourier Frequency, Defect Preservation, and Auxiliary Degradation loss.
    """
    def __init__(self, w_charb=1.0, w_ms_ssim=0.4, w_edge=0.3, w_freq=0.2, w_defect=0.3, lambda_deg=0.05):
        super().__init__()
        self.w_charb = w_charb
        self.w_ms_ssim = w_ms_ssim
        self.w_edge = w_edge
        self.w_freq = w_freq
        self.w_defect = w_defect

        self.charbonnier = CharbonnierLoss()
        self.ms_ssim = MultiScaleSSIMLoss()
        self.edge = SobelGradientEdgeLoss()
        self.freq = FourierFrequencyLoss()
        self.defect = DefectPreservationLoss()
        self.deg_loss = MultiLabelDegradationLoss(lambda_deg=lambda_deg)

    def forward(self, pred, target, router_logits=None, target_deg_vector=None):
        l_charb = self.charbonnier(pred, target)
        l_ssim = self.ms_ssim(pred, target)
        l_edge = self.edge(pred, target)
        l_freq = self.freq(pred, target)
        l_defect = self.defect(pred, target)

        total = (
            self.w_charb * l_charb +
            self.w_ms_ssim * l_ssim +
            self.w_edge * l_edge +
            self.w_freq * l_freq +
            self.w_defect * l_defect
        )

        l_deg_val = 0.0
        if router_logits is not None and target_deg_vector is not None:
            l_deg = self.deg_loss(router_logits, target_deg_vector)
            total = total + l_deg
            l_deg_val = l_deg.item()

        loss_dict = {
            "charbonnier": l_charb.item(),
            "ms_ssim": l_ssim.item(),
            "edge_loss": l_edge.item(),
            "freq_loss": l_freq.item(),
            "defect_loss": l_defect.item(),
            "deg_loss": l_deg_val,
            "total_loss": total.item()
        }

        return total, loss_dict


if __name__ == "__main__":
    criterion = RestorationLoss()
    pred = torch.randn(2, 1, 256, 256, requires_grad=True)
    target = torch.randn(2, 1, 256, 256)
    logits = torch.randn(2, 3, requires_grad=True)
    target_vector = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    
    loss, details = criterion(pred, target, router_logits=logits, target_deg_vector=target_vector)
    print("Loss Calculation Output:", details)
    loss.backward()
    print("Backpropagation check passed successfully.")
