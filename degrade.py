"""
degrade.py — Synthetic degradation pipeline for KLA semiconductor-inspection restoration task.

Simulates, on grayscale images:
  1. Speckle noise   — multiplicative noise that can push pixel values beyond the
                        ground-truth range (I_noisy = I + I * n, n ~ N(0, sigma))
  2. Gaussian noise/blur — soft, hazy edges (additive Gaussian noise + light blur)
  3. Spatial resolution reduction — bicubic downsampling by a random factor (2x or 4x)

All three are applied together in randomized combinations/severities so a model
trained on this generalizes rather than overfitting to one fixed degradation recipe.
This lets you start training NOW, before KLA's real paired dataset is released —
swap this generator's output for the real dataset later and fine-tune.
"""
import numpy as np
import cv2


def add_speckle_noise(img, sigma_range=(0.05, 0.25)):
    """Multiplicative speckle noise. Can legitimately push values outside [0,1]."""
    sigma = np.random.uniform(*sigma_range)
    noise = np.random.randn(*img.shape).astype(np.float32) * sigma
    noisy = img + img * noise
    return noisy  # NOTE: intentionally NOT clipped here — matches KLA's note that
                  # degraded-image intensity range may exceed ground-truth range.


def add_gaussian_degradation(img, noise_sigma_range=(0.01, 0.08), blur_prob=0.7, blur_ksize_range=(3, 7)):
    """Additive Gaussian noise + optional slight blur -> soft/hazy edges."""
    out = img.copy()
    if np.random.rand() < blur_prob:
        k = np.random.choice(range(blur_ksize_range[0], blur_ksize_range[1] + 1, 2))
        out = cv2.GaussianBlur(out, (int(k), int(k)), 0)
    sigma = np.random.uniform(*noise_sigma_range)
    out = out + np.random.randn(*out.shape).astype(np.float32) * sigma
    return out


def downsample(img, scale):
    h, w = img.shape
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return small


def degrade_image(gt_img_uint8, scale=2):
    """
    gt_img_uint8: HxW uint8 grayscale ground-truth image (values 0-255).
    scale: KLA's spec only uses 2x (512->256 or 256->128), so this defaults to 2
    and stays fixed unless you have evidence otherwise from the real dataset.
    Returns: (degraded_img_float32 [can exceed 0-1], scale_used)
    """

    img = gt_img_uint8.astype(np.float32) / 255.0

    # Order matters: degrade at full res first (more realistic sensor-noise timing),
    # then downsample -- mirrors real acquisition where noise happens at capture time.
    img = add_speckle_noise(img)
    img = add_gaussian_degradation(img)
    img = downsample(img, scale)

    return img, scale


def make_pair(gt_img_uint8, scale=2):
    """Returns (degraded_float32_HxW, gt_float32_HxW_in_[0,1], scale)."""
    degraded, scale = degrade_image(gt_img_uint8, scale)
    gt = gt_img_uint8.astype(np.float32) / 255.0
    return degraded, gt, scale


if __name__ == "__main__":
    # quick self-test with a synthetic checkerboard-like pattern
    demo = (np.random.rand(256, 256) * 255).astype(np.uint8)
    d, gt, s = make_pair(demo, scale=4)
    print(f"GT shape: {gt.shape}, degraded shape: {d.shape}, scale: {s}")
    print(f"degraded range: [{d.min():.3f}, {d.max():.3f}]  (note: can exceed [0,1])")
