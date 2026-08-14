"""
inspection_assurance.py — Industrial Inspection Assurance, OOD Detection & Metrology Guard Engine

Transforms raw image restoration into an AI-Assisted Inspection Assurance System for Semiconductor Manufacturing:
  1. Physics Degradation Analyzer (Estimates Speckle %, Gaussian Noise %, Blur Radius, Resolution Reduction)
  2. Out-of-Distribution (OOD) Detector (Flags unknown patterns like 5nm FinFET / 3D NAND for Engineer Review)
  3. Metrology Guard (Pre vs Post CD, LER, Overlay Shift verification)
  4. Inspection Readiness Score Generator (Computes 0-100% score: PASS / RESCAN / FAIL)
  5. Automatic Failure Explainer (Provides human-readable operator recommendations)
  6. Factory Decision Engine (Emits PASS, RESCAN, ENGINEER_REVIEW)
  7. Continuous Factory Learning Logger (Appends accepted frames for fab retraining)
"""

import os
import time
import json
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from metrics import compute_cd_error, compute_overlay_error, compute_ler_fidelity, compute_psnr, compute_ssim


class PhysicsDegradationAnalyzer:
    """Estimates physical degradation parameters from raw optical / SEM inspection images."""
    def __init__(self):
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = sobel_x.T
        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

    def analyze(self, img_float):
        """img_float: HxW array in float32."""
        # 1. Estimate Speckle Multiplicative Noise %
        high_intensity_mask = img_float > np.percentile(img_float, 75)
        if np.sum(high_intensity_mask) > 10:
            speckle_var = float(np.var(img_float[high_intensity_mask]))
            speckle_pct = float(min(speckle_var * 100.0 * 2.5, 45.0))
        else:
            speckle_pct = 5.0

        # 2. Estimate Additive Gaussian Noise %
        laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
        sigma_g = float(np.median(np.abs(laplacian)) / 0.6745)
        gaussian_pct = float(min(sigma_g * 100.0, 30.0))

        # 3. Estimate Blur Radius (px)
        gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        mean_grad = float(np.mean(grad_mag))
        blur_radius = float(max(1.0, min(5.0, 3.5 / (mean_grad + 1e-4))))

        # 4. Estimate Spatial Resolution Reduction Factor
        h, w = img_float.shape
        fft = np.fft.fft2(img_float)
        fft_shift = np.fft.fftshift(fft)
        mag = np.abs(fft_shift)
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask_low = (x - cx)**2 + (y - cy)**2 <= r**2
        total_energy = np.sum(mag)
        high_freq_energy = np.sum(mag[~mask_low])
        hf_ratio = high_freq_energy / (total_energy + 1e-8)

        resolution_loss = 2.0 if hf_ratio < 0.15 else 1.0

        return {
            "speckle_pct": round(speckle_pct, 1),
            "gaussian_pct": round(gaussian_pct, 1),
            "blur_radius": round(blur_radius, 2),
            "resolution_scale": f"{int(resolution_loss)}x",
            "high_freq_ratio": round(float(hf_ratio), 4)
        }


class OutOfDistributionDetector:
    """Detects unknown wafer patterns (e.g. uncalibrated 5nm FinFET / 3D NAND) to prevent bad AI restoration."""
    def __init__(self, entropy_threshold_min=1.5, entropy_threshold_max=7.5, max_speckle_limit=40.0):
        self.entropy_min = entropy_threshold_min
        self.entropy_max = entropy_threshold_max
        self.max_speckle = max_speckle_limit

    def detect(self, img_float, degradation_report):
        from scipy.stats import entropy
        hist, _ = np.histogram(img_float, bins=256, range=(0, 1), density=True)
        hist = hist[hist > 0]
        img_entropy = float(entropy(hist, base=2))

        reasons = []
        is_ood = False

        if img_entropy < self.entropy_min:
            is_ood = True
            reasons.append(f"Entropy too low ({img_entropy:.2f} bits < {self.entropy_min} bits) - Uniform blank wafer field.")
        elif img_entropy > self.entropy_max:
            is_ood = True
            reasons.append(f"Entropy too high ({img_entropy:.2f} bits > {self.entropy_max} bits) - Extreme noise breakdown.")

        if degradation_report["speckle_pct"] > self.max_speckle:
            is_ood = True
            reasons.append(f"Speckle noise ({degradation_report['speckle_pct']}%) exceeds fab tolerance threshold ({self.max_speckle}%).")

        return {
            "is_ood": is_ood,
            "entropy_bits": round(img_entropy, 3),
            "status": "OOD_FLAGGED" if is_ood else "KNOWN_PATTERN",
            "reasons": reasons
        }


class MetrologyGuard:
    """Verifies Critical Dimension (CD), Line Edge Roughness (LER), and Overlay Shift before vs after restoration."""
    def __init__(self, cd_threshold_nm=0.20, overlay_threshold_px=0.05):
        self.cd_threshold_nm = cd_threshold_nm
        self.overlay_threshold_px = overlay_threshold_px

    def verify(self, restored_float, gt_or_input_float, nm_per_pixel=1.5):
        p_norm = np.clip(restored_float, 0, 1)
        t_norm = np.clip(gt_or_input_float, 0, 1)

        # Match spatial dimensions if input is downsampled (e.g., 256x256 vs restored 512x512)
        if p_norm.shape != t_norm.shape:
            t_norm = cv2.resize(t_norm, (p_norm.shape[1], p_norm.shape[0]), interpolation=cv2.INTER_CUBIC)

        cd_res = compute_cd_error(p_norm, t_norm, nm_per_pixel=nm_per_pixel)
        overlay_res = compute_overlay_error(p_norm, t_norm)
        ler_res = compute_ler_fidelity(p_norm, t_norm)

        cd_mae_nm = cd_res["cd_mae_nm"]
        overlay_shift_px = overlay_res["overlay_shift_px"]

        pass_cd = cd_mae_nm <= self.cd_threshold_nm
        pass_overlay = overlay_shift_px <= self.overlay_threshold_px
        pass_metrology = pass_cd and pass_overlay

        return {
            "cd_mae_nm": round(cd_mae_nm, 3),
            "overlay_shift_px": round(overlay_shift_px, 3),
            "ler_error_px": round(ler_res["ler_error_px"], 3),
            "pass_cd_metrology": pass_cd,
            "pass_overlay_metrology": pass_overlay,
            "pass_metrology_guard": pass_metrology,
            "status": "PASS" if pass_metrology else "METROLOGY_VIOLATION"
        }


class InspectionReadinessScore:
    """Computes overall Inspection Readiness Score (0-100%) and decision rating."""
    def compute(self, restored_float, raw_float, metrology_guard, confidence_map=None):
        # 1. Edge Sharpness Score
        gx = cv2.Sobel(restored_float, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(restored_float, cv2.CV_32F, 0, 1)
        edge_score = float(min(100.0, np.mean(np.sqrt(gx**2 + gy**2)) * 250.0))

        # 2. Contrast Score
        contrast_score = float(min(100.0, np.std(restored_float) * 500.0))

        # 3. Frequency Recovery Score
        fft_p = np.abs(np.fft.fft2(restored_float))
        freq_score = float(min(100.0, (np.sum(fft_p) / (restored_float.size + 1e-6)) * 10.0))

        # 4. Confidence Score
        if confidence_map is not None:
            overall_conf = float(np.mean(confidence_map) * 100.0)
            edge_conf = float(np.mean(confidence_map[gx**2 + gy**2 > 0.01]) * 100.0) if np.sum(gx**2 + gy**2 > 0.01) > 0 else overall_conf
            texture_conf = float(np.mean(confidence_map[gx**2 + gy**2 <= 0.01]) * 100.0) if np.sum(gx**2 + gy**2 <= 0.01) > 0 else overall_conf
        else:
            overall_conf = 95.0
            edge_conf = 92.0
            texture_conf = 94.0

        # Weighted Readiness Calculation
        readiness = 0.25 * edge_score + 0.20 * contrast_score + 0.15 * freq_score + 0.40 * overall_conf

        # Deduct if metrology guard failed
        if not metrology_guard["pass_metrology_guard"]:
            readiness -= 25.0

        readiness = float(max(0.0, min(100.0, readiness)))

        if readiness >= 85.0:
            decision = "PASS"
        elif readiness >= 65.0:
            decision = "RESCAN"
        else:
            decision = "FAIL"

        return {
            "inspection_readiness_pct": round(readiness, 1),
            "decision": decision,
            "breakdown": {
                "edge_sharpness_score": round(edge_score, 1),
                "contrast_score": round(contrast_score, 1),
                "frequency_recovery_score": round(freq_score, 1),
                "overall_confidence_pct": round(overall_conf, 1),
                "edge_confidence_pct": round(edge_conf, 1),
                "texture_confidence_pct": round(texture_conf, 1)
            }
        }


class IndustrialAssuranceEngine:
    """Master Pipeline orchestrating Ingestion, Quality Analysis, OOD Detection, Restoration, Metrology Guard & Telemetry."""
    def __init__(self, model_instance=None):
        self.degradation_analyzer = PhysicsDegradationAnalyzer()
        self.ood_detector = OutOfDistributionDetector()
        self.metrology_guard = MetrologyGuard()
        self.readiness_evaluator = InspectionReadinessScore()
        self.model = model_instance

    def process_wafer_frame(self, raw_img_float, metadata=None, model_weights_path="checkpoints/best_model.pt"):
        start_t = time.time()

        # 1. Physics Degradation Analysis
        deg_report = self.degradation_analyzer.analyze(raw_img_float)

        # 2. Out-of-Distribution Detection
        ood_report = self.ood_detector.detect(raw_img_float, deg_report)

        # 3. Model Inference (if known pattern)
        if self.model is None and os.path.exists(model_weights_path):
            from model import build_model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = build_model(scale=2, size="semicon_restornet").to(device)
            checkpoint = torch.load(model_weights_path, map_location=device)
            state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

        if self.model is not None and not ood_report["is_ood"]:
            device = next(self.model.parameters()).device
            tensor_in = torch.from_numpy(raw_img_float).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out_img, out_conf, _ = self.model(tensor_in, return_confidence=True)
            restored_np = out_img.squeeze().cpu().numpy()
            conf_np = out_conf.squeeze().cpu().numpy()
        else:
            # Fallback if OOD or no model
            restored_np = raw_img_float.copy()
            conf_np = np.ones_like(raw_img_float) * 0.5

        # 4. Metrology Guard Verification
        met_report = self.metrology_guard.verify(restored_np, raw_img_float)

        # 5. Inspection Readiness Score
        readiness_report = self.readiness_evaluator.compute(restored_np, raw_img_float, met_report, conf_np)

        # 6. Automatic Failure Explanation
        reasons = []
        if ood_report["is_ood"]:
            reasons.extend(ood_report["reasons"])
        if not met_report["pass_cd_metrology"]:
            reasons.append(f"CD Metrology Error ({met_report['cd_mae_nm']} nm) exceeds tolerance threshold (0.20 nm).")
        if readiness_report["inspection_readiness_pct"] < 85.0:
            reasons.append(f"Inspection Readiness ({readiness_report['inspection_readiness_pct']}%) is below fab pass threshold (85.0%).")

        failure_explanation = " | ".join(reasons) if reasons else "No anomalies detected. Wafer frame ready for inspection."

        # Final Factory Decision
        if ood_report["is_ood"]:
            factory_decision = "ENGINEER_REVIEW"
        elif readiness_report["decision"] == "PASS" and met_report["pass_metrology_guard"]:
            factory_decision = "PASS"
        elif readiness_report["decision"] == "RESCAN":
            factory_decision = "RESCAN"
        else:
            factory_decision = "FAIL"

        processing_ms = (time.time() - start_t) * 1000.0

        # Build Complete Factory Inspection Assurance Output
        result = {
            "metadata": metadata or {},
            "degradation_analysis": deg_report,
            "ood_detection": ood_report,
            "metrology_guard": met_report,
            "readiness_score": readiness_report,
            "factory_decision": factory_decision,
            "failure_explanation": failure_explanation,
            "telemetry": {
                "processing_time_ms": round(processing_ms, 2),
                "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
            }
        }

        return result, restored_np, conf_np


def log_continuous_factory_learning(record_dict, log_file="logs/factory_learning_db.jsonl"):
    """Appends accepted wafer inspection records for continuous model retraining."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict) + "\n")
