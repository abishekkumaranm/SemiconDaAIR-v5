"""
inspect_wafer.py — Digital Manufacturing Dashboard & Industrial Wafer Inspection CLI.

Runs the complete AI-Assisted Inspection Assurance Engine on an image frame:
  - Physics Degradation Analyzer
  - OOD Detector
  - Adaptive SemiconRestorNet v2.0
  - Metrology Guard (Pre vs Post CD, LER, Overlay Shift)
  - Inspection Readiness Score
  - Factory Decision Engine (PASS / RESCAN / ENGINEER_REVIEW)
  - Automatic Failure Explainer
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import torch

from inspection_assurance import IndustrialAssuranceEngine
from degrade import degrade_image


def run_digital_manufacturing_inspection(image_path, weights_path="checkpoints/best_model.pt", wafer_id="WAF_300MM_8921", lot_id="LOT_EUV_9942", layer_id="M1_INTERCONNECT", apply_degradation=True):
    print("=" * 80)
    print("     KLA DIGITAL MANUFACTURING DASHBOARD - WAFER INSPECTION STATION     ")
    print("=" * 80)

    if not os.path.exists(image_path):
        print(f"Error: Input image file not found at {image_path}")
        return

    gt_uint8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gt_uint8 is None:
        print(f"Error: Could not decode grayscale image at {image_path}")
        return

    if apply_degradation:
        # Synthesize realistic speckle + blur + 2x downsampling input
        lq_float, scale = degrade_image(gt_uint8, scale=2)
        gt_float = gt_uint8.astype(np.float32) / 255.0
        input_float = lq_float
        reference_float = gt_float
    else:
        input_float = gt_uint8.astype(np.float32) / 255.0
        reference_float = input_float

    metadata = {
        "wafer_id": wafer_id,
        "lot_id": lot_id,
        "layer_id": layer_id,
        "acquisition_mode": "SEM_SECONDARY_ELECTRON",
        "magnification": "50000X",
        "resolution": f"{input_float.shape[1]}x{input_float.shape[0]}",
        "nm_per_pixel": 1.5
    }

    engine = IndustrialAssuranceEngine()
    
    # Run Inspection Assurance
    deg_report = engine.degradation_analyzer.analyze(input_float)
    ood_report = engine.ood_detector.detect(input_float, deg_report)

    if engine.model is None and os.path.exists(weights_path):
        from model import build_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine.model = build_model(scale=2, size="semicon_restornet").to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        engine.model.load_state_dict(state_dict, strict=False)
        engine.model.eval()

    device = next(engine.model.parameters()).device
    tensor_in = torch.from_numpy(input_float).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_img, out_conf, _ = engine.model(tensor_in, return_confidence=True)
    restored_np = out_img.squeeze().cpu().numpy()
    conf_np = out_conf.squeeze().cpu().numpy()

    # Compare restored output against ground-truth reference for metrology guard
    met_report = engine.metrology_guard.verify(restored_np, reference_float)
    readiness_report = engine.readiness_evaluator.compute(restored_np, reference_float, met_report, conf_np)

    reasons = []
    if ood_report["is_ood"]:
        reasons.extend(ood_report["reasons"])
    if not met_report["pass_cd_metrology"]:
        reasons.append(f"CD Metrology Error ({met_report['cd_mae_nm']} nm) exceeds tolerance threshold (0.20 nm).")
    if readiness_report["inspection_readiness_pct"] < 85.0:
        reasons.append(f"Inspection Readiness ({readiness_report['inspection_readiness_pct']}%) is below fab pass threshold (85.0%).")

    failure_explanation = " | ".join(reasons) if reasons else "No anomalies detected. Wafer frame ready for inspection."

    if ood_report["is_ood"]:
        factory_decision = "ENGINEER_REVIEW"
    elif readiness_report["decision"] == "PASS" and met_report["pass_metrology_guard"]:
        factory_decision = "PASS"
    elif readiness_report["decision"] == "RESCAN":
        factory_decision = "RESCAN"
    else:
        factory_decision = "FAIL"

    print(f"\n[INSPECTION TARGET METADATA]")
    print(f"  Wafer ID          : {metadata['wafer_id']}")
    print(f"  Lot ID            : {metadata['lot_id']}")
    print(f"  Layer ID          : {metadata['layer_id']}")
    print(f"  Acquisition Mode  : {metadata['acquisition_mode']} ({metadata['magnification']})")
    print(f"  Resolution        : {metadata['resolution']} (Grid: {metadata['nm_per_pixel']} nm/px)")

    print(f"\n[PHYSICS DEGRADATION ANALYZER]")
    print(f"  Speckle Noise     : {deg_report['speckle_pct']}%")
    print(f"  Gaussian Noise    : {deg_report['gaussian_pct']}%")
    print(f"  Blur Radius       : {deg_report['blur_radius']} px")
    print(f"  Resolution Scale  : {deg_report['resolution_scale']}")
    print(f"  High-Freq Ratio   : {deg_report['high_freq_ratio']}")

    print(f"\n[OUT-OF-DISTRIBUTION (OOD) DETECTOR]")
    print(f"  Pattern Status    : {ood_report['status']}")
    print(f"  Entropy           : {ood_report['entropy_bits']} bits/pixel")
    if ood_report["is_ood"]:
        print(f"  OOD Reasons       : {ood_report['reasons']}")

    print(f"\n[EXPLAINABLE RESTORATION & CONFIDENCE MAP]")
    print(f"  Overall Conf      : {readiness_report['breakdown']['overall_confidence_pct']}%")
    print(f"  Edge Conf         : {readiness_report['breakdown']['edge_confidence_pct']}%")
    print(f"  Texture Conf      : {readiness_report['breakdown']['texture_confidence_pct']}%")

    print(f"\n[METROLOGY GUARD VERIFICATION]")
    print(f"  Critical Dimension (CD) MAE : {met_report['cd_mae_nm']} nm  (Limit: 0.20 nm) -> {'[PASS]' if met_report['pass_cd_metrology'] else '[FAIL]'}")
    print(f"  Overlay Registration Shift : {met_report['overlay_shift_px']} px  (Limit: 0.05 px) -> {'[PASS]' if met_report['pass_overlay_metrology'] else '[FAIL]'}")
    print(f"  Line Edge Roughness (LER)  : {met_report['ler_error_px']} px")

    print(f"\n[FACTORY DECISION ENGINE]")
    print(f"  INSPECTION READINESS SCORE : {readiness_report['inspection_readiness_pct']}%")
    print(f"  FACTORY DECISION RATING    : [{factory_decision}]")
    print(f"  OPERATOR EXPLANATION       : {failure_explanation}")
    print("=" * 80)

    os.makedirs("results/dashboard_outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(f"results/dashboard_outputs/{base_name}_restored.png", (np.clip(restored_np, 0, 1) * 255.0).astype(np.uint8))
    cv2.imwrite(f"results/dashboard_outputs/{base_name}_confidence.png", (np.clip(conf_np, 0, 1) * 255.0).astype(np.uint8))

    result_json = {
        "metadata": metadata,
        "degradation_analysis": deg_report,
        "ood_detection": ood_report,
        "metrology_guard": met_report,
        "readiness_score": readiness_report,
        "factory_decision": factory_decision,
        "failure_explanation": failure_explanation
    }

    with open(f"results/dashboard_outputs/{base_name}_report.json", "w") as f:
        json.dump(result_json, f, indent=2)

    print(f"\nInspection artifacts saved to results/dashboard_outputs/{base_name}_*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLA Digital Manufacturing Inspection Station CLI")
    parser.add_argument("--image", type=str, default="data/clean_images/img_0.png", help="Path to input inspection image")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pt", help="Path to model weights")
    parser.add_argument("--wafer_id", type=str, default="WAF_300MM_8921")
    parser.add_argument("--lot_id", type=str, default="LOT_EUV_9942")
    parser.add_argument("--layer_id", type=str, default="M1_INTERCONNECT")
    parser.add_argument("--no_degrade", action="store_true", help="Disable synthetic degradation of clean input")

    args = parser.parse_args()
    run_digital_manufacturing_inspection(
        args.image,
        weights_path=args.weights,
        wafer_id=args.wafer_id,
        lot_id=args.lot_id,
        layer_id=args.layer_id,
        apply_degradation=not args.no_degrade
    )
