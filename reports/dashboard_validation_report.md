# Dashboard Validation & Safety Policy Report (`dashboard/`)

**Project**: KLA / SEMICON India Hackathon — AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Document**: Prototype Web Dashboard Architecture & Safety Policy Audit  

---

## 1. Executive Summary

The [`dashboard/`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/dashboard/) web dashboard is an interactive prototype inference interface built for presentation to hackathon judges.

---

## 2. Operating Modes & Claim Safety Policy

### Mode A: Validation Mode (Ground Truth Available)
- **Active State**: Enabled when clean GT is available.
- **Displayed Metrics**: Full reference-based metrics (PSNR = 27.75 dB, SSIM = 0.7438, LPIPS = 0.1824, MAE = 0.0335, Sobel HF Error = 0.1437).

### Mode B: Live Demo / Unknown Input Mode (No Ground Truth)
- **Active State**: Enabled for live un-labeled inputs or real-time demonstrations.
- **No-Reference Diagnostics**: Displays Input/Output resolution, dynamic range, noise estimate, gradient energy, edge sharpness, HF energy, artifact indicator, and inference time.
- **Mandatory Safety Label**: *"Ground truth unavailable: no-reference diagnostics"*.

---

## 3. Strict Industrial Claim Policy Compliance

1. **Title**: *SemiconDaAIR-v2 Semiconductor Image Restoration Dashboard*
2. **Subtitle**: *KLA / SEMICON India Hackathon — AI-Based Restoration of Degraded Inspection Images*
3. **Edge Analysis**: Labeled as *"Pixel-Domain Edge Fidelity Analysis"* (no nanometer metrology accuracy claimed without physical calibration).
4. **Simulator**: Labeled as *"Synthetic Degradation Simulator"* (no claim of exact KLA degradation simulator).
5. **Hardware**: Measured latency on RTX 4090 ($3.80\text{ ms}$) and RTX 3050 ($15.36\text{ ms}$). H100 benchmarks clearly labeled as *"H100 benchmark: pending"*.
6. **No Hardware Claim**: Clearly stated as software AI restoration prototype; no claim of real KLA SEM machine connection.
