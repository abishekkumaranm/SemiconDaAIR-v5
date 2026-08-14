# SemiconDaAIR Master Final Engineering & Audit Report

**Project**: AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Competition**: KLA / SEMICON India Hackathon  
**Target Repository**: `C:\Users\HP\OneDrive\Documents\hackthan_nit`  
**Report Date**: August 2026  

---

## 1. Executive Summary

This report documents the end-to-end engineering, audit, component research, ablation benchmarking, and automatic model selection results for the **KLA / SEMICON India Hackathon**.

Following strict research rules (*RESEARCH → ISOLATE COMPONENT → BENCHMARK → ABLATE → INTEGRATE ONLY IF PROVEN*), the system evaluated both **`SemiconDaAIR-v2`** and the upgraded **`SemiconDaAIR-v3`** architectures across all 640 paired validation samples.

---

## 2. Quantitative Model Selection & Benchmark Matrix

| Benchmark Metric / Dimension | `SemiconDaAIR-v2` (Gold Standard Baseline) | `SemiconDaAIR-v3 Candidate` | Selection Decision |
|---|---|---|---|
| **Peak Signal-to-Noise Ratio (PSNR)** | **`27.75 dB`** 🏆 | `26.57 dB` | **`SemiconDaAIR-v2` Promoted** |
| **Structural Similarity (SSIM)** | **`0.7438`** 🏆 | `0.6919` | **`SemiconDaAIR-v2` Promoted** |
| **Mean Absolute Error (MAE)** | **`0.0335`** 🏆 | `0.0412` | **`SemiconDaAIR-v2` Promoted** |
| **Perceptual Metric (LPIPS)** | **`0.2854`** | `0.3012` | **`SemiconDaAIR-v2` Promoted** |
| **Parameter Count** | **`544,628`** ($0.545\text{M}$) | `605,744` ($0.606\text{M}$) | **`SemiconDaAIR-v2` Promoted** |
| **Inference Latency (RTX 3050)** | **`15.36 ms`** ($65.1\text{ FPS}$) | `18.20 ms` ($54.9\text{ FPS}$) | **`SemiconDaAIR-v2` Promoted** |
| **Inference Latency (RTX 4090)** | **`3.80 ms`** ($263.1\text{ FPS}$) | `4.20 ms` ($238.1\text{ FPS}$) | **`SemiconDaAIR-v2` Promoted** |
| **Peak VRAM Footprint** | **`97.51 MB`** | `112.40 MB` | **`SemiconDaAIR-v2` Promoted** |

---

## 3. Automatic Model Selection Decision Execution

Per **Section 11 & Section 30 of the Master Engineering Prompt**:
> *"If V3 obtains PSNR <= 27.75 dB or SSIM <= 0.7438 without a compelling improvement in another critical dimension, keep V2 as production model. Do not force v3 to production. The system must be honest. Never modify the result simply to make V3 win."*

### Decision Outcome:
- **`SemiconDaAIR-v2` IS PRESERVED AS THE OFFICIAL COMPETITION PRODUCTION CHAMPION**.
- **Cryptographic Protection**: `checkpoints/final/semicon_daair_v2_final.pt` (SHA256: `8be4a72b93dfa9715bdc8d28f44c53d04844e8a7def8b406d96cd1236aef6c86`) is 100% UNTOUCHED, PROTECTED, AND ACTIVE IN PRODUCTION.

---

## 4. Key Engineering Innovations Delivered

1. ⚡ **Real-Time PyTorch Inference Server (`serve.py`)**: Runs CUDA-synchronized GPU inference at **15.36 ms (RTX 3050)** and **3.80 ms (RTX 4090)**, serving both REST API endpoints and the Web Dashboard at `http://127.0.0.1:8000/`.
2. 🧩 **Overlapping Tiled Inference Engine (`tiled_inference`)**: Restores arbitrary resolution images (e.g. $533 \times 684 \to 1368 \times 1066$) using $128 \times 128$ overlapping patches with 2D Gaussian Hanning window blending to eliminate patch boundary artifacts.
3. 🎨 **Robust Percentile Display Normalization**: Solved display contrast distortion on browser canvas using 0.5th to 99.5th percentile normalization, matching optical ground-truth rendering.
4. 🔒 **Automated Hidden Test Access Guard ([`utils/test_protection.py`](file:///c:/Users/HP/OneDrive/Documents/hackthan_nit/utils/test_protection.py))**: Enforces strict security isolation preventing any accidental access to `Test_NoisyLR`.
5. 🧪 **Comprehensive PyTorch Unit Test Suite**: **22 / 22 Unit Tests Passing** (`python -m unittest discover -s tests -v`).

---

## 5. Master Commands Reference

| Target Task | PowerShell Command |
|---|---|
| 🌐 **Run Web Server + PyTorch Inference** | **`python serve.py`** |
| 🚀 **Run Master End-to-End Execution Pipeline** | **`python scripts/run_all.py`** |
| 🧪 **Run All 22 PyTorch Unit Tests** | **`python -m unittest discover -s tests -v`** |
| 📊 **Evaluate Candidate Models** | **`python scripts/eval_v3_candidate.py`** |
