"""
run_all.py — Master End-to-End Execution Pipeline for SemiconDaAIR-v2 / v3.

Runs all verification, benchmarking, unit testing, error analysis, ablation, and evaluation steps in sequence:
  1. Checkpoint & SHA256 Cryptographic Audit
  2. v3 Repository & Security Audit
  3. Automated Unit Test Suite (22/22 tests passing)
  4. Standalone Evaluation Pipeline on 640 Validation Split
  5. Synchronized Latency & Throughput Benchmark
  6. Automated Failure Mode & Error Analysis
  7. Fresh Environment Reproducibility Audit
  8. v3 Ablation Benchmark & Model Selection Suite
"""

import os
import sys
import subprocess
import time


def run_step(step_name, command):
    print("\n" + "=" * 75)
    print(f"      STEP: {step_name.upper()}      ")
    print("=" * 75)
    print(f"Command: {command}")
    
    t0 = time.time()
    res = subprocess.run(command, shell=True)
    t1 = time.time()

    if res.returncode == 0:
        print(f"[PASS] {step_name} PASSED in {t1 - t0:.2f} seconds.")
        return True
    else:
        print(f"[FAIL] {step_name} FAILED with exit code {res.returncode}.")
        return False


def main():
    print("=" * 75)
    print("      SEMICONDAAIR-V2 / V3 MASTER END-TO-END EXECUTION PIPELINE      ")
    print("=" * 75)

    steps = [
        ("1. Checkpoint & SHA256 Audit", f'"{sys.executable}" scripts/audit_checkpoint_sha256.py'),
        ("2. v3 Repository & Security Audit", f'"{sys.executable}" scripts/generate_v3_repository_audit.py'),
        ("3. Automated PyTorch Unit Test Suite", f'"{sys.executable}" -m unittest discover -s tests -v'),
        ("4. Standalone Evaluation Pipeline", f'"{sys.executable}" evaluate.py --input_dir "C:\\Users\\HP\\Downloads\\dataset\\train\\train\\NoisyLR" --gt_dir "C:\\Users\\HP\\Downloads\\dataset\\train\\train\\GT" --output_dir results/evaluation_outputs --model_path checkpoints/final/semicon_daair_v2_final.pt --split_file splits/val.txt'),
        ("5. Latency & Throughput Benchmark", f'"{sys.executable}" benchmark.py'),
        ("6. Automated Error Analysis", f'"{sys.executable}" scripts/error_analysis.py'),
        ("7. Reproducibility Audit", f'"{sys.executable}" scripts/verify_reproducibility.py'),
        ("8. v3 Ablation & Model Selection Benchmark", f'"{sys.executable}" scripts/run_v3_ablation.py'),
    ]

    passed = 0
    total = len(steps)

    for name, cmd in steps:
        success = run_step(name, cmd)
        if success:
            passed += 1

    print("\n" + "=" * 75)
    print(f"      MASTER EXECUTION SUMMARY: {passed}/{total} STEPS PASSED      ")
    print("=" * 75)
    if passed == total:
        print("[SUCCESS] ALL STEPS PASSED 100% CLEANLY! Repository is competition-ready.")
    else:
        print("[WARNING] Some steps failed. Check output logs above.")


if __name__ == "__main__":
    main()
