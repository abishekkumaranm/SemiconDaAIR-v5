"""
test_protection.py — Automated Hidden Test Set Access Guard for SemiconDaAIR-v3.

Enforces strict isolation of C:\\Users\\HP\\Downloads\\dataset\\Test_NoisyLR.
Any attempt to open, list, load, or read from hidden test directories during development,
training, ablation, or OOD evaluation raises an explicit SecurityRuntimeError.
"""

import os
import sys

FORBIDDEN_TEST_KEYWORDS = [
    "test_noisylr",
    "test_noisy",
    "hidden_test",
    "test_set_klA"
]


class SecurityRuntimeError(RuntimeError):
    """Raised when forbidden hidden test set access is detected."""
    pass


def assert_not_hidden_test_path(path_str: str):
    """Checks path string and terminates if hidden test dataset path is accessed."""
    if not path_str:
        return
    norm_path = os.path.normpath(str(path_str)).lower()
    for kw in FORBIDDEN_TEST_KEYWORDS:
        if kw in norm_path:
            raise SecurityRuntimeError(
                f"[SECURITY VIOLATION] Access to hidden test dataset is STRICTLY FORBIDDEN during v3 development! "
                f"Attempted path: '{path_str}'. Development must rely exclusively on splits/val.txt (640 samples)."
            )


def verify_split_isolation(train_split_file: str, val_split_file: str):
    """Verifies 100% disjoint isolation between training and validation splits."""
    with open(train_split_file, "r") as f:
        train_files = set(line.strip() for line in f if line.strip())
    with open(val_split_file, "r") as f:
        val_files = set(line.strip() for line in f if line.strip())

    intersection = train_files.intersection(val_files)
    if len(intersection) > 0:
        raise RuntimeError(f"[DATA LEAKAGE ERROR] Found {len(intersection)} overlapping files between train and val splits!")

    return len(train_files), len(val_files)


if __name__ == "__main__":
    print("[TEST PROTECTION] Testing hidden test set protection module...")
    try:
        assert_not_hidden_test_path(r"C:\Users\HP\Downloads\dataset\Test_NoisyLR\000001.npy")
        print("[FAIL] Protection failed to trigger!")
    except SecurityRuntimeError as e:
        print(f"[PASS] Successfully caught forbidden access: {e}")
