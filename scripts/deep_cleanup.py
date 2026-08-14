"""
deep_cleanup.py — Performs a deep clean of redundant legacy scripts and directories,
leaving only the primary competition pipeline, models, tests, splits, and reports.
"""

import os
import stat
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories to remove if they exist and contain legacy duplicates
DIRS_TO_REMOVE = [
    os.path.join(PROJECT_ROOT, "src"),
    os.path.join(PROJECT_ROOT, "datasets"),
    os.path.join(PROJECT_ROOT, "logs"),
    os.path.join(PROJECT_ROOT, "weights"),
]

# Legacy/redundant single files in root
FILES_TO_REMOVE = [
    os.path.join(PROJECT_ROOT, "analyze_dataset.py"),
    os.path.join(PROJECT_ROOT, "benchmark.py"),
    os.path.join(PROJECT_ROOT, "benchmark_realtime.py"),
    os.path.join(PROJECT_ROOT, "export_onnx_tensorrt.py"),
    os.path.join(PROJECT_ROOT, "inspection_assurance.py"),
    os.path.join(PROJECT_ROOT, "inspect_wafer.py"),
    os.path.join(PROJECT_ROOT, "load_dncnn_pretrained.py"),
    os.path.join(PROJECT_ROOT, "eval_with_metrics.py"),
    os.path.join(PROJECT_ROOT, "serve.py"),
    os.path.join(PROJECT_ROOT, "test_api.py"),
    os.path.join(PROJECT_ROOT, "visualize.py"),
    os.path.join(PROJECT_ROOT, "Dockerfile"),
]


def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def deep_clean():
    removed_count = 0

    for dpath in DIRS_TO_REMOVE:
        if os.path.exists(dpath):
            try:
                shutil.rmtree(dpath, onerror=remove_readonly)
                print(f"Removed directory: {os.path.relpath(dpath, PROJECT_ROOT)}")
                removed_count += 1
            except Exception as e:
                print(f"Skipped {dpath}: {e}")

    for fpath in FILES_TO_REMOVE:
        if os.path.exists(fpath):
            try:
                os.chmod(fpath, stat.S_IWRITE)
                os.remove(fpath)
                print(f"Removed file: {os.path.relpath(fpath, PROJECT_ROOT)}")
                removed_count += 1
            except Exception as e:
                print(f"Skipped {fpath}: {e}")

    print(f"\nDeep cleanup completed! ({removed_count} redundant files/folders removed)")


if __name__ == "__main__":
    deep_clean()
