"""
cleanup_workspace.py — Safely removes temporary generated files and folders on Windows,
handling read-only permissions and file locks cleanly.
"""

import os
import stat
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DIRS_TO_REMOVE = [
    os.path.join(PROJECT_ROOT, "results", "baseline_outputs"),
    os.path.join(PROJECT_ROOT, "results", "self_eval"),
    os.path.join(PROJECT_ROOT, "results", "test_eval_output"),
    os.path.join(PROJECT_ROOT, "results", "visual_inspection"),
    os.path.join(PROJECT_ROOT, "results", "worst_cases"),
    os.path.join(PROJECT_ROOT, "logs", "ingestion_queue"),
    os.path.join(PROJECT_ROOT, "data", "clean_images"),
]

FILES_TO_REMOVE = [
    os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "latest_checkpoint.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "restorenet.pt"),
    os.path.join(PROJECT_ROOT, "results", "ref_check_000000.npy"),
    os.path.join(PROJECT_ROOT, "results", "ref_check_000000.png"),
    os.path.join(PROJECT_ROOT, "results", "ref_check_000000_diagnostic.png"),
    os.path.join(PROJECT_ROOT, "results", "ref_check_000001.npy"),
    os.path.join(PROJECT_ROOT, "results", "ref_check_000001.png"),
    os.path.join(PROJECT_ROOT, "results", "ref_check_000001_diagnostic.png"),
]


def remove_readonly(func, path, exc_info):
    """Clear the read-only attribute on Windows files/directories."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup():
    removed_count = 0

    # 1. Remove temporary output directories with read-only handler
    for dpath in DIRS_TO_REMOVE:
        if os.path.exists(dpath):
            try:
                shutil.rmtree(dpath, onerror=remove_readonly)
                print(f"Removed directory: {os.path.relpath(dpath, PROJECT_ROOT)}")
                removed_count += 1
            except Exception as e:
                print(f"Skipped {os.path.relpath(dpath, PROJECT_ROOT)}: {e}")

    # 2. Remove obsolete/temporary individual files
    for fpath in FILES_TO_REMOVE:
        if os.path.exists(fpath):
            try:
                os.chmod(fpath, stat.S_IWRITE)
                os.remove(fpath)
                print(f"Removed file: {os.path.relpath(fpath, PROJECT_ROOT)}")
                removed_count += 1
            except Exception as e:
                pass

    # 3. Clean up __pycache__ directories
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for dname in dirs:
            if dname == "__pycache__":
                cache_dir = os.path.join(root, dname)
                try:
                    shutil.rmtree(cache_dir, onerror=remove_readonly)
                    print(f"Removed __pycache__: {os.path.relpath(cache_dir, PROJECT_ROOT)}")
                    removed_count += 1
                except Exception as e:
                    pass

    print(f"\nCleanup completed successfully! ({removed_count} items cleaned)")


if __name__ == "__main__":
    cleanup()
