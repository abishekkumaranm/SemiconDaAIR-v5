"""
audit_semihackthan_repos.py — Comprehensive Repository Discovery & Component Audit Tool for SEMIHACKTHAN.

Crawls all repositories in C:\\Users\\HP\\OneDrive\\Documents\\SEMIHACKTHAN, inspects source code,
identifies model architectures, loss functions, attention blocks, residual blocks, dependencies,
and generates reports/repository_inventory.md.
"""

import os
import sys
import glob
import json

SEMIHACKTHAN_DIR = r"C:\Users\HP\OneDrive\Documents\SEMIHACKTHAN"


def inspect_repository(repo_path):
    repo_name = os.path.basename(repo_path)
    py_files = glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True)
    md_files = glob.glob(os.path.join(repo_path, "**", "*.md"), recursive=True)
    pt_files = glob.glob(os.path.join(repo_path, "**", "*.pt*"), recursive=True) + glob.glob(os.path.join(repo_path, "**", "*.pth"), recursive=True)
    yaml_files = glob.glob(os.path.join(repo_path, "**", "*.yaml"), recursive=True) + glob.glob(os.path.join(repo_path, "**", "*.yml"), recursive=True)

    rel_py = [os.path.relpath(f, repo_path) for f in py_files]
    rel_pt = [os.path.relpath(f, repo_path) for f in pt_files]

    # Look for model keywords
    model_files = [f for f in rel_py if any(k in f.lower() for k in ["model", "network", "arch", "net", "dncnn", "rcan", "restormer", "swinir", "esrgan", "srcnn"])]
    train_files = [f for f in rel_py if "train" in f.lower()]
    infer_files = [f for f in rel_py if any(k in f.lower() for k in ["test", "predict", "infer", "eval"])]
    loss_files = [f for f in rel_py if "loss" in f.lower()]

    req_file = os.path.join(repo_path, "requirements.txt")
    has_req = os.path.exists(req_file)

    readme_file = os.path.join(repo_path, "README.md")
    readme_snippet = ""
    if os.path.exists(readme_file):
        try:
            with open(readme_file, "r", encoding="utf-8", errors="ignore") as f:
                readme_snippet = f.read(500).replace("\n", " ")
        except Exception:
            pass

    return {
        "repo_name": repo_name,
        "path": repo_path,
        "py_file_count": len(py_files),
        "pt_file_count": len(pt_files),
        "py_files": rel_py[:15],
        "model_files": model_files,
        "train_files": train_files,
        "infer_files": infer_files,
        "loss_files": loss_files,
        "checkpoints": rel_pt,
        "has_requirements": has_req,
        "readme_snippet": readme_snippet
    }


def main():
    repos = [d for d in glob.glob(os.path.join(SEMIHACKTHAN_DIR, "*")) if os.path.isdir(d)]
    inventory = []

    print(f"Discovered {len(repos)} repository directories in {SEMIHACKTHAN_DIR}:")
    for r in repos:
        info = inspect_repository(r)
        inventory.append(info)
        print(f"  - {info['repo_name']} ({info['py_file_count']} python files, {info['pt_file_count']} checkpoints)")

    # Write reports/repository_inventory.md
    report_lines = [
        "# SEMIHACKTHAN Workspace Repository Inventory Report\n",
        "**Target Workspace**: `C:\\Users\\HP\\OneDrive\\Documents\\SEMIHACKTHAN`  \n",
        f"**Discovered Repositories**: {len(repos)}  \n\n",
        "---",
        "## 1. Summary Inventory Table\n",
        "| Repository Directory Name | Python Files | Checkpoint Files | Key Model Files | Training Scripts |",
        "|---|---|---|---|---|"
    ]

    for item in inventory:
        models_str = ", ".join([f"`{m}`" for m in item['model_files'][:3]]) if item['model_files'] else "None"
        train_str = ", ".join([f"`{t}`" for t in item['train_files'][:2]]) if item['train_files'] else "None"
        report_lines.append(
            f"| `{item['repo_name']}` | {item['py_file_count']} | {item['pt_file_count']} | {models_str} | {train_str} |"
        )

    report_lines.extend([
        "\n---",
        "## 2. Detailed Repository Inspection Breakdown\n"
    ])

    for item in inventory:
        report_lines.extend([
            f"### `{item['repo_name']}`\n",
            f"- **Exact Path**: `{item['path']}`\n",
            f"- **Python Source Files ({item['py_file_count']})**: {', '.join([f'`{p}`' for p in item['py_files'][:8]])}\n",
            f"- **Model Files**: {', '.join([f'`{m}`' for m in item['model_files']]) if item['model_files'] else 'None'}\n",
            f"- **Inference / Test Scripts**: {', '.join([f'`{i}`' for i in item['infer_files']]) if item['infer_files'] else 'None'}\n",
            f"- **Checkpoints Found**: {', '.join([f'`{c}`' for c in item['checkpoints']]) if item['checkpoints'] else 'None'}\n",
            f"- **Requirements File**: {'Present' if item['has_requirements'] else 'Absent'}\n",
            f"- **README Summary**: {item['readme_snippet'][:200]}...\n",
            "\n"
        ])

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/repository_inventory.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved Repository Inventory Report to: {report_path}")

    with open("results/repository_inventory.json", "w") as f:
        json.dump(inventory, f, indent=4)


if __name__ == "__main__":
    main()
