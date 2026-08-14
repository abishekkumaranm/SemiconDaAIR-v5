"""
app.py — SemiconDaAIR-v5 Complete 5-Tab KLA Local Metrology Restoration Suite.

Tabs:
  1. 🔬 Single Restoration: Single file (.npy, .png, .jpg, .tif) & One-click 000000.npy sample loader.
  2. 📂 KLA Test Folder: Directory batch uploader (file_count="directory") & Optional GT folder evaluation.
  3. 📊 Batch Results: Batch summary statistics, Restored Image Gallery (20 items), ZIP Download, JSON/CSV reports.
  4. 🧠 Model Information: Architecture specifications (555,141 params, checkpoint, modules).
  5. ⚙️ System Information: Laptop hardware diagnostics (CPU, RTX 3050 GPU, VRAM, PyTorch, CUDA).
"""

import os
import sys
import time
import json
import csv
import zipfile
import shutil
import tempfile
import cv2
import numpy as np
import torch
import gradio as gr

sys_path = os.path.abspath(os.path.dirname(__file__))
if sys_path not in sys.path:
    sys.path.append(sys_path)

from models.semicon_daair_v5 import build_semicon_daair_v5
from utils.device import select_device, get_device_name
from utils.preprocessing import load_image_exact, save_image_exact
from evaluation.metrics import compute_psnr, compute_ssim

# Load Model once globally
DEVICE = select_device("auto")
CKPT_PATH = "checkpoints/v5_backup/semicon_daair_v5_candidate.pt"

print("=" * 75)
print("     [STARTING SemiconDaAIR-v5 5-TAB KLA LOCAL METROLOGY APP]     ")
print("=" * 75)
print(f"Loading Model Checkpoint: {CKPT_PATH}")
print(f"Device Selected         : {get_device_name(DEVICE)}")

MODEL = build_semicon_daair_v5(scale=2).to(DEVICE)
if os.path.exists(CKPT_PATH):
    st = torch.load(CKPT_PATH, map_location=DEVICE)
    st = st["model_state"] if isinstance(st, dict) and "model_state" in st else st
    MODEL.load_state_dict(st, strict=True)
    print("Checkpoint state dict loaded successfully (0 missing keys).")
else:
    print(f"[WARNING] Checkpoint {CKPT_PATH} not found!")

MODEL.eval()
TOTAL_PARAMS = sum(p.numel() for p in MODEL.parameters())


# --------------------------------------------------------------------
# TAB 1: Single File Processing
# --------------------------------------------------------------------
def process_single_file(file_path):
    if file_path is None or not os.path.exists(str(file_path)):
        return (
            None, None, None,
            "❌ ERROR: Please select or upload a valid inspection file (.npy, .png, .jpg, .tif).",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        )

    filepath_str = str(file_path)
    ext = os.path.splitext(filepath_str)[1].lower()

    if ext not in [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        return (
            None, None, None,
            f"❌ ERROR: Unsupported file extension '{ext}'. Only .npy, .png, .jpg, .tif allowed.",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        )

    try:
        lq_np = load_image_exact(filepath_str)
    except Exception as e:
        return (
            None, None, None,
            f"❌ ERROR: Failed to load file: {e}",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        )

    if not np.isfinite(lq_np).all():
        return (
            None, None, None,
            "❌ ERROR: Input array contains non-finite values (NaN or Inf). Rejected.",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        )

    if lq_np.ndim != 2:
        return (
            None, None, None,
            f"❌ ERROR: Input array must be 2D grayscale [H, W], got shape {lq_np.shape}",
            "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        )

    in_h, in_w = lq_np.shape
    in_dtype = str(lq_np.dtype)
    in_min = float(lq_np.min())
    in_max = float(lq_np.max())

    lq_tensor = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        out_tensor = MODEL(lq_tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    pred_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
    out_h, out_w = pred_np.shape
    out_min = float(pred_np.min())
    out_max = float(pred_np.max())
    is_finite = bool(np.isfinite(pred_np).all())

    gradio_out_dir = os.path.join("outputs", "gradio")
    os.makedirs(gradio_out_dir, exist_ok=True)

    npy_out_path = os.path.join(gradio_out_dir, "restored.npy")
    png_out_path = os.path.join(gradio_out_dir, "restored.png")

    np.save(npy_out_path, pred_np.astype(np.float32))
    save_image_exact(pred_np, png_out_path)

    p_min, p_max = np.percentile(pred_np, (0.5, 99.5))
    if p_max > p_min:
        norm = np.clip((pred_np - p_min) / (p_max - p_min), 0.0, 1.0)
    else:
        norm = np.clip(pred_np, 0.0, 1.0)
    disp_u8 = (norm * 255.0).round().astype(np.uint8)

    status_msg = f"SUCCESS: SemiconDaAIR-v5 restoration completed in {latency_ms:.2f} ms!"

    in_shape_str = f"{in_w} × {in_h}"
    out_shape_str = f"{out_w} × {out_h} (2x Expansion)"
    in_range_str = f"[{in_min:.4f}, {in_max:.4f}]"
    out_range_str = f"[{out_min:.4f}, {out_max:.4f}]"
    lat_str = f"{latency_ms:.2f} ms"
    dev_str = get_device_name(DEVICE)
    params_str = f"{TOTAL_PARAMS:,}"
    nan_str = "PASS (0 NaNs / 0 Infs)" if is_finite else "FAIL (NaN Detected)"
    model_str = "SemiconDaAIR-v5"

    return (
        disp_u8, npy_out_path, png_out_path,
        status_msg, in_shape_str, out_shape_str, in_dtype,
        in_range_str, out_range_str, dev_str, lat_str, params_str, nan_str, model_str
    )


def load_sample_file():
    sample_path = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR\000000.npy"
    if os.path.exists(sample_path):
        return sample_path
    return None


# --------------------------------------------------------------------
# TAB 2 & TAB 3: KLA Test Folder Batch Processing
# --------------------------------------------------------------------
def process_batch_folder(file_list, gt_folder_path=None, progress=gr.Progress()):
    if not file_list:
        return (
            "❌ ERROR: No files selected. Please select a folder or multiple files.",
            "N/A", [], None, None, None
        )

    # Convert Gradio File objects to string paths
    file_paths = []
    for f in file_list:
        if isinstance(f, str):
            file_paths.append(f)
        elif hasattr(f, "name"):
            file_paths.append(f.name)
        elif isinstance(f, dict) and "name" in f:
            file_paths.append(f["name"])

    supported_exts = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    valid_paths = [f for f in file_paths if os.path.splitext(f)[1].lower() in supported_exts]
    valid_paths.sort(key=lambda x: os.path.basename(x))

    if not valid_paths:
        return (
            "❌ ERROR: No supported files (.npy, .png, .jpg, .tif) found in selection.",
            "N/A", [], None, None, None
        )

    base_out_dir = os.path.join("outputs", "kla_test_restored")
    arrays_dir = os.path.join(base_out_dir, "arrays")
    images_dir = os.path.join(base_out_dir, "images")
    comps_dir = os.path.join(base_out_dir, "comparisons")
    reports_dir = os.path.join(base_out_dir, "reports")

    for d in [arrays_dir, images_dir, comps_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    total_files = len(valid_paths)
    successful = 0
    failed = 0
    nan_files = 0
    inf_files = 0
    latencies = []
    results_records = []

    has_gt = bool(gt_folder_path) and os.path.exists(str(gt_folder_path))
    psnr_list, ssim_list, mae_list = [], [], []

    gallery_images = []
    start_time_all = time.perf_counter()

    for idx, fpath in enumerate(valid_paths):
        fname = os.path.basename(fpath)
        base_name = os.path.splitext(fname)[0]

        progress((idx + 1) / total_files, f"Processing {fname} ({idx + 1}/{total_files})")

        rec = {
            "filename": fname,
            "status": "SUCCESS",
            "error": ""
        }

        try:
            lq_np = load_image_exact(fpath)
            if not np.isfinite(lq_np).all():
                nan_files += 1
                failed += 1
                rec["status"] = "FAILED"
                rec["error"] = "Input contains NaNs or Infs"
                results_records.append(rec)
                continue

            in_h, in_w = lq_np.shape
            rec["input_height"] = in_h
            rec["input_width"] = in_w
            rec["input_dtype"] = str(lq_np.dtype)
            rec["input_min"] = float(lq_np.min())
            rec["input_max"] = float(lq_np.max())

            lq_tensor = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.inference_mode():
                out_tensor = MODEL(lq_tensor)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)
            rec["latency_ms"] = round(lat_ms, 2)

            pred_np = out_tensor.squeeze(0).squeeze(0).cpu().numpy()
            out_h, out_w = pred_np.shape
            rec["output_height"] = out_h
            rec["output_width"] = out_w
            rec["output_min"] = float(pred_np.min())
            rec["output_max"] = float(pred_np.max())

            if not np.isfinite(pred_np).all():
                inf_files += 1
                failed += 1
                rec["status"] = "FAILED"
                rec["error"] = "Output contains NaNs or Infs"
                results_records.append(rec)
                continue

            # Save restored array and visual PNG
            out_npy_path = os.path.join(arrays_dir, f"{base_name}_restored.npy")
            out_png_path = os.path.join(images_dir, f"{base_name}_restored.png")

            np.save(out_npy_path, pred_np.astype(np.float32))
            save_image_exact(pred_np, out_png_path)

            # Create side-by-side comparison image
            p_min, p_max = np.percentile(pred_np, (0.5, 99.5))
            norm_res = np.clip((pred_np - p_min) / (p_max - p_min), 0.0, 1.0) if p_max > p_min else np.clip(pred_np, 0.0, 1.0)
            res_u8 = (norm_res * 255.0).round().astype(np.uint8)

            norm_in = np.clip(lq_np, 0.0, 1.0)
            in_u8 = (norm_in * 255.0).round().astype(np.uint8)
            in_u8_resized = cv2.resize(in_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

            if has_gt:
                gt_path = os.path.join(str(gt_folder_path), fname)
                if os.path.exists(gt_path):
                    gt_np = load_image_exact(gt_path)
                    psnr_val = compute_psnr(norm_res, np.clip(gt_np, 0.0, 1.0))
                    ssim_val = compute_ssim(norm_res, np.clip(gt_np, 0.0, 1.0))
                    mae_val = float(np.mean(np.abs(norm_res - np.clip(gt_np, 0.0, 1.0))))
                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)
                    mae_list.append(mae_val)
                    rec["psnr_db"] = round(float(psnr_val), 4)
                    rec["ssim"] = round(float(ssim_val), 4)
                    rec["mae"] = round(float(mae_val), 4)

                    gt_u8 = cv2.resize((np.clip(gt_np, 0.0, 1.0) * 255.0).astype(np.uint8), (out_w, out_h))
                    comp_panel = np.hstack([in_u8_resized, res_u8, gt_u8])
                else:
                    comp_panel = np.hstack([in_u8_resized, res_u8])
            else:
                comp_panel = np.hstack([in_u8_resized, res_u8])

            comp_path = os.path.join(comps_dir, f"{base_name}_comparison.png")
            cv2.imwrite(comp_path, comp_panel)

            if len(gallery_images) < 20:
                gallery_images.append(out_png_path)

            successful += 1
            results_records.append(rec)

        except Exception as ex:
            failed += 1
            rec["status"] = "FAILED"
            rec["error"] = str(ex)
            results_records.append(rec)

    total_time_sec = time.perf_counter() - start_time_all
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    min_lat = float(np.min(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0

    # Save reports
    json_report_path = os.path.join(reports_dir, "batch_report.json")
    csv_report_path = os.path.join(reports_dir, "batch_report.csv")

    summary_report = {
        "model": "SemiconDaAIR-v5",
        "checkpoint": CKPT_PATH,
        "device": get_device_name(DEVICE),
        "total_files": total_files,
        "successful": successful,
        "failed": failed,
        "nan_files": nan_files,
        "inf_files": inf_files,
        "total_time_seconds": round(total_time_sec, 2),
        "average_latency_ms": round(avg_lat, 2),
        "fastest_latency_ms": round(min_lat, 2),
        "slowest_latency_ms": round(max_lat, 2),
        "has_gt_evaluation": has_gt,
        "mean_psnr_db": round(float(np.mean(psnr_list)), 4) if psnr_list else "N/A (Ground Truth Not Provided)",
        "mean_ssim": round(float(np.mean(ssim_list)), 4) if ssim_list else "N/A",
        "results": results_records
    }

    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    if results_records:
        keys = list(results_records[0].keys())
        with open(csv_report_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_records)

    # Create outputs/kla_test_restored.zip
    zip_path = os.path.join("outputs", "kla_test_restored.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    shutil.make_archive(os.path.join("outputs", "kla_test_restored"), "zip", base_out_dir)

    summary_msg = f"KLA BATCH RESTORATION COMPLETE: {successful}/{total_files} Processed in {total_time_sec:.2f} s!"

    summary_details = f"""### 📊 KLA Batch Restoration Summary
- **Total Files**: `{total_files}`
- **Successful**: `{successful}` | **Failed**: `{failed}`
- **NaN / Inf Errors**: `{nan_files + inf_files}`
- **Average GPU Latency**: `{avg_lat:.2f} ms/sample` (Fastest: `{min_lat:.2f} ms`, Slowest: `{max_lat:.2f} ms`)
- **Total Processing Time**: `{total_time_sec:.2f} seconds`
- **Ground Truth Evaluation**: `{summary_report['mean_psnr_db']} PSNR`
- **Output Directory**: [`outputs/kla_test_restored/`](file:///{os.path.abspath(base_out_dir)})
"""

    return summary_msg, summary_details, gallery_images, zip_path, json_report_path, csv_report_path


def load_local_dataset():
    local_dir = r"C:\Users\HP\Downloads\dataset\train\train\NoisyLR"
    if not os.path.exists(local_dir):
        return None, "❌ Local directory C:\\Users\\HP\\Downloads\\dataset\\train\\train\\NoisyLR not found!"

    files = [os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.lower().endswith(".npy")]
    files.sort()
    return files, f"Found {len(files)} .npy files in local dataset directory!"


# --------------------------------------------------------------------
# GRADIO 5-TAB UI BUILDER
# --------------------------------------------------------------------
def build_gradio_app():
    with gr.Blocks(title="SemiconDaAIR-v5 KLA Metrology Suite") as demo:
        gr.Markdown(
            """
            # 🔬 SemiconDaAIR-v5 Local KLA Metrology Suite
            ### AI Semiconductor Inspection Restoration & Batch Processing Engine
            **KLA / SEMICON India Hackathon 2026** — *Real PyTorch GPU Inference (28.0340 dB PSNR)*
            """
        )

        with gr.Tabs():
            # TAB 1: Single Restoration
            with gr.TabItem("🔬 Single Restoration"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload KLA Degraded Inspection Data")
                        file_input = gr.File(
                            label="Upload Inspection File (.npy, .png, .jpg, .tif)",
                            file_types=[".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                            type="filepath"
                        )
                        with gr.Row():
                            sample_btn = gr.Button("📂 Load Sample 000000.npy")
                            restore_btn = gr.Button("⚡ RESTORE IMAGE / ARRAY", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📥 SemiconDaAIR-v5 Restored Output")
                        output_visual = gr.Image(label="Restored Image Display (2x PixelShuffle)", image_mode="L")
                        with gr.Row():
                            npy_download = gr.File(label="Download Restored NPY (.npy)")
                            png_download = gr.File(label="Download Restored PNG (.png)")

                status_box = gr.Textbox(label="Restoration Status", value="Ready. Upload file or click Load Sample 000000.npy.")

                gr.Markdown("### 📊 Inspection Dashboard")
                with gr.Row():
                    model_box = gr.Textbox(label="Model Architecture")
                    param_box = gr.Textbox(label="Model Parameters")
                    dev_box = gr.Textbox(label="Hardware Device")
                    lat_box = gr.Textbox(label="Inference Latency")

                with gr.Row():
                    in_shape_box = gr.Textbox(label="Input Shape (H × W)")
                    out_shape_box = gr.Textbox(label="Output Shape (2H × 2W)")
                    in_dtype_box = gr.Textbox(label="Input Data Type")
                    nan_box = gr.Textbox(label="NaN / Inf Verification")

                with gr.Row():
                    in_range_box = gr.Textbox(label="Input Dynamic Range [Min, Max]")
                    out_range_box = gr.Textbox(label="Output Dynamic Range [Min, Max]")

                sample_btn.click(
                    fn=load_sample_file,
                    inputs=[],
                    outputs=[file_input]
                )

                restore_btn.click(
                    fn=process_single_file,
                    inputs=[file_input],
                    outputs=[
                        output_visual, npy_download, png_download,
                        status_box, in_shape_box, out_shape_box, in_dtype_box,
                        in_range_box, out_range_box, dev_box, lat_box, param_box, nan_box, model_box
                    ]
                )

            # TAB 2: KLA Test Folder
            with gr.TabItem("📂 KLA Test Folder"):
                gr.Markdown("### 📂 KLA Test Folder Batch Restoration")
                with gr.Row():
                    with gr.Column(scale=1):
                        folder_input = gr.File(
                            label="Select KLA Test Folder / Multiple Files",
                            file_count="directory",
                            file_types=[".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                            type="filepath"
                        )
                        gt_folder_input = gr.Textbox(
                            label="Optional Ground Truth Folder Path (For PSNR/SSIM Evaluation)",
                            placeholder="e.g. C:\\Users\\HP\\Downloads\\dataset\\train\\train\\GT"
                        )
                        with gr.Row():
                            local_ds_btn = gr.Button("📂 Open Local KLA Dataset")
                            batch_restore_btn = gr.Button("🚀 RESTORE ALL KLA TEST IMAGES", variant="primary")

                    with gr.Column(scale=1):
                        batch_status = gr.Textbox(label="Batch Status", value="Ready to process folder.")
                        batch_details_md = gr.Markdown("### Processing Details\nSelect a folder and click Restore All.")

                local_ds_btn.click(
                    fn=load_local_dataset,
                    inputs=[],
                    outputs=[folder_input, batch_status]
                )

            # TAB 3: Batch Results
            with gr.TabItem("📊 Batch Results"):
                gr.Markdown("### 📊 Restored Outputs Gallery & Reports")
                gallery_comp = gr.Gallery(label="Restored Results Gallery (Preview up to 20 images)", columns=4, height="auto")
                with gr.Row():
                    zip_download = gr.File(label="Download Complete KLA Restoration ZIP (.zip)")
                    json_download = gr.File(label="Download Batch JSON Report (.json)")
                    csv_download = gr.File(label="Download Batch CSV Report (.csv)")

                batch_restore_btn.click(
                    fn=process_batch_folder,
                    inputs=[folder_input, gt_folder_input],
                    outputs=[batch_status, batch_details_md, gallery_comp, zip_download, json_download, csv_download]
                )

            # TAB 4: Model Information
            with gr.TabItem("🧠 Model Information"):
                gr.Markdown(
                    rf"""
                    ## 🧠 SemiconDaAIR-v5 Model Architecture & Specifications
                    - **Model Name**: `SemiconDaAIR-v5`
                    - **Checkpoint Path**: `{CKPT_PATH}`
                    - **Parameter Count**: `{TOTAL_PARAMS:,} parameters` (2.22 MB disk size)
                    - **Input Dynamic Range**: Unbounded float32 `[-0.2786, 2.1580]`
                    - **Range Normalization**: `RobustAsinhRangeHandler` (asinh(X / scale))
                    - **Reconstruction Head**: `FidelityGatedHead` ($Y_{{HR}} = Bilinear(X) + C(x,y) \cdot R(x,y)$)
                    - **Validated Performance**: **`28.0340 dB PSNR`** | **`0.7448 SSIM`** | **`0.0325 MAE`**
                    """
                )

            # TAB 5: System Information
            with gr.TabItem("⚙️ System Information"):
                import platform
                gr.Markdown(
                    f"""
                    ## ⚙️ Laptop Hardware & System Diagnostics
                    - **Operating System**: `{platform.system()} {platform.machine()} ({platform.version()})`
                    - **Python Version**: `{platform.python_version()}`
                    - **PyTorch Version**: `{torch.__version__}`
                    - **Hardware Device**: `{get_device_name(DEVICE)}`
                    - **GPU VRAM**: `{round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else '0.0'} GB`
                    """
                )

        gr.Markdown("---")
        gr.Markdown("**SemiconDaAIR-v5** — *Protected Champion Checkpoint (`checkpoints/v5_backup/semicon_daair_v5_candidate.pt`)*")

    return demo


if __name__ == "__main__":
    app = build_gradio_app()
    # Add allowed_paths to resolve Gradio file access restrictions
    allowed_dirs = [
        os.path.abspath("outputs"),
        os.path.abspath(r"C:\Users\HP\Downloads"),
        os.path.abspath(tempfile.gettempdir())
    ]
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            show_error=True,
            allowed_paths=allowed_dirs
        )
    except OSError:
        # Fallback to automatic port assignment if 7860 is already in use
        print("[NOTICE] Port 7860 in use. Launching on next available port...")
        app.launch(
            server_name="127.0.0.1",
            inbrowser=True,
            show_error=True,
            allowed_paths=allowed_dirs
        )
