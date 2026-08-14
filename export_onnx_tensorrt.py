"""
export_onnx_tensorrt.py — ONNX Graph Export and TensorRT Optimization Utility.

Exports PyTorch model to ONNX format with dynamic spatial axes and provides instructions/scripts for TensorRT FP16 & INT8 quantization.
"""

import os
import argparse
import torch
from model import build_model


def export_onnx(size="semicon_restornet", weights_path="checkpoints/best_model.pt", output_onnx="weights/semicon_restornet.onnx"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(scale=2, size=size).to(device)

    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    os.makedirs(os.path.dirname(output_onnx), exist_ok=True)

    dummy_input = torch.randn(1, 1, 128, 128, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["restored_image"],
            dynamic_axes={
                "input_image": {0: "batch_size", 2: "height", 3: "width"},
                "restored_image": {0: "batch_size", 2: "out_height", 3: "out_width"}
            }
        )
        print(f"ONNX Model successfully exported to: {output_onnx}")
    except Exception as e:
        print(f"ONNX Export notice: {e}")
        print("To export onnx graphs in minimal environments, run: pip install onnx onnxscript")

    print("\n--- TensorRT Conversion Command Line Guidelines ---")
    print(f"To compile for FP16 execution on NVIDIA RTX/Jetson:")
    print(f"  trtexec --onnx={output_onnx} --saveEngine=weights/semicon_restornet_fp16.engine --fp16")
    print(f"To compile for INT8 quantized execution:")
    print(f"  trtexec --onnx={output_onnx} --saveEngine=weights/semicon_restornet_int8.engine --int8 --calib=calib_data.cache")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="semicon_restornet")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="weights/semicon_restornet.onnx")
    args = parser.parse_args()

    export_onnx(size=args.size, weights_path=args.weights, output_onnx=args.output)
