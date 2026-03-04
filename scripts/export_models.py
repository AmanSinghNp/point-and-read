"""
export_models.py — Generates optimized versions of the TrOCR model.

This script creates:
1. An INT8 dynamically quantized PyTorch model (best for CPU performance).
2. An ONNX exported model (best for cross-platform and specific runtimes).

Usage:
    python scripts/export_models.py [--onnx] [--int8] [--model microsoft/trocr-base-handwritten]
"""

import argparse
import os
import time

def export_int8(model_id: str, output_dir: str):
    """Apply dynamic INT8 quantization to the PyTorch model for CPU speedups."""
    import torch
    from transformers import VisionEncoderDecoderModel
    
    print(f"Loading '{model_id}' for INT8 quantization...")
    start_time = time.time()
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    
    print("Applying dynamic quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    os.makedirs(output_dir, exist_ok=True)
    quantized_model.save_pretrained(output_dir)
    elapsed = time.time() - start_time
    print(f"Successfully saved INT8 model to '{output_dir}' in {elapsed:.1f}s.")

def export_onnx(model_id: str, output_dir: str):
    """Export the model to ONNX format using HuggingFace Optimum."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as e:
        print("ERROR: ONNX export requires the optimum package. Run: pip install optimum[onnxruntime]")
        raise e
        
    print(f"Exporting '{model_id}' to ONNX format...")
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We use main_export from optimum to do the heavy lifting.
    # Exporting without-past for stability since kv-cache export has known optimum issues.
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="image-to-text",
        batch_size=8,
        width=384,
        height=384,
        no_post_process=True # Prevent complex post-processing for better ONNX runtime stability
    )
    
    elapsed = time.time() - start_time
    print(f"Successfully saved ONNX model to '{output_dir}' in {elapsed:.1f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export optimized TrOCR models.")
    parser.add_argument("--model", type=str, default="microsoft/trocr-base-handwritten",
                        help="HuggingFace model ID or local path to base model.")
    parser.add_argument("--int8", action="store_true", help="Generate INT8 quantized PyTorch model.")
    parser.add_argument("--onnx", action="store_true", help="Generate ONNX model.")
    parser.add_argument("--all", action="store_true", help="Generate all optimized formats.")
    
    args = parser.parse_args()
    
    if not (args.int8 or args.onnx or args.all):
        print("Please specify an export target: --int8, --onnx, or --all")
        parser.print_help()
        exit(1)
        
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    
    if args.int8 or args.all:
        int8_dir = os.path.join(base_dir, "trocr-int8")
        export_int8(args.model, int8_dir)
        
    if args.onnx or args.all:
        onnx_dir = os.path.join(base_dir, "trocr-onnx")
        export_onnx(args.model, onnx_dir)
