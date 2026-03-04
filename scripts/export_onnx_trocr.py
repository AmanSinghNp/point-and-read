"""
export_onnx_trocr.py -- Export TrOCR model to ONNX with Optimum CLI.

Usage:
    python scripts/export_onnx_trocr.py
    python scripts/export_onnx_trocr.py --model microsoft/trocr-small-handwritten --out onnx_trocr_small
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TrOCR model to ONNX.")
    parser.add_argument(
        "--model",
        default="microsoft/trocr-base-handwritten",
        help="HuggingFace model id or local model path.",
    )
    parser.add_argument(
        "--out",
        default="onnx_trocr_model",
        help="Output directory for ONNX export.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        args.model,
        "--task",
        "vision2seq-lm",
        args.out,
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "optimum-cli not found. Install with: pip install optimum[onnxruntime]"
        ) from exc

    print(f"ONNX export complete: {args.out}")
    print("Set POINTREAD_INFERENCE_BACKEND=onnx to force ONNX runtime.")


if __name__ == "__main__":
    main()

