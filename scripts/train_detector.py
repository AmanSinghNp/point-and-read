"""
Train a YOLO text-line detector on IAM-derived annotations.

Typical usage:
  python scripts/train_detector.py --data dataset.yaml --epochs 50 --imgsz 640 --batch 16 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO detector for handwritten text lines.")
    parser.add_argument("--data", default="dataset.yaml", help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO base model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for training, e.g. 'cpu', '0', or '0,1'.",
    )
    parser.add_argument("--project", default="runs/detect", help="Project output directory.")
    parser.add_argument("--name", default="iam_line_detector", help="Run name.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=25, help="Early-stop patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[train_detector] ERROR: data config not found: {data_path}")
        print("[train_detector] Run dataset prep first, e.g.:")
        print(
            "  python scripts/prep_yolo_dataset.py "
            "--xml_dir data/iam/xml --images_dir data/iam/forms "
            "--out_dir data/yolo_dataset --split_csv_dir data/processed"
        )
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[train_detector] ERROR: ultralytics is not installed.")
        print("[train_detector] Install it with:")
        print("  pip install ultralytics")
        sys.exit(1)

    print(f"[train_detector] Data: {data_path}")
    print(f"[train_detector] Model: {args.model}")
    print(
        f"[train_detector] epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={args.batch}, device={args.device}"
    )

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        cache=args.cache,
    )

    run_dir = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"[train_detector] Training complete. Best weights expected at: {run_dir}")


if __name__ == "__main__":
    # Keep OpenMP warnings quieter on some Windows setups.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
