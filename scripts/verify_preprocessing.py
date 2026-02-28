"""
Run preprocessing on a few IAM (or any) line images and save outputs for visual inspection.

Usage (from point-and-read or project root):
  python scripts/verify_preprocessing.py --csv data/processed/train.csv --out_dir data/processed/preview --n 5
  python scripts/verify_preprocessing.py --csv data/processed/train.csv --out_dir data/processed/preview --n 5 --no_deskew

Saves original (resized for comparison), with deskew, and without deskew so you can compare.
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from preprocessing.clean import Preprocessor


def tensor_to_vis(tensor, mean=0.485, std=0.229):
    """Reverse normalize and scale to 0-255 for saving as PNG."""
    t = tensor.squeeze(0).clone()
    t = t * std + mean
    t = torch.clamp(t, 0, 1)
    return (t.numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Preview preprocessing on a few images")
    parser.add_argument("--csv", default="data/processed/train.csv", help="CSV with image_path,transcription")
    parser.add_argument("--out_dir", default="data/processed/preview", help="Where to save preview images")
    parser.add_argument("--n", type=int, default=5, help="Number of images to process")
    parser.add_argument("--no_deskew", action="store_true", help="Only run with deskew disabled (IAM-friendly)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[verify_preprocessing] CSV not found: {args.csv}")
        print("Run parse_iam.py first to create data/processed/train.csv")
        sys.exit(1)

    rows = []
    with open(args.csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and os.path.exists(row[0].strip()):
                rows.append((row[0].strip(), row[1].strip()))
                if len(rows) >= args.n:
                    break

    if not rows:
        print("[verify_preprocessing] No valid image paths found in CSV")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    preprocessor_deskew = Preprocessor(target_height=64, is_training=False, apply_deskew=True)
    preprocessor_no_deskew = Preprocessor(target_height=64, is_training=False, apply_deskew=False)

    for i, (img_path, trans) in enumerate(rows):
        name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            if not args.no_deskew:
                t = preprocessor_deskew.process(img_path)
                vis = tensor_to_vis(t)
                out_path = os.path.join(args.out_dir, f"{i:02d}_{name}_deskew.png")
                cv2.imwrite(out_path, vis)
                print(f"  {out_path}")

            t = preprocessor_no_deskew.process(img_path)
            vis = tensor_to_vis(t)
            suffix = "no_deskew" if not args.no_deskew else "preprocessed"
            out_path = os.path.join(args.out_dir, f"{i:02d}_{name}_{suffix}.png")
            cv2.imwrite(out_path, vis)
            print(f"  {out_path}")
        except Exception as e:
            print(f"  Skip {img_path}: {e}")

    print(f"\n[verify_preprocessing] Done. Check {args.out_dir} to compare deskew vs no_deskew for IAM.")


if __name__ == "__main__":
    main()
