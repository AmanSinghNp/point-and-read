"""
Parse IAM lines.txt and produce data/processed/train.csv, val.csv, test.csv, and vocab.json.

Image path uses the three-level nested IAM structure:
  line_id "a01-000u-00"  ->  lines/a01/a01-000u/a01-000u-00.png
  (part before first dash) (part before second dash) (line_id.png)

Run from project root or point-and-read. Example:
  python parse_iam.py --lines_txt data/iam/ascii/lines.txt --iam_root data/iam --out_dir data/processed
"""

import argparse
import csv
import os
import random
import sys

# Allow imports from point-and-read when run from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from recognition.vocab import Vocabulary


def line_id_to_image_path(iam_root: str, line_id: str) -> str:
    """
    IAM three-level structure: lines/{part_before_1st_dash}/{part_before_2nd_dash}/{line_id}.png
    e.g. a01-000u-00 -> lines/a01/a01-000u/a01-000u-00.png
    """
    parts = line_id.split("-")
    if len(parts) < 2:
        # fallback: flat under lines/
        return os.path.join(iam_root, "lines", f"{line_id}.png")
    level1 = parts[0]
    level2 = "-".join(parts[:2])
    return os.path.join(iam_root, "lines", level1, level2, f"{line_id}.png")


def parse_lines_txt(path: str, iam_root: str, ok_only: bool = True, check_exists: bool = True):
    """
    Parse lines.txt; yield (image_path, transcription) for each valid line.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            line_id = parts[0]
            status = parts[1]
            # graylevel, n_components, x, y, w, h = parts[2:8]
            transcription = " ".join(parts[8:]).replace("|", " ").strip()
            if ok_only and status != "ok":
                continue
            if not transcription:
                continue
            image_path = line_id_to_image_path(iam_root, line_id)
            if check_exists and not os.path.exists(image_path):
                continue
            yield image_path, transcription


def main():
    parser = argparse.ArgumentParser(description="Parse IAM lines.txt -> train/val/test CSV + vocab.json")
    parser.add_argument("--lines_txt", default="data/iam/ascii/lines.txt", help="Path to IAM lines.txt")
    parser.add_argument("--iam_root", default="data/iam", help="Root of IAM data (lines/ lives under here)")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory for CSVs and vocab.json")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Val split ratio")
    parser.add_argument("--ok_only", action="store_true", default=True, help="Only use lines with status 'ok'")
    parser.add_argument("--include_err", action="store_true", help="Include 'err' lines too (overrides ok_only)")
    parser.add_argument("--no_check_exists", action="store_true", help="Do not skip missing image files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    if args.include_err:
        args.ok_only = False

    lines_txt = os.path.normpath(args.lines_txt)
    iam_root = os.path.normpath(args.iam_root)
    out_dir = os.path.normpath(args.out_dir)

    if not os.path.exists(lines_txt):
        print(f"[parse_iam] ERROR: lines.txt not found: {lines_txt}")
        sys.exit(1)

    # Collect (image_path, transcription), grouped by form for split
    rows = list(parse_lines_txt(lines_txt, iam_root, ok_only=args.ok_only, check_exists=not args.no_check_exists))
    if not rows:
        print("[parse_iam] No valid rows (check iam_root and that line images exist under lines/level1/level2/)")
        sys.exit(1)

    # Split by form (line_id form = everything before last '-XX') to avoid writer leakage
    form_to_rows = {}
    for image_path, trans in rows:
        # line_id from path: .../line_id.png
        line_id = os.path.splitext(os.path.basename(image_path))[0]
        # form: a01-000u-00 -> a01-000u
        parts = line_id.rsplit("-", 1)
        form = parts[0] if len(parts) == 2 else line_id
        form_to_rows.setdefault(form, []).append((image_path, trans))

    forms = list(form_to_rows.keys())
    random.seed(args.seed)
    random.shuffle(forms)

    n = len(forms)
    n_train = max(1, int(n * args.train_ratio))
    n_val = max(0, int(n * args.val_ratio))
    n_test = n - n_train - n_val

    train_forms = set(forms[:n_train])
    val_forms = set(forms[n_train : n_train + n_val])
    test_forms = set(forms[n_train + n_val :])

    train_rows = []
    for f in train_forms:
        train_rows.extend(form_to_rows[f])
    val_rows = [r for f in val_forms for r in form_to_rows[f]]
    test_rows = [r for f in test_forms for r in form_to_rows[f]]

    os.makedirs(out_dir, exist_ok=True)

    for name, subset in [("train.csv", train_rows), ("val.csv", val_rows), ("test.csv", test_rows)]:
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(subset)
        print(f"[parse_iam] Wrote {path} ({len(subset)} rows)")

    # Build vocab from train transcriptions only (no val/test leakage)
    train_texts = [row[1] for row in train_rows]
    vocab = Vocabulary.build_from_texts(train_texts)
    vocab_path = os.path.join(out_dir, "vocab.json")
    vocab.save(vocab_path)
    print(f"[parse_iam] Built vocabulary ({vocab.num_classes} chars) -> {vocab_path}")
    print("[parse_iam] Done. train.py can use data/processed/ as-is.")


if __name__ == "__main__":
    main()
