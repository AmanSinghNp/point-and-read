"""
Verify IAM line image folder structure (three-level: lines/level1/level2/line_id.png).

Usage (from point-and-read or project root):
  python scripts/verify_iam_structure.py --lines_txt data/iam/ascii/lines.txt --iam_root data/iam --n 10

Prints expected paths for the first n line IDs and whether each file exists.
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)

# Reuse path logic from parse_iam
from parse_iam import line_id_to_image_path


def main():
    parser = argparse.ArgumentParser(description="Verify IAM three-level line image paths")
    parser.add_argument("--lines_txt", default="data/iam/ascii/lines.txt")
    parser.add_argument("--iam_root", default="data/iam")
    parser.add_argument("--n", type=int, default=10, help="Number of line IDs to check")
    args = parser.parse_args()

    lines_txt = os.path.normpath(args.lines_txt)
    iam_root = os.path.normpath(args.iam_root)

    if not os.path.exists(lines_txt):
        print(f"lines.txt not found: {lines_txt}")
        sys.exit(1)

    line_ids = []
    with open(lines_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and line.split():
                line_id = line.split()[0]
                line_ids.append(line_id)
                if len(line_ids) >= args.n:
                    break

    print(f"IAM root: {iam_root}")
    print(f"Expected structure: lines/{{level1}}/{{level2}}/{{line_id}}.png\n")
    found = 0
    for lid in line_ids:
        path = line_id_to_image_path(iam_root, lid)
        exists = os.path.exists(path)
        if exists:
            found += 1
        status = "OK" if exists else "MISSING"
        print(f"  {lid} -> {path} [{status}]")
    print(f"\n{found}/{len(line_ids)} files found.")
    if found == 0:
        print("Ensure line images are extracted under iam_root/lines/ with the three-level structure.")


if __name__ == "__main__":
    main()
