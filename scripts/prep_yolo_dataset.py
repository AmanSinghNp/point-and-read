"""
Prepare IAM form XML annotations as a YOLO line-detection dataset.

This script converts IAM XML line boxes (or line boxes derived from word boxes)
into YOLO labels and builds:
  - <out_dir>/images/{train,val,test}/
  - <out_dir>/labels/{train,val,test}/
  - <out_dir>/data.yaml

Examples:
  python scripts/prep_yolo_dataset.py ^
    --xml_dir data/iam/xml ^
    --images_dir data/iam/forms ^
    --out_dir data/yolo_dataset

  python scripts/prep_yolo_dataset.py ^
    --xml_dir data/iam/xml ^
    --images_dir data/iam/forms ^
    --out_dir data/yolo_dataset ^
    --split_csv_dir data/processed
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPT_DIR)


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass
class FormAnnotation:
    form_id: str
    image_path: str
    bboxes: list[tuple[float, float, float, float]]


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bbox_from_attrs(attrs: dict[str, str]) -> tuple[float, float, float, float] | None:
    x = _to_float(
        attrs.get("x")
        or attrs.get("left")
        or attrs.get("l")
        or attrs.get("xmin")
        or attrs.get("x1")
    )
    y = _to_float(
        attrs.get("y")
        or attrs.get("top")
        or attrs.get("t")
        or attrs.get("ymin")
        or attrs.get("y1")
    )
    w = _to_float(attrs.get("width") or attrs.get("w"))
    h = _to_float(attrs.get("height") or attrs.get("h"))

    if (w is None or h is None) and x is not None and y is not None:
        x2 = _to_float(
            attrs.get("right")
            or attrs.get("r")
            or attrs.get("xmax")
            or attrs.get("x2")
        )
        y2 = _to_float(
            attrs.get("bottom")
            or attrs.get("b")
            or attrs.get("ymax")
            or attrs.get("y2")
        )
        if x2 is not None and y2 is not None:
            w = x2 - x
            h = y2 - y

    if x is None or y is None or w is None or h is None:
        return None
    if w <= 1 or h <= 1:
        return None
    return (x, y, w, h)


def _bbox_from_line_words(line_elem: ET.Element) -> tuple[float, float, float, float] | None:
    word_boxes: list[tuple[float, float, float, float]] = []
    for child in line_elem.iter():
        if _local_name(child.tag).lower() != "word":
            continue
        bbox = _bbox_from_attrs(child.attrib)
        if bbox is not None:
            word_boxes.append(bbox)

    if not word_boxes:
        return None

    x_min = min(x for x, _, _, _ in word_boxes)
    y_min = min(y for _, y, _, _ in word_boxes)
    x_max = max(x + w for x, _, w, _ in word_boxes)
    y_max = max(y + h for _, y, _, h in word_boxes)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _extract_line_boxes(xml_path: Path) -> list[tuple[float, float, float, float]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes: list[tuple[float, float, float, float]] = []
    for elem in root.iter():
        if _local_name(elem.tag).lower() != "line":
            continue
        bbox = _bbox_from_attrs(elem.attrib)
        if bbox is None:
            bbox = _bbox_from_line_words(elem)
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def _build_image_index(images_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue
        # Keep first match for each form stem.
        index.setdefault(path.stem, str(path))
    return index


def _form_id_from_line_id(line_id: str) -> str:
    parts = line_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else line_id


def _load_form_splits_from_csv_dir(csv_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in SPLITS:
        csv_path = csv_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                image_path = row[0].strip()
                if not image_path:
                    continue
                line_id = Path(image_path).stem
                form_id = _form_id_from_line_id(line_id)
                split_map.setdefault(form_id, split)
    return split_map


def _assign_splits_random(
    form_ids: list[str],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to < 1.0")

    ids = list(form_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    split_map: dict[str, str] = {}
    for fid in ids:
        if fid in train_ids:
            split_map[fid] = "train"
        elif fid in val_ids:
            split_map[fid] = "val"
        else:
            split_map[fid] = "test"
    return split_map


def _safe_norm(v: float, max_v: int) -> float:
    if max_v <= 0:
        return 0.0
    return min(1.0, max(0.0, v / float(max_v)))


def _to_yolo_lines(
    bboxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    class_id: int,
) -> list[str]:
    lines: list[str] = []
    for x, y, w, h in bboxes:
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(img_w), x + w)
        y2 = min(float(img_h), y + h)
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            continue

        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        lines.append(
            f"{class_id} "
            f"{_safe_norm(xc, img_w):.6f} "
            f"{_safe_norm(yc, img_h):.6f} "
            f"{_safe_norm(bw, img_w):.6f} "
            f"{_safe_norm(bh, img_h):.6f}"
        )
    return lines


def _ensure_dir_structure(out_dir: Path) -> None:
    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _place_image(src: Path, dst: Path, mode: str) -> None:
    if mode == "none":
        return
    if dst.exists():
        return
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _write_data_yaml(out_dir: Path, class_name: str) -> None:
    yaml_path = out_dir / "data.yaml"
    content = (
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> None:
    xml_dir = Path(args.xml_dir)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    if not xml_dir.exists():
        raise FileNotFoundError(f"xml_dir not found: {xml_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir}")

    print(f"[prep_yolo] Indexing images under: {images_dir}")
    image_index = _build_image_index(images_dir)
    if not image_index:
        raise RuntimeError(f"No images found under {images_dir}")

    annotations: list[FormAnnotation] = []
    xml_files = sorted(p for p in xml_dir.rglob("*.xml") if p.is_file())
    for xml_path in xml_files:
        form_id = xml_path.stem
        image_path = image_index.get(form_id)
        if image_path is None:
            continue
        boxes = _extract_line_boxes(xml_path)
        annotations.append(FormAnnotation(form_id=form_id, image_path=image_path, bboxes=boxes))

    if not annotations:
        raise RuntimeError(
            "No form annotations found. Check XML/image directories and file naming."
        )

    form_ids = sorted({a.form_id for a in annotations})
    if args.split_csv_dir:
        split_map = _load_form_splits_from_csv_dir(Path(args.split_csv_dir))
        if split_map:
            for fid in form_ids:
                split_map.setdefault(fid, "train")
            print("[prep_yolo] Using split map from CSV directory.")
        else:
            split_map = _assign_splits_random(
                form_ids,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
            print("[prep_yolo] CSV split dir provided but no usable entries; fell back to random split.")
    else:
        split_map = _assign_splits_random(
            form_ids,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print("[prep_yolo] Using random form-level split.")

    _ensure_dir_structure(out_dir)

    total_labels = 0
    per_split_counts = {s: 0 for s in SPLITS}
    per_split_images = {s: 0 for s in SPLITS}
    for ann in annotations:
        split = split_map.get(ann.form_id, "train")
        if split not in SPLITS:
            split = "train"

        src_img = Path(ann.image_path)
        img = cv2.imread(str(src_img))
        if img is None:
            continue
        h, w = img.shape[:2]
        yolo_lines = _to_yolo_lines(ann.bboxes, w, h, args.class_id)
        total_labels += len(yolo_lines)

        dst_img = out_dir / "images" / split / src_img.name
        _place_image(src_img, dst_img, args.image_mode)

        dst_lbl = out_dir / "labels" / split / f"{src_img.stem}.txt"
        dst_lbl.write_text("\n".join(yolo_lines), encoding="utf-8")
        per_split_images[split] += 1
        per_split_counts[split] += len(yolo_lines)

    _write_data_yaml(out_dir, args.class_name)

    print("[prep_yolo] Done.")
    print(f"[prep_yolo] Forms: {len(form_ids)}")
    print(f"[prep_yolo] XML parsed: {len(annotations)}")
    print(f"[prep_yolo] Total line labels: {total_labels}")
    for split in SPLITS:
        print(
            f"  {split}: images={per_split_images[split]}, "
            f"labels={per_split_counts[split]}"
        )
    print(f"[prep_yolo] data.yaml: {out_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IAM XML line annotations to YOLO dataset format."
    )
    parser.add_argument("--xml_dir", default="data/iam/xml", help="Directory containing IAM XML files.")
    parser.add_argument("--images_dir", default="data/iam/forms", help="Directory containing form images.")
    parser.add_argument("--out_dir", default="data/yolo_dataset", help="Output YOLO dataset directory.")
    parser.add_argument(
        "--split_csv_dir",
        default="",
        help=(
            "Optional directory with train.csv/val.csv/test.csv from parse_iam.py. "
            "If provided, form-level split is derived from these files."
        ),
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio (random mode).")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio (random mode).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--class_id", type=int, default=0, help="YOLO class id for text lines.")
    parser.add_argument("--class_name", default="line", help="Class name written in data.yaml.")
    parser.add_argument(
        "--image_mode",
        choices=("link", "copy", "none"),
        default="link",
        help="How to place images in YOLO folder (hard-link, copy, or keep labels only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
