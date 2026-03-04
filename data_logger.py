"""
data_logger.py -- Save user OCR corrections into a local personal dataset.

Dataset layout:
    data/personal_dataset/
        images/
            <uuid>.png
        labels.csv
"""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np


def _ensure_dirs(dataset_root: str) -> tuple[str, str]:
    images_dir = os.path.join(dataset_root, "images")
    labels_csv = os.path.join(dataset_root, "labels.csv")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir, labels_csv


def _iter_line_pairs(results: list[dict], edited_text: str):
    edited_lines = edited_text.replace("\r\n", "\n").split("\n")
    count = min(len(results), len(edited_lines))
    for idx in range(count):
        result = results[idx]
        corrected = edited_lines[idx].strip()
        raw = str(result.get("raw_text") or result.get("text") or "").strip()
        yield idx, result, raw, corrected


def save_corrections(
    results: list[dict],
    edited_text: str,
    dataset_root: str = "data/personal_dataset",
) -> dict:
    """Save changed OCR lines into a local dataset folder.

    Args:
        results: OCR line results returned from predictor. Expected to include
            "raw_text" and optionally "crop" (np.ndarray).
        edited_text: Full edited text block from UI (line-separated).
        dataset_root: Root folder for saved image+label pairs.

    Returns:
        Summary dict with counts:
            compared, changed, saved, skipped, labels_csv
    """
    images_dir, labels_csv = _ensure_dirs(dataset_root)
    compared = 0
    changed = 0
    saved = 0
    skipped = 0
    rows: list[list[str]] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for idx, result, raw_text, corrected_text in _iter_line_pairs(results, edited_text):
        compared += 1
        if not corrected_text or corrected_text == raw_text:
            continue
        changed += 1

        crop = result.get("crop")
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            skipped += 1
            continue

        uid = uuid.uuid4().hex
        image_name = f"{uid}.png"
        image_path = os.path.join(images_dir, image_name)
        image = crop
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if not cv2.imwrite(image_path, image):
            skipped += 1
            continue

        rows.append(
            [
                timestamp,
                image_name,
                str(idx),
                raw_text,
                corrected_text,
                str(result.get("confidence", "")),
                str(result.get("bbox", "")),
            ]
        )
        saved += 1

    if rows:
        write_header = not os.path.exists(labels_csv)
        with open(labels_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "timestamp_utc",
                        "image_file",
                        "line_index",
                        "raw_text",
                        "corrected_text",
                        "confidence",
                        "bbox",
                    ]
                )
            writer.writerows(rows)

    return {
        "compared": compared,
        "changed": changed,
        "saved": saved,
        "skipped": skipped,
        "labels_csv": labels_csv,
    }

