import csv
import os

import numpy as np

from data_logger import save_corrections


def test_save_corrections_saves_changed_lines(tmp_path):
    dataset_root = str(tmp_path / "personal_dataset")
    crop = np.ones((32, 64), dtype=np.uint8) * 255
    results = [
        {"raw_text": "hello world", "text": "hello world", "confidence": 0.8, "bbox": (1, 2, 3, 4), "crop": crop},
        {"raw_text": "second line", "text": "second line", "confidence": 0.7, "bbox": (2, 3, 4, 5), "crop": crop},
    ]
    edited = "hello word\nsecond line"

    summary = save_corrections(results, edited, dataset_root=dataset_root)
    assert summary["compared"] == 2
    assert summary["changed"] == 1
    assert summary["saved"] == 1
    assert summary["skipped"] == 0

    labels_csv = os.path.join(dataset_root, "labels.csv")
    assert os.path.exists(labels_csv)
    images_dir = os.path.join(dataset_root, "images")
    assert len(os.listdir(images_dir)) == 1

    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["raw_text"] == "hello world"
    assert rows[0]["corrected_text"] == "hello word"


def test_save_corrections_skips_when_no_crop(tmp_path):
    dataset_root = str(tmp_path / "personal_dataset")
    results = [{"raw_text": "abc", "text": "abc", "confidence": 0.6, "bbox": (0, 0, 1, 1)}]
    summary = save_corrections(results, "abd", dataset_root=dataset_root)
    assert summary["changed"] == 1
    assert summary["saved"] == 0
    assert summary["skipped"] == 1
