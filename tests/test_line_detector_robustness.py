"""
test_line_detector_robustness.py -- Fast detector-only robustness checks.

These tests avoid model inference and validate that OpenCV line detection
still finds text bands under uneven lighting and page-edge artifacts.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detection.line_detector import (
    LineDetector,
    YOLOLineDetector,
    DetectionConfig,
    DetectionMode,
)


def _make_shadowed_multiline_image() -> np.ndarray:
    img = np.ones((260, 700), dtype=np.uint8) * 255
    lines = ["My name is Aman", "I live in Sydney", "Today is hot"]
    for i, text in enumerate(lines):
        y = 70 + i * 70
        cv2.putText(img, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)

    # Add an uneven shadow gradient on the left half.
    shadow = np.tile(np.linspace(45, 0, img.shape[1], dtype=np.uint8), (img.shape[0], 1))
    shadowed = cv2.subtract(img, shadow)

    # Add camera-framing artifacts on image edges.
    shadowed[:, :4] = 20
    shadowed[:, -4:] = 20
    shadowed[:3, :] = 25
    return cv2.cvtColor(shadowed, cv2.COLOR_GRAY2BGR)


def test_line_detector_handles_shadowed_multiline_image():
    detector = LineDetector()
    img = _make_shadowed_multiline_image()
    boxes = detector.detect(img)
    assert len(boxes) >= 2, f"Expected at least 2 lines, got {boxes}"


def test_line_detector_avoids_tiny_edge_noise_boxes():
    detector = LineDetector(DetectionConfig(mode=DetectionMode.LINE))
    img = _make_shadowed_multiline_image()
    h, w = img.shape[:2]
    boxes = detector.detect(img)
    assert boxes, "No boxes detected"
    for x, y, bw, bh in boxes:
        assert bw >= int(w * 0.12), f"Suspicious tiny-width box near edge: {(x, y, bw, bh)}"
        assert bh >= 10, f"Suspicious tiny-height box near edge: {(x, y, bw, bh)}"


def test_yolo_detector_gracefully_handles_missing_weights():
    img = _make_shadowed_multiline_image()
    detector = YOLOLineDetector(model_path="does_not_exist.pt")
    boxes = detector.detect(img)
    crops = detector.detect_and_crop(img)
    assert boxes == []
    assert crops == []
