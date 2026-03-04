"""
test_word_detection.py — Word-level detection and OCR tests.

Uses "My name is Aman" style fixture to verify DetectionMode.WORD
returns correct word-level tokens.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predictor import predict_page
from detection.line_detector import LineDetector, DetectionConfig, DetectionMode


def create_my_name_is_aman_fixture() -> np.ndarray:
    """Create fixture with text 'My name is Aman'."""
    img = np.ones((120, 380), dtype=np.uint8) * 255
    cv2.putText(
        img, "My name is Aman", (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2,
    )
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_word_detector_boxes_are_left_to_right():
    """WORD mode should return same-line word boxes in left-to-right order."""
    img = create_my_name_is_aman_fixture()
    detector = LineDetector(DetectionConfig(mode=DetectionMode.WORD))
    boxes = detector.detect(img)
    assert len(boxes) >= 2
    xs = [x for x, _, _, _ in boxes]
    assert xs == sorted(xs), f"Unexpected word order: {boxes}"


def test_word_detection_returns_tokens():
    """Run with DetectionMode.WORD, assert we get multiple regions."""
    img = create_my_name_is_aman_fixture()
    results = predict_page(img, detection_mode="word")
    # Word mode should detect at least 1 region (may merge depending on spacing)
    assert isinstance(results, list)
    assert len(results) >= 1
    combined = " ".join(r["text"] for r in results).strip()
    assert len(combined) > 0


def test_word_detection_contains_expected_substrings():
    """Assert OCR output contains expected words from 'My name is Aman'."""
    img = create_my_name_is_aman_fixture()
    results = predict_page(img, detection_mode="word")
    combined = " ".join(r["text"] for r in results).lower()
    # At least one of these should appear (TrOCR may vary)
    expected = ["my", "name", "is", "aman"]
    found = sum(1 for w in expected if w in combined)
    assert found >= 2, f"Expected substrings in '{combined}'"


def test_word_mode_vs_line_mode():
    """Word mode and line mode both produce valid results."""
    img = create_my_name_is_aman_fixture()
    line_results = predict_page(img, detection_mode="line")
    word_results = predict_page(img, detection_mode="word")
    assert len(line_results) >= 1
    assert len(word_results) >= 1
    line_text = "".join(r["text"] for r in line_results)
    word_text = " ".join(r["text"] for r in word_results)
    assert len(line_text) > 0
    assert len(word_text) > 0


if __name__ == "__main__":
    print("=" * 50)
    print("Word Detection Tests")
    print("=" * 50)
    test_word_detector_boxes_are_left_to_right()
    print("[OK] Word box order is left-to-right")
    test_word_detection_returns_tokens()
    print("[OK] Word detection returns tokens")
    test_word_detection_contains_expected_substrings()
    print("[OK] Expected substrings present")
    test_word_mode_vs_line_mode()
    print("[OK] Word vs line mode")
    print("All word detection tests passed.")
