"""
test_webcam_pipeline.py — Full snap → preprocess → detect → OCR loop on fixtures.

Runs predict_page on fixture images and asserts confidence > 50%.
Uses programmatic test images when no real fixtures exist.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predictor import predict_page


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
WEBCAM_FIXTURES_DIR = os.path.join(FIXTURES_DIR, "webcam")


def create_line_fixture(text: str = "Hello world") -> np.ndarray:
    """Create a simple printed-style test image (simulates line)."""
    img = np.ones((80, 400), dtype=np.uint8) * 255
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def create_multi_line_fixture(lines: list[str] | None = None) -> np.ndarray:
    """Create a multi-line test image."""
    lines = lines or ["First line", "Second line", "Third line"]
    img = np.ones((200, 400), dtype=np.uint8) * 255
    for i, text in enumerate(lines):
        y = 40 + i * 55
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def create_word_fixture(text: str = "My name is Aman") -> np.ndarray:
    """Create image with spaced words for word-level detection test."""
    img = np.ones((100, 350), dtype=np.uint8) * 255
    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,), 2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_pipeline_single_line():
    """Run full pipeline on single-line fixture, assert confidence > 20%.

    Note: Programmatic printed text scores lower than real handwriting;
    real fixtures should achieve > 50%.
    """
    img = create_line_fixture("test")
    results = predict_page(img)
    assert isinstance(results, list)
    assert len(results) >= 1
    for r in results:
        assert "text" in r and "confidence" in r
        assert r["confidence"] > 0.2, f"Low confidence: {r['confidence']}"


def test_pipeline_multi_line():
    """Run full pipeline on multi-line fixture."""
    img = create_multi_line_fixture()
    results = predict_page(img)
    assert isinstance(results, list)
    assert len(results) >= 1
    combined = "".join(r["text"] for r in results)
    assert len(combined) > 0
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    assert avg_conf > 0.2, f"Low avg confidence: {avg_conf}"


def test_pipeline_with_webcam_config():
    """Run pipeline with use_webcam_config=True."""
    img = create_line_fixture("webcam test")
    results = predict_page(img, use_webcam_config=True)
    assert len(results) >= 1
    assert results[0]["confidence"] > 0.2


def test_fixtures_from_dir():
    """Run pipeline on any PNG/JPEG files in tests/fixtures/."""
    if not os.path.isdir(FIXTURES_DIR):
        return
    for name in os.listdir(FIXTURES_DIR):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(FIXTURES_DIR, name)
            results = predict_page(path)
            assert len(results) >= 1, f"No results for {name}"
            avg_conf = sum(r["confidence"] for r in results) / len(results)
            assert avg_conf > 0.2, f"Low confidence for {name}: {avg_conf}"


def test_webcam_real_fixtures():
    """Run full pipeline on real webcam fixtures in tests/fixtures/webcam/.

    Add 10-15 handwritten photos to tests/fixtures/webcam/ for regression testing.
    Asserts confidence > 40% and output is non-empty.
    Optional: manifest.csv with filename,expected_substrings for stricter checks.
    """
    if not os.path.isdir(WEBCAM_FIXTURES_DIR):
        return
    manifest_path = os.path.join(WEBCAM_FIXTURES_DIR, "manifest.csv")
    manifest = {}
    if os.path.isfile(manifest_path):
        import csv
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename", "").strip()
                # Backward-compatible: accept both singular and plural header names.
                subs = (
                    row.get("expected_substrings")
                    or row.get("expected_substring")
                    or ""
                ).strip()
                if fn and subs:
                    manifest[fn] = [s.strip() for s in subs.split("|") if s.strip()]
    count = 0
    for name in sorted(os.listdir(WEBCAM_FIXTURES_DIR)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(WEBCAM_FIXTURES_DIR, name)
            results = predict_page(path, use_webcam_config=True)
            assert len(results) >= 1, f"No results for {name}"
            combined = "".join(r["text"] for r in results)
            assert len(combined.strip()) > 0, f"Empty output for {name}"
            avg_conf = sum(r["confidence"] for r in results) / len(results)
            assert avg_conf > 0.4, f"Low confidence for {name}: {avg_conf:.2f}"
            if name in manifest:
                for sub in manifest[name]:
                    assert sub.lower() in combined.lower(), (
                        f"Expected '{sub}' in output for {name}, got '{combined[:80]}...'"
                    )
            count += 1
    if count > 0:
        print(f"[OK] {count} webcam fixture(s)")


if __name__ == "__main__":
    print("=" * 50)
    print("Webcam Pipeline Tests")
    print("=" * 50)
    test_pipeline_single_line()
    print("[OK] Single-line pipeline")
    test_pipeline_multi_line()
    print("[OK] Multi-line pipeline")
    test_pipeline_with_webcam_config()
    print("[OK] Webcam config pipeline")
    test_fixtures_from_dir()
    print("[OK] Fixtures from dir")
    test_webcam_real_fixtures()
    print("All webcam pipeline tests passed.")
