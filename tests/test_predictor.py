"""
test_predictor.py -- Tests for the TrOCR predictor.

Creates dummy handwriting images with OpenCV, runs predict() and
predict_with_confidence(), and verifies the outputs. Run this before
launching the GUI to confirm the model loads and infers correctly.

Usage:
    python tests/test_predictor.py
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predictor import predict, predict_with_confidence, predict_page, get_available_models


def create_test_image(text: str = "hello world") -> np.ndarray:
    """Create a simple image with printed text for testing."""
    img = np.ones((100, 500), dtype=np.uint8) * 255
    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)
    return img


def create_multi_line_test_image(lines: list[str] = None) -> np.ndarray:
    """Create an image with multiple lines of text for predict_page testing."""
    lines = lines or ["First line", "Second line", "Third line"]
    img = np.ones((300, 500), dtype=np.uint8) * 255
    y_step = 100
    for i, text in enumerate(lines):
        y = 50 + i * y_step
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)
    return img


def test_predict_from_numpy():
    """Test predict() with a numpy array (BGR, like cv2.imread returns)."""
    print("[test] predict(numpy array) ... ", end="", flush=True)
    img_gray = create_test_image("hello world")
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    result = predict(img_bgr)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Expected non-empty output"
    print(f"OK -> '{result}'")


def test_predict_from_pil():
    """Test predict() with a PIL Image."""
    print("[test] predict(PIL Image) ... ", end="", flush=True)
    img_gray = create_test_image("test image")
    pil_img = Image.fromarray(img_gray).convert("RGB")
    result = predict(pil_img)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Expected non-empty output"
    print(f"OK -> '{result}'")


def test_predict_from_file():
    """Test predict() with a file path."""
    print("[test] predict(file path) ... ", end="", flush=True)
    img = create_test_image("file test")
    tmp_path = os.path.join(os.path.dirname(__file__), "_test_temp.png")
    cv2.imwrite(tmp_path, img)
    try:
        result = predict(tmp_path)
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Expected non-empty output"
        print(f"OK -> '{result}'")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_predict_from_grayscale():
    """Test predict() with a grayscale numpy array."""
    print("[test] predict(grayscale array) ... ", end="", flush=True)
    img_gray = create_test_image("grayscale")
    result = predict(img_gray)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Expected non-empty output"
    print(f"OK -> '{result}'")


def test_predict_with_confidence():
    """Test predict_with_confidence() returns (str, float)."""
    print("[test] predict_with_confidence() ... ", end="", flush=True)
    img_gray = create_test_image("confidence")
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    text, confidence = predict_with_confidence(img_bgr)
    assert isinstance(text, str), f"Expected str, got {type(text)}"
    assert isinstance(confidence, float), f"Expected float, got {type(confidence)}"
    assert len(text) > 0, "Expected non-empty output"
    assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of [0, 1] range"
    print(f"OK -> '{text}' (confidence: {confidence:.3f})")


def test_available_models():
    """Test get_available_models() returns a list."""
    print("[test] get_available_models() ... ", end="", flush=True)
    models = get_available_models()
    assert isinstance(models, list), f"Expected list, got {type(models)}"
    assert len(models) >= 3, f"Expected at least 3 models, got {len(models)}"
    assert "small" in models, "'small' not in available models"
    assert "base" in models, "'base' not in available models"
    assert "large" in models, "'large' not in available models"
    print(f"OK -> {models}")


def test_predict_page():
    """Test predict_page() returns list of dicts with text, confidence, bbox."""
    print("[test] predict_page() ... ", end="", flush=True)
    img = create_multi_line_test_image()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    results = predict_page(img_bgr)
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) >= 1, "Expected at least one result"
    for r in results:
        assert "text" in r, f"Missing 'text' in result: {r}"
        assert "confidence" in r, f"Missing 'confidence' in result: {r}"
        assert "bbox" in r, f"Missing 'bbox' in result: {r}"
        assert isinstance(r["text"], str), f"text should be str, got {type(r['text'])}"
        assert isinstance(r["confidence"], float), f"confidence should be float, got {type(r['confidence'])}"
        assert r["bbox"] is None or isinstance(r["bbox"], tuple), f"bbox should be tuple or None, got {type(r['bbox'])}"
    print(f"OK -> {len(results)} line(s)")


def test_predict_auto_detect_lines():
    """Test predict() with auto_detect_lines=True returns joined multi-line string."""
    print("[test] predict(auto_detect_lines=True) ... ", end="", flush=True)
    img = create_multi_line_test_image()
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = predict(img_bgr, auto_detect_lines=True)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Expected non-empty output"
    # With multiple lines, result should contain newlines (or at least text)
    print(f"OK -> '{result[:50]}...' (len={len(result)})")


if __name__ == "__main__":
    print("=" * 50)
    print("TrOCR Predictor Test")
    print("=" * 50)
    print(f"Default model: microsoft/trocr-base-handwritten")
    print()

    print("[test] Loading model (first call downloads if needed)...")
    test_available_models()
    test_predict_from_numpy()
    test_predict_from_pil()
    test_predict_from_file()
    test_predict_from_grayscale()
    test_predict_with_confidence()
    test_predict_page()
    test_predict_auto_detect_lines()

    print()
    print("All tests passed.")
