"""
test_shadow_preprocessing.py -- Fast checks for shadow correction helper.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.clean import remove_shadows


def _make_shadowed_text_image() -> np.ndarray:
    img = np.ones((220, 640), dtype=np.uint8) * 245
    cv2.putText(img, "My name is Aman", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25,), 2)
    cv2.putText(img, "I live in Sydney", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25,), 2)
    shadow = np.tile(np.linspace(90, 0, img.shape[1], dtype=np.uint8), (img.shape[0], 1))
    shadowed = cv2.subtract(img, shadow)
    return cv2.cvtColor(shadowed, cv2.COLOR_GRAY2BGR)


def test_remove_shadows_reduces_left_right_lighting_gap():
    image = _make_shadowed_text_image()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corrected = remove_shadows(image)

    h, w = gray.shape
    left_mean_before = float(gray[:, : w // 3].mean())
    right_mean_before = float(gray[:, 2 * w // 3 :].mean())
    left_mean_after = float(corrected[:, : w // 3].mean())
    right_mean_after = float(corrected[:, 2 * w // 3 :].mean())

    gap_before = abs(left_mean_before - right_mean_before)
    gap_after = abs(left_mean_after - right_mean_after)
    assert gap_after < gap_before, (gap_before, gap_after)


def test_remove_shadows_output_is_valid_grayscale_uint8():
    image = _make_shadowed_text_image()
    corrected = remove_shadows(image)
    assert corrected.ndim == 2
    assert corrected.dtype == np.uint8
    assert corrected.shape[:2] == image.shape[:2]
