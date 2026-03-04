"""
test_orientation_preprocessing.py -- Fast tests for axis-orientation alignment.

These tests do not load TrOCR. They validate Phase 1 orientation handling:
detecting vertical text blocks and rotating by 90 degrees.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.clean import align_text_axis_minarearect


def _make_horizontal_text_image(text: str = "My name is Aman") -> np.ndarray:
    img = np.ones((180, 520), dtype=np.uint8) * 255
    cv2.putText(
        img,
        text,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0,),
        2,
    )
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_axis_alignment_rotates_vertical_text_to_horizontal():
    base = _make_horizontal_text_image()
    vertical = cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE)

    aligned, meta = align_text_axis_minarearect(vertical)

    assert meta["axis_rotated_90"] is True
    assert meta["axis_rotation_degrees"] == 90
    # After alignment, image should be landscape-like again.
    assert aligned.shape[1] >= aligned.shape[0]


def test_axis_alignment_keeps_horizontal_text_unchanged():
    base = _make_horizontal_text_image()
    aligned, meta = align_text_axis_minarearect(base)

    assert meta["axis_rotation_degrees"] == 0
    assert meta["axis_rotated_90"] is False
    assert aligned.shape == base.shape
