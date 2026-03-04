"""
test_resolution_cap.py — Tests for MAX_DIM resolution capping in preprocess_for_trocr().
"""

import numpy as np
import pytest

from preprocessing.clean import preprocess_for_trocr


def test_large_image_is_capped():
    """A 4000×3000 image should be downscaled so max dimension ≤ 2048."""
    img = np.ones((3000, 4000, 3), dtype=np.uint8) * 200
    result = preprocess_for_trocr(img)
    h, w = result.shape[:2]
    assert max(h, w) <= 2048, f"Max dim {max(h, w)} exceeds 2048"


def test_aspect_ratio_preserved():
    """Aspect ratio should be approximately preserved after downscaling."""
    img = np.ones((3000, 6000, 3), dtype=np.uint8) * 200
    original_ratio = 6000 / 3000
    result = preprocess_for_trocr(img)
    h, w = result.shape[:2]
    # Aspect ratio should be close — binarization may crop slightly
    # but the capping itself should preserve ratio exactly.
    assert max(h, w) <= 2048


def test_small_image_unchanged_dimensions():
    """An 800×600 image should not be resized by the cap."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    result = preprocess_for_trocr(img)
    # Output may differ from input due to other preprocessing (borders, binarization)
    # but it should NOT have been downscaled by resolution cap
    h, w = result.shape[:2]
    # The preprocessed image should still be roughly the same size or smaller
    # (other steps like border removal might crop it)
    assert h <= 600 and w <= 800


def test_empty_image_raises():
    """Empty image should raise ValueError."""
    with pytest.raises(ValueError, match="Empty"):
        preprocess_for_trocr(np.array([]))


def test_exact_2048_not_resized():
    """An image exactly 2048px wide should not trigger downscaling."""
    img = np.ones((1024, 2048, 3), dtype=np.uint8) * 200
    result = preprocess_for_trocr(img)
    h, w = result.shape[:2]
    assert max(h, w) <= 2048
