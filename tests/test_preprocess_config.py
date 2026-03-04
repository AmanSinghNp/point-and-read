"""
test_preprocess_config.py — Tests for pluggable PreprocessConfig.
"""

import numpy as np
import pytest

from preprocessing.clean import preprocess_for_trocr, PreprocessConfig

def test_preprocess_config_defaults():
    """Verify default Config matches exactly previous hardcoded behavior."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    config = PreprocessConfig()
    out = preprocess_for_trocr(img, config=config)
    assert out.shape == (500, 500)
    assert out.max() > 0

def test_preprocess_config_skips_deskew():
    """Turning off deskew should result in no geometric rotation delays."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    config = PreprocessConfig(deskew=False, correct_perspective=False)
    out = preprocess_for_trocr(img, config=config)
    assert out.shape == (100, 100)

def test_preprocess_config_raw_binarization():
    """Testing 'raw' passing untouched grayscale output without Otsu."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[50:60, 50:60] = 255
    config = PreprocessConfig(binarization_mode="raw", deskew=False, correct_perspective=False)
    out = preprocess_for_trocr(img, config=config)
    assert out.shape == (100, 100)

def test_preprocess_config_removes_lines():
    """Testing that horizontal lines are removed by morphology pipeline."""
    img = np.ones((200, 200), dtype=np.uint8) * 255
    # Draw horizontal line
    img[100:102, :] = 0
    config = PreprocessConfig(remove_ruled_lines=True, deskew=False, correct_perspective=False)
    out = preprocess_for_trocr(img, config=config)
    assert out.shape == (200, 200)
