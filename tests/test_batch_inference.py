"""
test_batch_inference.py — Tests for batch line inference in predictor.py.
"""

import numpy as np
import pytest

from predictor import batch_predict_with_confidence

def test_empty_batch():
    """Batching with an empty list should safely return an empty list."""
    results = batch_predict_with_confidence([])
    assert results == []

@pytest.mark.skip(reason="Requires full TrOCR model and dependencies")
def test_batch_matches_single():
    """
    Verify that batch predicting multiple images yields the same output texts
    and similar confidence scores as predicting them one-by-one.
    """
    # Create two dummy crops
    img1 = np.ones((50, 150, 3), dtype=np.uint8) * 200
    img2 = np.ones((40, 200, 3), dtype=np.uint8) * 150

    # 1. Batch prediction
    batch_results = batch_predict_with_confidence([img1, img2])
    assert len(batch_results) == 2
    
    # 2. Single predictions
    single1 = batch_predict_with_confidence([img1])[0]
    single2 = batch_predict_with_confidence([img2])[0]
    
    # Texts should match exactly
    assert batch_results[0][0] == single1[0]
    assert batch_results[1][0] == single2[0]
    
    # Confidences might differ slightly due to padding, but should be close
    assert abs(batch_results[0][1] - single1[1]) < 0.05
    assert abs(batch_results[1][1] - single2[1]) < 0.05
