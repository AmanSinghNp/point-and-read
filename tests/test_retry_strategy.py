"""
test_retry_strategy.py — Tests for confidence-based inverted color retries.
"""

import numpy as np
import pytest

from predictor import _run_with_preprocessed
from preprocessing.clean import PreprocessConfig

@pytest.mark.skip(reason="Requires full TrOCR model and dependencies")
def test_low_confidence_triggers_retry(mocker):
    """
    Mock the batched inference loop to return low confidence on the first pass
    and high confidence on the inverted pass. Ensure the high confidence is kept.
    """
    # Create a 200x50 crop simulate
    img = np.ones((100, 300), dtype=np.uint8) * 128
    
    # Mock batch_predict_with_confidence
    # Pass 1: returns low confidence (0.3)
    # Pass 2: (the retry) returns high confidence (0.9)
    mock_batch = mocker.patch("predictor.batch_predict_with_confidence")
    mock_batch.side_effect = [
        [("Low confidence text", 0.3)],
        [("High confidence text", 0.9)]
    ]
    
    # We also need to mock _detect_main_crops to return a single dummy crop
    mocker.patch("predictor._detect_main_crops", return_value=([(img, (0, 0, 300, 100))], "opencv"))
    
    # Run the loop
    results, _, _ = _run_with_preprocessed(img)
    
    assert len(results) == 1
    # Ensure the accepted text and confidence are from the second mock pass
    assert results[0].text == "High confidence text"
    assert results[0].confidence == 0.9
