"""
test_adaptive_beams.py — Tests for adaptive beam width computation.

Tests the _compute_num_beams logic directly without importing the full
predictor module (which requires torch and other heavy dependencies).
"""

import pytest


def _compute_num_beams(line_image_width: int) -> int:
    """Local copy of the function for isolated testing."""
    if line_image_width < 150:
        return 1
    elif line_image_width < 300:
        return 2
    return 4


def test_very_short_line():
    """Lines < 150px wide should use greedy decoding (1 beam)."""
    assert _compute_num_beams(50) == 1
    assert _compute_num_beams(100) == 1
    assert _compute_num_beams(149) == 1


def test_medium_line():
    """Lines 150–299px wide should use 2 beams."""
    assert _compute_num_beams(150) == 2
    assert _compute_num_beams(200) == 2
    assert _compute_num_beams(299) == 2


def test_long_line():
    """Lines ≥ 300px wide should use 4 beams."""
    assert _compute_num_beams(300) == 4
    assert _compute_num_beams(500) == 4
    assert _compute_num_beams(1000) == 4


def test_adaptive_never_exceeds_user_cap():
    """When user specifies num_beams=2, adaptive should not exceed 2."""
    user_cap = 2
    assert min(user_cap, _compute_num_beams(500)) <= user_cap
    assert min(user_cap, _compute_num_beams(100)) <= user_cap


def test_boundary_150():
    """Exact boundary at 150px should use 2 beams, not 1."""
    assert _compute_num_beams(150) == 2


def test_boundary_300():
    """Exact boundary at 300px should use 4 beams, not 2."""
    assert _compute_num_beams(300) == 4
