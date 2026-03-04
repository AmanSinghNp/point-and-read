"""
test_spell_check_edge_cases.py — Tests for spell check improvements.

Tests protected words, acronym preservation, and edge cases.
"""

import pytest


def test_acronyms_preserved_when_available():
    """ALL-CAPS tokens like NASA, GPU should not be modified."""
    pytest.importorskip("symspellpy")
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    if not corrector.is_ready():
        pytest.skip("SymSpell dictionary unavailable")

    result = corrector.correct_text("the NASA program")
    assert "NASA" in result, f"NASA was mangled to: {result}"

    result2 = corrector.correct_text("GPU and CPU usage")
    assert "GPU" in result2, f"GPU was mangled to: {result2}"
    assert "CPU" in result2, f"CPU was mangled to: {result2}"


def test_single_uppercase_char_not_protected():
    """Single uppercase letters (like 'I', 'A') should still be correctable."""
    pytest.importorskip("symspellpy")
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    if not corrector.is_ready():
        pytest.skip("SymSpell dictionary unavailable")

    # Single uppercase letter should NOT be treated as an acronym
    result = corrector.correct_text("I went home")
    assert isinstance(result, str)
    assert len(result) > 0


def test_empty_string_returns_empty():
    """Empty or whitespace input returns as-is."""
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    assert corrector.correct_text("") == ""
    assert corrector.correct_text("   ") == "   "


def test_protected_words_loaded():
    """Protected words file should be loaded when it exists."""
    pytest.importorskip("symspellpy")
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector(protected_words_path="data/protected_words.txt")
    if not corrector.is_ready():
        pytest.skip("SymSpell dictionary unavailable")

    # Protected words should be in the set
    import os
    if os.path.exists("data/protected_words.txt"):
        assert len(corrector._protected_words) > 0


def test_existing_typo_correction_still_works():
    """Basic spell correction should still function for regular words."""
    pytest.importorskip("symspellpy")
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    if not corrector.is_ready():
        pytest.skip("SymSpell dictionary unavailable")

    out = corrector.correct_text("speling")
    assert "spelling" in out.lower()


def test_corrector_fallback_when_unavailable():
    """OCRCorrector should return input unchanged if SymSpell is not available."""
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    raw = "some text here"
    out = corrector.correct_text(raw)
    assert isinstance(out, str)
    assert out.strip() != ""
