"""
test_spell_check.py -- Optional SymSpell integration checks.
"""

import pytest


def test_ocr_corrector_has_safe_fallback_when_unavailable():
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    # Should never raise; if not ready it must return input unchanged.
    raw = "Poo cess data"
    out = corrector.correct_text(raw)
    assert isinstance(out, str)
    assert out.strip() != ""


def test_ocr_corrector_can_fix_simple_typo_when_available():
    pytest.importorskip("symspellpy")
    from nlp.spell_check import OCRCorrector

    corrector = OCRCorrector()
    if not corrector.is_ready():
        pytest.skip("SymSpell dictionary unavailable in this environment")

    out = corrector.correct_text("speling")
    assert "spelling" in out.lower()
