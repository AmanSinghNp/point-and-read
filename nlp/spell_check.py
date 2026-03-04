"""
SymSpell-based OCR post-processing.

Corrects likely OCR spelling artifacts (e.g. split/merged words) using an
English frequency dictionary.
"""

from __future__ import annotations

from importlib import resources


class OCRCorrector:
    """Lightweight spell corrector for OCR output."""

    def __init__(self, *, max_edit_distance: int = 2, prefix_length: int = 7):
        self.sym_spell = None
        self.enabled = False
        self.error: str | None = None

        try:
            from symspellpy import SymSpell
        except Exception as exc:
            self.error = f"symspellpy not available: {exc}"
            return

        try:
            self.sym_spell = SymSpell(
                max_dictionary_edit_distance=max_edit_distance,
                prefix_length=prefix_length,
            )

            dictionary_path = resources.files("symspellpy").joinpath(
                "frequency_dictionary_en_82_765.txt"
            )
            loaded = self.sym_spell.load_dictionary(
                str(dictionary_path), term_index=0, count_index=1
            )
            if not loaded:
                self.error = "Failed to load default SymSpell dictionary."
                return
            self.enabled = True
        except Exception as exc:
            self.error = f"Failed to initialize OCRCorrector: {exc}"
            self.sym_spell = None
            self.enabled = False

    def is_ready(self) -> bool:
        return self.enabled and self.sym_spell is not None

    def correct_text(self, text: str, *, max_edit_distance: int = 2) -> str:
        """Return spell-corrected text if available, else input text."""
        if not text or not text.strip():
            return text
        if not self.is_ready():
            return text

        raw = text
        stripped = raw.strip()
        leading = raw[: len(raw) - len(raw.lstrip())]
        trailing = raw[len(raw.rstrip()) :]

        suggestions = self.sym_spell.lookup_compound(
            stripped, max_edit_distance=max_edit_distance
        )
        if not suggestions:
            return raw
        corrected = suggestions[0].term
        return f"{leading}{corrected}{trailing}"
