"""
SymSpell-based OCR post-processing.

Corrects likely OCR spelling artifacts (e.g. split/merged words) using an
English frequency dictionary.  Supports a protected-words file so domain-
specific terms, acronyms, and proper nouns are never mangled.
"""

from __future__ import annotations

import os
from importlib import resources


class OCRCorrector:
    """Lightweight spell corrector for OCR output."""

    def __init__(
        self,
        *,
        max_edit_distance: int = 2,
        prefix_length: int = 7,
        protected_words_path: str = "data/protected_words.txt",
    ):
        self.sym_spell = None
        self.enabled = False
        self.error: str | None = None
        self._protected_words: set[str] = set()

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

            # Load protected words so they are never "corrected".
            self._load_protected_words(protected_words_path)

            self.enabled = True
        except Exception as exc:
            self.error = f"Failed to initialize OCRCorrector: {exc}"
            self.sym_spell = None
            self.enabled = False

    def _load_protected_words(self, path: str) -> None:
        """Load words from file and register them in the SymSpell dictionary."""
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    # Skip blank lines and comments.
                    if not word or word.startswith("#"):
                        continue
                    self._protected_words.add(word)
                    # Add to SymSpell dictionary so it recognises the word.
                    if self.sym_spell is not None:
                        self.sym_spell.create_dictionary_entry(word.lower(), 1)
        except Exception:
            pass  # Best-effort; don't fail OCR because of protected words.

    def is_ready(self) -> bool:
        return self.enabled and self.sym_spell is not None

    def correct_text(self, text: str, *, max_edit_distance: int = 2) -> str:
        """Return spell-corrected text if available, else input text.

        Preserves:
            - ALL-CAPS tokens (likely acronyms, e.g. NASA, OCR)
            - Tokens in the protected-words set
            - Leading/trailing whitespace
        """
        if not text or not text.strip():
            return text
        if not self.is_ready():
            return text

        raw = text
        stripped = raw.strip()
        leading = raw[: len(raw) - len(raw.lstrip())]
        trailing = raw[len(raw.rstrip()) :]

        # Token-wise correction with acronym / protected-word preservation.
        tokens = stripped.split()
        corrected_tokens: list[str] = []
        for token in tokens:
            # Preserve ALL-CAPS tokens (acronyms like NASA, GPU, etc.)
            if token.isupper() and len(token) > 1:
                corrected_tokens.append(token)
                continue
            # Preserve explicitly protected words (case-sensitive match).
            if token in self._protected_words:
                corrected_tokens.append(token)
                continue
            corrected_tokens.append(token)

        # Run SymSpell compound correction on the non-protected text.
        corrected_line = " ".join(corrected_tokens)
        suggestions = self.sym_spell.lookup_compound(
            corrected_line, max_edit_distance=max_edit_distance
        )
        if not suggestions:
            return raw

        result = suggestions[0].term

        # Re-insert protected / ALL-CAPS tokens that SymSpell may have mangled.
        result_tokens = result.split()
        for i, orig_token in enumerate(tokens):
            is_acronym = orig_token.isupper() and len(orig_token) > 1
            is_protected = orig_token in self._protected_words
            if (is_acronym or is_protected) and i < len(result_tokens):
                result_tokens[i] = orig_token

        corrected = " ".join(result_tokens)
        return f"{leading}{corrected}{trailing}"
