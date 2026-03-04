"""
ocr_result.py — Typed data transfer objects for OCR pipeline results.

Replaces dict-based returns with structured dataclasses for IDE support,
type safety, and convenient computed properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class LineResult:
    """Result for a single recognised text line or word region."""

    text: str
    raw_text: str
    confidence: float
    bbox: tuple[int, int, int, int] | None
    spell_corrected: bool = False
    line_index: Optional[int] = None
    crop: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict (backward-compatible with existing API)."""
        d: dict[str, Any] = {
            "text": self.text,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "spell_corrected": self.spell_corrected,
        }
        if self.crop is not None and isinstance(self.crop, np.ndarray) and self.crop.size > 0:
            d["crop"] = self.crop
        return d


@dataclass
class PageResult:
    """Aggregated result for a full page of recognised text lines."""

    lines: list[LineResult] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Join all line texts with newlines."""
        return "\n".join(line.text for line in self.lines)

    @property
    def mean_confidence(self) -> float:
        """Average confidence across all lines (0.0 if empty)."""
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to list of dicts (backward-compatible with existing API)."""
        return [line.to_dict() for line in self.lines]
