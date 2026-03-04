"""
test_ocr_result_dto.py — Tests for LineResult and PageResult dataclasses.
"""

import numpy as np
import pytest

from models.ocr_result import LineResult, PageResult


class TestLineResult:
    def test_construction(self):
        lr = LineResult(
            text="hello world",
            raw_text="hello worlld",
            confidence=0.85,
            bbox=(10, 20, 300, 40),
            spell_corrected=True,
        )
        assert lr.text == "hello world"
        assert lr.raw_text == "hello worlld"
        assert lr.confidence == 0.85
        assert lr.bbox == (10, 20, 300, 40)
        assert lr.spell_corrected is True
        assert lr.line_index is None
        assert lr.crop is None

    def test_to_dict(self):
        lr = LineResult(
            text="abc", raw_text="abc", confidence=0.9,
            bbox=(0, 0, 100, 50), spell_corrected=False,
        )
        d = lr.to_dict()
        assert d == {
            "text": "abc",
            "raw_text": "abc",
            "confidence": 0.9,
            "bbox": (0, 0, 100, 50),
            "spell_corrected": False,
        }
        # crop should NOT be in dict when None
        assert "crop" not in d

    def test_to_dict_with_crop(self):
        crop = np.ones((50, 100), dtype=np.uint8)
        lr = LineResult(
            text="x", raw_text="x", confidence=0.5,
            bbox=(0, 0, 100, 50), crop=crop,
        )
        d = lr.to_dict()
        assert "crop" in d
        assert d["crop"] is crop

    def test_to_dict_excludes_empty_crop(self):
        lr = LineResult(
            text="x", raw_text="x", confidence=0.5,
            bbox=(0, 0, 100, 50), crop=np.array([]),
        )
        d = lr.to_dict()
        assert "crop" not in d


class TestPageResult:
    def test_full_text_joins_lines(self):
        pr = PageResult(lines=[
            LineResult(text="Line 1", raw_text="Line 1", confidence=0.9, bbox=None),
            LineResult(text="Line 2", raw_text="Line 2", confidence=0.8, bbox=None),
            LineResult(text="Line 3", raw_text="Line 3", confidence=0.7, bbox=None),
        ])
        assert pr.full_text == "Line 1\nLine 2\nLine 3"

    def test_mean_confidence(self):
        pr = PageResult(lines=[
            LineResult(text="a", raw_text="a", confidence=0.9, bbox=None),
            LineResult(text="b", raw_text="b", confidence=0.6, bbox=None),
        ])
        assert abs(pr.mean_confidence - 0.75) < 1e-9

    def test_empty_page_result(self):
        pr = PageResult()
        assert pr.full_text == ""
        assert pr.mean_confidence == 0.0
        assert pr.lines == []

    def test_to_dict_list(self):
        pr = PageResult(lines=[
            LineResult(text="x", raw_text="x", confidence=0.5, bbox=(1, 2, 3, 4)),
        ])
        dicts = pr.to_dict_list()
        assert len(dicts) == 1
        assert dicts[0]["text"] == "x"
        assert dicts[0]["bbox"] == (1, 2, 3, 4)
