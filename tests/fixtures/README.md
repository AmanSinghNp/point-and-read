# Test fixtures for webcam pipeline and word detection

## Purpose

This directory holds handwritten image fixtures for testing the full
snap → preprocess → detect → OCR pipeline against real handwriting.

## Structure

- **fixtures/**: General test images (any PNG/JPEG).
- **fixtures/webcam/**: Real webcam-captured handwritten notes for regression testing.

## Adding webcam fixtures

1. Photograph 10–15 handwritten notes with your webcam.
2. Save as PNG or JPEG in `fixtures/webcam/`.
3. Run `python tests/test_webcam_pipeline.py` — asserts confidence > 50% and non-empty output.

## Manifest (optional)

In `fixtures/webcam/`, create `manifest.csv`:

```csv
filename,expected_substrings
note1.png,My name is Aman
note2.jpg,hello|world
```

Use `|` to separate multiple expected substrings. The test checks that each appears in the output.

## Placeholder fixtures

The test suite uses programmatically generated images when no real fixtures exist.
Add real handwritten photos to `webcam/` for safety-net regression tests.
