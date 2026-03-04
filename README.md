# Point and Read

A handwriting recognition app using **TrOCR** (Transformer-based OCR). Point your webcam at handwritten text, snap a photo, and the app recognizes it and reads it aloud.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Webcam capture** — Snap handwritten notes with built-in blur and motion gates
- **TrOCR recognition** — Microsoft's Vision Transformer + decoder for handwriting
- **Multi-line detection** — OpenCV morphological line/word detection
- **TTS** — Read aloud with adjustable speed
- **Clipboard** — Copy recognized text with one click

## Installation

```bash
# Clone the repository
git clone https://github.com/AmanSinghNp/point-and-read.git
cd point-and-read

# Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The GUI opens with an input panel (drag-and-drop, file picker, or webcam capture) and an output panel (recognized text, TTS, copy).

## Model Options

| Model  | Size   | Speed   | Accuracy |
|--------|--------|---------|----------|
| Small  | ~240MB | Fastest | Good     |
| Base   | ~900MB | Medium  | Better   |
| Large  | ~1.7GB | Slowest | Best     |

Select from the dropdown in the header. **Small** is the default for webcam; use **Base** or **Large** for higher accuracy on difficult handwriting.

Models are downloaded from HuggingFace on first run and cached locally.

## Project Structure

```
point-and-read/
├── main.py              # Entry point
├── predictor.py         # TrOCR inference wrapper
├── trocr/               # Model config and paths
├── preprocessing/       # Border strip, deskew, CLAHE, binarization
│   └── clean.py
├── detection/           # Line/word detection with OpenCV
│   └── line_detector.py
├── gui/                 # PyQt6 interface
│   ├── main_window.py
│   ├── input_panel.py   # Image input, webcam, snap
│   └── output_panel.py  # Text, TTS, clipboard
├── tests/               # Predictor and pipeline tests
│   ├── test_predictor.py
│   ├── test_webcam_pipeline.py
│   └── fixtures/        # Test images
└── requirements.txt
```

## Testing

```bash
python tests/test_predictor.py
python tests/test_webcam_pipeline.py
```

Add handwritten photos to `tests/fixtures/webcam/` for regression tests. See `tests/fixtures/README.md` for the manifest format.

## Pipeline Overview

1. **Frame capture** — Blur gate and motion stability before snap
2. **Preprocessing** — Strip dark borders (e.g. DroidCam), perspective correction, CLAHE, deskew, binarization
3. **Detection** — Morphological line/word detection
4. **Crop prep** — Invert if needed, pad to square, resize to 384×384
5. **TrOCR** — Beam search decoding
6. **Retry** — If confidence &lt; 40%, retry with adaptive binarization or raw grayscale

## Requirements

- Python 3.10+
- CUDA (optional) for faster inference
- Webcam for live capture

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- [Microsoft TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)
- [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the GUI
