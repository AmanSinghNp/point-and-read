# Point and Read 📖🔍

A deep learning-based text recognition system using **CRNN + BiLSTM + CTC** architecture. Point your camera at any text and let the model read it for you.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

The recognition pipeline is built on a **CRNN** (Convolutional Recurrent Neural Network) with:

- **CNN Backbone** — ResNet-34 (pretrained on ImageNet), modified for single-channel grayscale input
- **Sequence Modelling** — 2-layer Bidirectional LSTM (hidden size 256)
- **Output Layer** — Fully connected projection with CTC (Connectionist Temporal Classification) decoding

```
Input Image (1×64×W) → ResNet-34 → AdaptivePool → BiLSTM → FC → CTC Decode → Text
```

## 📂 Project Structure

```
point-and-read/
├── recognition/          # Core recognition module
│   ├── model.py          #   CRNN model definition
│   ├── dataset.py        #   Dataset & data loading
│   ├── train.py          #   Training loop
│   ├── evaluate.py       #   Evaluation metrics (CER, WER)
│   ├── inference.py      #   Single-image inference
│   └── vocab.py          #   Vocabulary / character set
├── preprocessing/        # Image preprocessing & cleaning
│   └── clean.py          #   Binarization, deskew, noise removal
├── detection/            # Text detection (WIP)
│   ├── annotate/         #   Annotation tools
│   └── train/            #   Detection model training
├── scripts/              # Utility scripts
│   ├── verify_iam_structure.py
│   └── verify_preprocessing.py
├── tests/                # Smoke tests & test fixtures
├── parse_iam.py          # IAM dataset parser
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or AMD GPU with DirectML

### Installation

```bash
# Clone the repository
git clone https://github.com/AmanSinghNp/point-and-read.git
cd point-and-read

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

This project uses the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). The dataset is **not included** in this repository due to its size.

1. Download the IAM dataset and place it in `data/iam/`
2. Run the parser to prepare the data:

```bash
python parse_iam.py
```

3. Verify the dataset structure:

```bash
python scripts/verify_iam_structure.py
```

### Training

```bash
python -m recognition.train
```

### Inference

```bash
python -m recognition.inference --image path/to/image.png
```

### Evaluation

```bash
python -m recognition.evaluate
```

## 🧪 Testing

Run smoke tests to verify the pipeline:

```bash
python -m tests.smoke_test
```

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **CER** | Character Error Rate |
| **WER** | Word Error Rate |

## 🗺️ Roadmap

- [x] CRNN + BiLSTM + CTC recognition model
- [x] IAM dataset parsing & preprocessing
- [x] Training pipeline with CTC loss
- [x] Evaluation with CER/WER metrics
- [ ] Text detection module (EAST / CRAFT)
- [ ] End-to-end pipeline: detect → crop → recognise
- [ ] Real-time camera inference
- [ ] Web / mobile demo app

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [PyTorch](https://pytorch.org/)
- Inspired by the CRNN paper: *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (Shi et al., 2015)
