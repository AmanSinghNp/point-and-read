"""
Quick smoke test: creates dummy data, runs 3 epochs of training,
then runs inference on one image to verify the full pipeline.
"""
import os
import sys
import csv
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recognition.vocab import Vocabulary
from recognition.train import train
from recognition.inference import load_model, recognise_line
from preprocessing.clean import Preprocessor


def create_dummy_data(out_dir, n=20):
    """Create n dummy images with simple OpenCV text."""
    os.makedirs(out_dir, exist_ok=True)
    phrases = [
        "hello world", "good morning", "test image", "deep learning",
        "neural network", "open source", "machine learning", "computer vision",
        "text recognition", "handwriting", "pytorch model", "training data",
        "sample text", "another line", "random words", "final test",
        "batch size", "gradient clip", "learning rate", "early stop",
    ]
    rows = []
    for i in range(n):
        img = np.ones((100, 500), dtype=np.uint8) * 255
        text = phrases[i % len(phrases)]
        cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,), 2)
        path = os.path.join(out_dir, f"dummy_{i:03d}.jpg")
        cv2.imwrite(path, img)
        rows.append((os.path.abspath(path), text))
    return rows


def main():
    base = os.path.join(os.path.dirname(__file__), "..")
    tmp_dir = os.path.join(base, "tests", "smoke")
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Create dummy data
    rows = create_dummy_data(os.path.join(tmp_dir, "images"), n=20)

    # 2. Write train/val CSVs (80/20 split)
    split = int(len(rows) * 0.8)
    for name, subset in [("train.csv", rows[:split]), ("val.csv", rows[split:])]:
        p = os.path.join(tmp_dir, name)
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(subset)

    # 3. Train for 3 epochs
    config = {
        "train_csv": os.path.join(tmp_dir, "train.csv"),
        "val_csv": os.path.join(tmp_dir, "val.csv"),
        "vocab_path": os.path.join(tmp_dir, "vocab.json"),
        "epochs": 3,
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "patience": 15,
        "save_dir": os.path.join(tmp_dir, "weights"),
        "log_csv": os.path.join(tmp_dir, "weights", "training_log.csv"),
    }
    train(config)

    # 4. Inference on one image
    ckpt_path = os.path.join(tmp_dir, "weights", "best.pth")
    if os.path.exists(ckpt_path):
        vocab = Vocabulary.load(config["vocab_path"])
        model = load_model(ckpt_path, vocab)
        preprocessor = Preprocessor(target_height=64, is_training=False)
        pred = recognise_line(rows[0][0], model, vocab, preprocessor)
        print(f"\n[smoke] Inference result: '{pred}'")
        print(f"[smoke] Ground truth:     '{rows[0][1]}'")
    else:
        print("[smoke] No checkpoint saved — training may have failed.")

    print("\n[smoke] ✓ Pipeline smoke test complete.")


if __name__ == "__main__":
    main()
