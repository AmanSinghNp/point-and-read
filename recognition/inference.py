"""
inference.py — End-to-end inference: image → text.

Usage:
    python recognition/inference.py <image_path> [--checkpoint weights/best.pth] [--vocab data/processed/vocab.json]
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recognition.model import CRNN
from recognition.vocab import Vocabulary
from recognition.evaluate import greedy_decode, compute_cer, compute_wer
from preprocessing.clean import Preprocessor


def load_model(checkpoint_path, vocab, device="cpu"):
    """Load a trained CRNN from a checkpoint."""
    model = CRNN(num_classes=vocab.num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best_cer = ckpt.get("best_cer", "?")
    print(f"[inference] Loaded checkpoint from epoch {epoch} (CER {best_cer})")
    return model


def recognise_line(image_path, model, vocab, preprocessor, device="cpu"):
    """
    Preprocess a single line image and run CRNN recognition.

    Returns:
        Predicted text string.
    """
    tensor = preprocessor.process(image_path)  # (1, H, W)
    tensor = tensor.unsqueeze(0).to(device)     # (1, 1, H, W) — batch dim

    with torch.no_grad():
        output = model(tensor)  # (T, 1, num_classes)

    predicted_text = greedy_decode(output[:, 0, :], vocab)
    return predicted_text


def main():
    parser = argparse.ArgumentParser(description="CRNN handwriting recognition inference")
    parser.add_argument("image", help="Path to a handwritten line image")
    parser.add_argument("--checkpoint", default="weights/best.pth", help="Path to model checkpoint")
    parser.add_argument("--vocab", default="data/processed/vocab.json", help="Path to vocab.json")
    parser.add_argument("--gt", default=None, help="Optional ground-truth text for CER/WER calculation")
    args = parser.parse_args()

    device = torch.device("cpu")

    vocab = Vocabulary.load(args.vocab)
    model = load_model(args.checkpoint, vocab, device)
    preprocessor = Preprocessor(target_height=64, is_training=False)

    predicted_text = recognise_line(args.image, model, vocab, preprocessor, device)

    print(f"\nPredicted: {predicted_text}")

    if args.gt:
        cer = compute_cer(predicted_text, args.gt)
        wer = compute_wer(predicted_text, args.gt)
        print(f"Ground truth: {args.gt}")
        print(f"CER: {cer:.4f}  |  WER: {wer:.4f}")


if __name__ == "__main__":
    main()
