"""
fine_tune_trocr.py -- Fine-tune TrOCR on personal correction dataset.

Expected dataset structure:
    data/personal_dataset/
        images/
            <uuid>.png
        labels.csv

labels.csv should include columns:
    image_file, corrected_text

Usage:
    python scripts/fine_tune_trocr.py
    python scripts/fine_tune_trocr.py --epochs 4 --batch-size 2
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


@dataclass
class Sample:
    image_path: str
    text: str


def _load_samples(labels_csv: str, images_dir: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_file = (row.get("image_file") or "").strip()
            corrected = (row.get("corrected_text") or row.get("text") or "").strip()
            if not image_file or not corrected:
                continue
            image_path = os.path.join(images_dir, image_file)
            if os.path.isfile(image_path):
                samples.append(Sample(image_path=image_path, text=corrected))
    return samples


def _split_train_val(
    samples: list[Sample], val_split: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    if len(samples) < 2 or val_split <= 0:
        return samples, []
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_split))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1
    val = shuffled[:val_count]
    train = shuffled[val_count:]
    return train, val


class OCRLineDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        processor: TrOCRProcessor,
        max_target_length: int = 96,
    ):
        self.samples = samples
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            sample.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on personal dataset.")
    parser.add_argument("--dataset-root", default="data/personal_dataset")
    parser.add_argument("--labels-csv", default="")
    parser.add_argument("--images-dir", default="")
    parser.add_argument("--model", default="microsoft/trocr-base-handwritten")
    parser.add_argument("--output-dir", default="weights/trocr/model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()

    labels_csv = args.labels_csv or os.path.join(args.dataset_root, "labels.csv")
    images_dir = args.images_dir or os.path.join(args.dataset_root, "images")
    if not os.path.isfile(labels_csv):
        raise SystemExit(f"labels.csv not found: {labels_csv}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"images directory not found: {images_dir}")

    samples = _load_samples(labels_csv, images_dir)
    if len(samples) < 2:
        raise SystemExit(
            f"Need at least 2 samples to fine-tune. Found {len(samples)} in {labels_csv}."
        )

    train_samples, val_samples = _split_train_val(samples, args.val_split, args.seed)
    print(f"Loaded {len(samples)} samples -> train={len(train_samples)}, val={len(val_samples)}")

    processor = TrOCRProcessor.from_pretrained(args.model, use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    train_ds = OCRLineDataset(
        train_samples, processor=processor, max_target_length=args.max_target_length
    )
    eval_ds = (
        OCRLineDataset(val_samples, processor=processor, max_target_length=args.max_target_length)
        if val_samples
        else None
    )

    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        report_to=[],
        fp16=use_cuda,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=_collate_batch,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Fine-tuned model saved to: {args.output_dir}")
    print("Next: re-export to ONNX via scripts/export_onnx_trocr.py")


if __name__ == "__main__":
    main()

