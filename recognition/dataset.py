import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import sys

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocessing.clean import Preprocessor
from recognition.vocab import Vocabulary


class HandwritingDataset(Dataset):
    """
    PyTorch Dataset for handwriting line images.

    Expects a CSV file with columns: image_path, transcription
    """

    def __init__(self, csv_path, vocab, target_height=64, is_training=False, apply_deskew=True):
        """
        Args:
            csv_path: Path to CSV with (image_path, transcription) rows.
            vocab: Vocabulary instance for encoding labels.
            target_height: Fixed image height.
            is_training: Enables augmentation.
            apply_deskew: If True, preprocessor runs deskew. Set False for IAM (already-cropped lines).
        """
        self.vocab = vocab
        self.preprocessor = Preprocessor(
            target_height=target_height, is_training=is_training, apply_deskew=apply_deskew
        )
        self.samples = []  # list of (image_path, transcription)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    img_path, transcription = row[0].strip(), row[1].strip()
                    if os.path.exists(img_path) and len(transcription) > 0:
                        self.samples.append((img_path, transcription))

        print(f"[HandwritingDataset] Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, transcription = self.samples[idx]

        # Preprocess image -> tensor (1, H, W)
        image_tensor = self.preprocessor.process(img_path)

        # Encode label
        encoded_label = self.vocab.encode(transcription)
        label_tensor = torch.tensor(encoded_label, dtype=torch.long)
        label_length = len(encoded_label)

        return image_tensor, label_tensor, label_length


def collate_fn(batch):
    """
    Custom collation for CTC training.

    - Pads images to the max width in the batch (right-pad with zeros).
    - Concatenates labels into a flat 1D tensor (CTC requirement).
    - Returns input_lengths and target_lengths tensors.
    """
    images, labels, label_lengths = zip(*batch)

    # Per-sample input lengths (actual width before padding) — required for correct CTC loss
    # CNN downsamples width by factor of 32; use each image's true width, not max_width
    input_lengths = torch.tensor([img.shape[2] // 32 for img in images], dtype=torch.long)

    # Determine max width in this batch
    max_width = max(img.shape[2] for img in images)

    # Pad each image on the right to max_width
    padded_images = []
    for img in images:
        # img shape: (1, H, W)
        pad_amount = max_width - img.shape[2]
        if pad_amount > 0:
            # Pad on the right side only: (left, right, top, bottom)
            padded = torch.nn.functional.pad(img, (0, pad_amount, 0, 0), value=0)
        else:
            padded = img
        padded_images.append(padded)

    # Stack into (batch, 1, H, max_W)
    image_batch = torch.stack(padded_images, dim=0)

    # Concatenate all labels into a single flat 1D tensor
    label_batch = torch.cat(labels, dim=0)

    # Label lengths
    target_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return image_batch, label_batch, input_lengths, target_lengths


def create_dataloaders(
    train_csv, val_csv, vocab, batch_size=32, target_height=64, num_workers=0, apply_deskew=True
):
    """
    Convenience function that returns train and validation DataLoaders.
    Set apply_deskew=False for IAM to avoid over-rotation of pre-cropped line images.
    """
    train_ds = HandwritingDataset(
        train_csv, vocab, target_height=target_height, is_training=True, apply_deskew=apply_deskew
    )
    val_ds = HandwritingDataset(
        val_csv, vocab, target_height=target_height, is_training=False, apply_deskew=apply_deskew
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # No CUDA
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke test with a tiny CSV
    import numpy as np
    import cv2

    os.makedirs("tests", exist_ok=True)

    # Create 4 dummy images and a CSV
    csv_rows = []
    for i in range(4):
        img = np.ones((200, 400 + i * 100), dtype=np.uint8) * 255
        text = f"sample {i}"
        cv2.putText(img, text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,), 3)
        path = f"tests/dummy_{i}.jpg"
        cv2.imwrite(path, img)
        csv_rows.append((os.path.abspath(path), text))

    csv_path = "tests/dummy_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    vocab = Vocabulary()
    ds = HandwritingDataset(csv_path, vocab, target_height=64, is_training=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    for images, labels, input_lengths, target_lengths in loader:
        print(f"Image batch : {images.shape}")
        print(f"Labels      : {labels.shape}")
        print(f"Input lens  : {input_lengths}")
        print(f"Target lens : {target_lengths}")
        break
