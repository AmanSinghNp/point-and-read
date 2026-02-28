from recognition.evaluate import evaluate, batch_greedy_decode, compute_cer
from recognition.dataset import HandwritingDataset, collate_fn, create_dataloaders
from recognition.vocab import Vocabulary
from recognition.model import CRNN
import os
import sys
import csv
import time
import torch
import torch.nn as nn

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def train(config):
    """
    Full training pipeline.

    config is a dict with keys:
        train_csv, val_csv, vocab_path,
        epochs, batch_size, lr, weight_decay,
        grad_clip, patience, save_dir, log_csv
    """
    try:
        import torch_directml
        device = torch_directml.device()
        print("[train] Using device: DirectML (AMD GPU)")
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[train] Using device: CUDA ({torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("[train] Using device: CPU")

    # ---- Vocabulary ----
    if os.path.exists(config["vocab_path"]):
        vocab = Vocabulary.load(config["vocab_path"])
        print(f"[train] Loaded vocabulary from {config['vocab_path']}")
    else:
        # Build from training transcriptions
        texts = []
        with open(config["train_csv"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    texts.append(row[1].strip())
        vocab = Vocabulary.build_from_texts(texts)
        vocab.save(config["vocab_path"])
        print(
            f"[train] Built vocabulary ({vocab.num_classes} chars) → saved to {config['vocab_path']}")

    print(f"[train] Vocabulary: {vocab}")

    # ---- DataLoaders ----
    # Set apply_deskew=False for IAM (already-cropped lines) to avoid over-rotation
    apply_deskew = config.get("apply_deskew", True)
    train_loader, val_loader = create_dataloaders(
        config["train_csv"],
        config["val_csv"],
        vocab,
        batch_size=config["batch_size"],
        target_height=64,
        num_workers=0,
        apply_deskew=apply_deskew,
    )

    # ---- Model ----
    model = CRNN(num_classes=vocab.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(
        f"[train] Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Loss, Optimiser, Scheduler ----
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # ---- Logging ----
    os.makedirs(config["save_dir"], exist_ok=True)
    log_path = config.get("log_csv", os.path.join(
        config["save_dir"], "training_log.csv"))
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss",
                        "val_cer", "val_wer", "lr", "time_s"])

    # ---- Training ----
    best_cer = float("inf")
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels, input_lengths, target_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)  # (T, batch, num_classes)

            loss = criterion(outputs, labels, input_lengths, target_lengths)

            loss.backward()

            # Gradient clipping — essential for LSTMs
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["grad_clip"])

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, labels, input_lengths, target_lengths in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(
                    outputs, labels, input_lengths, target_lengths)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        # Compute CER & WER on validation set
        val_cer, val_wer = evaluate(model, val_loader, vocab, device=device)

        # Step scheduler on validation CER
        scheduler.step(val_cer)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:3d}/{config['epochs']} | "
            f"TrainLoss: {avg_train_loss:.4f} | "
            f"ValLoss: {avg_val_loss:.4f} | "
            f"CER: {val_cer:.4f} | "
            f"WER: {val_wer:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Log to CSV
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}",
                             f"{val_cer:.6f}", f"{val_wer:.6f}", f"{current_lr:.8f}",
                             f"{epoch_time:.2f}"])

        # ---- Checkpoint ----
        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0
            ckpt_path = os.path.join(config["save_dir"], "best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_cer": best_cer,
                "vocab_path": config["vocab_path"],
            }, ckpt_path)
            print(
                f"  ✓ New best CER {best_cer:.4f} — checkpoint saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(
                    f"  ✗ Early stopping after {config['patience']} epochs without improvement.")
                break

    print(f"\n[train] Done. Best validation CER: {best_cer:.4f}")
    return best_cer


if __name__ == "__main__":
    # ---- Default config — override paths as needed ----
    config = {
        "train_csv": "data/processed/train.csv",
        "val_csv": "data/processed/val.csv",
        "vocab_path": "data/processed/vocab.json",
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "patience": 15,
        "save_dir": "weights",
        "log_csv": "weights/training_log.csv",
        "apply_deskew": False,  # IAM lines are pre-cropped; set True for real-world photos
    }

    # Allow quick override from command line: python train.py <train_csv> <val_csv>
    if len(sys.argv) >= 3:
        config["train_csv"] = sys.argv[1]
        config["val_csv"] = sys.argv[2]

    train(config)
