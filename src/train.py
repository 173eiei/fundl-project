from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from src.data import make_loaders
from src.model import ParkingCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets, predictions = [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())
    return (
        total_loss / len(loader.dataset),
        float(accuracy_score(targets, predictions)),
        float(f1_score(targets, predictions, zero_division=0)),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a custom CNN on PKLot crops.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--max-val-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        raise ValueError("epochs/batch-size must be positive and image-size must be >= 32")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    train_loader, val_loader = make_loaders(
        args.manifest, args.batch_size, args.image_size, args.num_workers, args.seed,
        args.max_train_samples, args.max_val_samples,
    )
    # Calculate class weights from the exact training subset presented to the optimizer.
    train_labels = train_loader.dataset.frame["label"].astype(int)
    class_counts = np.bincount(train_labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(f"Both classes must appear in train split; counts={class_counts.tolist()}")
    class_weights = len(train_labels) / (2.0 * class_counts.astype(np.float64))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model = ParkingCNN(num_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    model_dir = Path("outputs/models")
    results_dir = Path("outputs/results")
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
            "train_f1_occupied": train_f1, "val_loss": val_loss,
            "val_accuracy": val_acc, "val_f1_occupied": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} "
            f"acc {train_acc:.4f} | val loss {val_loss:.4f} "
            f"acc {val_acc:.4f} F1 {val_f1:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "ParkingCNN",
                    "num_classes": 2,
                    "image_size": args.image_size,
                    "class_names": ["Empty", "Occupied"],
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "hyperparameters": vars(args),
                },
                model_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without validation-loss improvement.")
                break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(results_dir / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
    axes[0].plot(history_frame["epoch"], history_frame["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history_frame["epoch"], history_frame["train_accuracy"], label="Train")
    axes[1].plot(history_frame["epoch"], history_frame["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curves.png", dpi=160)
    plt.show()
    summary = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "epochs_completed": len(history),
        "hyperparameters": vars(args),
        "train_class_counts": {str(i): int(v) for i, v in enumerate(class_counts)},
    }
    (results_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Best checkpoint: {model_dir / 'best.pt'}")
    print(f"History and curves: {results_dir}")


if __name__ == "__main__":
    main()
