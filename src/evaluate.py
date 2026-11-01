from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import ParkingCropDataset
from src.model import ParkingCNN


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the final model on the held-out PUCPR test site."
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/models/best.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-failures", type=int, default=12)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(checkpoint.get("image_size", 128))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = ParkingCropDataset(args.manifest, "test", image_size, augment=False)
    loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = ParkingCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    targets, predictions, probabilities = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Held-out test evaluation"):
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(targets, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    frame = test_set.frame.copy()
    frame["true_label"] = y_true
    frame["predicted_label"] = y_pred
    frame["prob_occupied"] = probabilities
    frame["correct"] = frame["true_label"] == frame["predicted_label"]

    train_labels = pd.read_csv(args.manifest).query("split == 'train'")["label"].astype(int)
    majority_label = int(train_labels.value_counts().idxmax())
    baseline_predictions = np.full_like(y_true, majority_label)
    metrics = {
        "test_site": "PUCPR",
        "test_samples": int(len(y_true)),
        "model": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_occupied": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall_occupied": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_occupied": float(f1_score(y_true, y_pred, zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        },
        "majority_baseline": {
            "class": "Occupied" if majority_label == 1 else "Empty",
            "label": majority_label,
            "accuracy": float(accuracy_score(y_true, baseline_predictions)),
            "f1_occupied": float(f1_score(y_true, baseline_predictions, zero_division=0)),
        },
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Empty", "Occupied"], output_dict=True,
            zero_division=0,
        ),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_validation_loss": checkpoint.get("best_val_loss"),
    }
    output = Path("outputs/results")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "test_predictions.csv", index=False)
    (output / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["Empty", "Occupied"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Held-out PUCPR test set")
    fig.tight_layout()
    fig.savefig(output / "test_confusion_matrix.png", dpi=160)
    plt.show()
    successes_dir = output / "success_cases"
    successes_dir.mkdir(parents=True, exist_ok=True)
    correct_rows = frame.loc[frame["correct"]].head(3)
    for index, row in correct_rows.iterrows():
        source = Path(str(row["path"]))
        if source.is_file():
            name = (
                f"{index:06d}_true-{int(row['true_label'])}_"
                f"pred-{int(row['predicted_label'])}_{source.name}"
            )
            shutil.copy2(source, successes_dir / name)

    failures_dir = output / "failure_cases"
    failures_dir.mkdir(parents=True, exist_ok=True)
    failures = frame.loc[~frame["correct"]].head(args.max_failures)
    for index, row in failures.iterrows():
        source = Path(str(row["path"]))
        if source.is_file():
            name = (
                f"{index:06d}_true-{int(row['true_label'])}_"
                f"pred-{int(row['predicted_label'])}_{source.name}"
            )
            shutil.copy2(source, failures_dir / name)
    print(json.dumps(metrics["model"], indent=2))
    print("Majority baseline:", json.dumps(metrics["majority_baseline"], indent=2))
    print(f"Saved test metrics, predictions, confusion matrix, {len(correct_rows)} success examples, and {len(failures)} failure examples to {output}")


if __name__ == "__main__":
    main()
