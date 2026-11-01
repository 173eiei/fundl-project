from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate a majority-class baseline on validation data.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    train = frame.loc[frame["split"] == "train"]
    val = frame.loc[frame["split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("Manifest must contain non-empty train and val splits.")
    counts = train["label"].astype(int).value_counts()
    majority_label = int(counts.idxmax())
    y_true = val["label"].astype(int).to_numpy()
    predictions = [majority_label] * len(val)
    result = {
        "baseline": "majority_class",
        "majority_label": majority_label,
        "majority_class": "Occupied" if majority_label == 1 else "Empty",
        "training_class_counts": {str(k): int(v) for k, v in counts.items()},
        "validation_samples": int(len(val)),
        "validation_accuracy": float(accuracy_score(y_true, predictions)),
        "validation_f1_binary_occupied": float(f1_score(y_true, predictions, zero_division=0)),
    }
    output = Path("outputs/results")
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
