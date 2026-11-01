from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = ["Empty", "Occupied"]


class ParkingCropDataset(Dataset):
    """Load individual PKLot parking-space crops from a manifest CSV."""

    def __init__(self, manifest, split, image_size=128, augment=False):
        frame = pd.read_csv(manifest)
        required = {"path", "label", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No records found for split={split!r} in {manifest}")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        labels = set(self.frame["label"].astype(int).unique())
        if not labels.issubset({0, 1}):
            raise ValueError(f"Expected binary labels 0/1, found {sorted(labels)}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = Path(str(row["path"]))
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Crop not found: {image_path}. Re-run src.prepare_data before training."
            )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
            if random.random() < 0.6:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.15))
        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def make_loaders(manifest, batch_size=64, image_size=128, num_workers=2, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_set = ParkingCropDataset(manifest, "train", image_size, augment=True)
    val_set = ParkingCropDataset(manifest, "val", image_size, augment=False)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader
