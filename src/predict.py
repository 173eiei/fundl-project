from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model import ParkingCNN


def main():
    parser = argparse.ArgumentParser(description="Classify one cropped parking-space image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to one parking-space crop.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/models/best.pt"))
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    size = int(checkpoint.get("image_size", 128))
    names = checkpoint.get("class_names", ["Empty", "Occupied"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(args.image) as source:
        image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    model = ParkingCNN(num_classes=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
    prediction = int(probabilities.argmax())
    print(f"Prediction: {names[prediction]}")
    print(f"Probability Empty: {probabilities[0]:.4f}")
    print(f"Probability Occupied: {probabilities[1]:.4f}")


if __name__ == "__main__":
    main()
