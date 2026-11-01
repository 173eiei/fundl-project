from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def metadata_from_path(path):
    parts = [part.upper() for part in path.parts]
    site = None
    for candidate in ("UFPR04", "UFPR05", "PUCPR", "PUC"):
        if candidate in parts:
            site = "PUCPR" if candidate == "PUC" else candidate
            break
    if site is None:
        return None
    weather = "unknown"
    for candidate in ("SUNNY", "CLOUDY", "OVERCAST", "RAINY"):
        if candidate in parts:
            weather = "cloudy" if candidate == "OVERCAST" else candidate.lower()
            break
    match = DATE_PATTERN.search(str(path))
    if match is None:
        return None
    return site, weather, match.group(1)


def find_segmented_root(dataset_root):
    for candidate in dataset_root.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == "pklotsegmented":
            return candidate
    return None


def collect_segmented(dataset_root):
    segmented_root = find_segmented_root(dataset_root)
    if segmented_root is None:
        return []
    records = []
    image_paths = (
        path for path in segmented_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.parent.name.lower() in {"empty", "occupied"}
    )
    for image_path in tqdm(image_paths, desc="Indexing existing PKLot crops"):
        metadata = metadata_from_path(image_path)
        if metadata is None:
            continue
        site, weather, date = metadata
        label = 0 if image_path.parent.name.lower() == "empty" else 1
        records.append({
            "path": str(image_path.resolve()),
            "label": label,
            "class_name": "empty" if label == 0 else "occupied",
            "site": site,
            "weather": weather,
            "date": date,
            "frame_id": image_path.stem.split("#")[0],
            "space_id": image_path.stem.split("#")[-1] if "#" in image_path.stem else "",
        })
    return records


def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected 4 contour points, got shape {points.shape}")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.array([
        points[np.argmin(sums)],
        points[np.argmin(diffs)],
        points[np.argmax(sums)],
        points[np.argmax(diffs)],
    ], dtype=np.float32)


def crop_space(frame, points):
    quad = order_quad(points)
    width = int(max(
        np.linalg.norm(quad[2] - quad[3]),
        np.linalg.norm(quad[1] - quad[0]),
    ))
    height = int(max(
        np.linalg.norm(quad[1] - quad[2]),
        np.linalg.norm(quad[0] - quad[3]),
    ))
    if width < 8 or height < 8:
        return None
    destination = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad, destination)
    crop = cv2.warpPerspective(frame, transform, (width, height))
    return crop if crop.size else None


def collect_from_xml(dataset_root, crop_dir):
    records = []
    xml_paths = list(dataset_root.rglob("*.xml"))
    for xml_path in tqdm(xml_paths, desc="Cropping parking spaces from XML"):
        metadata = metadata_from_path(xml_path)
        if metadata is None:
            continue
        image_path = xml_path.with_suffix(".jpg")
        if not image_path.is_file():
            image_path = xml_path.with_suffix(".JPG")
        if not image_path.is_file():
            continue
        site, weather, date = metadata
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError):
            continue
        for space in root.findall(".//space"):
            occupied_value = space.get("occupied")
            if occupied_value not in {"0", "1"}:
                continue
            contour = space.find("contour")
            if contour is None:
                continue
            points = []
            for point in contour.findall("point"):
                try:
                    points.append([float(point.get("x")), float(point.get("y"))])
                except (TypeError, ValueError):
                    points = []
                    break
            if len(points) != 4:
                continue
            crop = crop_space(frame, np.asarray(points, dtype=np.float32))
            if crop is None:
                continue
            label = int(occupied_value)
            class_name = "occupied" if label == 1 else "empty"
            space_id = space.get("id", "unknown")
            destination = (
                crop_dir / site / weather / date / class_name /
                f"{image_path.stem}__space_{space_id}.jpg"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                cv2.imwrite(
                    str(destination), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
            records.append({
                "path": str(destination.resolve()),
                "label": label,
                "class_name": class_name,
                "site": site,
                "weather": weather,
                "date": date,
                "frame_id": image_path.stem,
                "space_id": str(space_id),
            })
    return records


def assign_splits(frame, seed, val_fraction):
    frame = frame.loc[frame["site"].isin({"UFPR04", "UFPR05", "PUCPR"})].copy()
    test_mask = frame["site"] == "PUCPR"
    trainval = frame.loc[~test_mask]
    dates = sorted(trainval["date"].unique())
    if len(dates) < 2:
        raise RuntimeError("Need at least two UFPR dates for a date-grouped train/validation split.")
    rng = np.random.default_rng(seed)
    shuffled_dates = rng.permutation(dates)
    val_count = max(1, int(round(len(dates) * val_fraction)))
    val_dates = set(shuffled_dates[:val_count])
    frame["split"] = "test"
    frame.loc[(~test_mask) & frame["date"].isin(val_dates), "split"] = "val"
    frame.loc[(~test_mask) & (~frame["date"].isin(val_dates)), "split"] = "train"
    train_dates = set(frame.loc[frame["split"] == "train", "date"])
    val_dates_actual = set(frame.loc[frame["split"] == "val", "date"])
    if train_dates.intersection(val_dates_actual):
        raise AssertionError("Date leakage detected between train and validation.")
    if set(frame.loc[frame["split"] == "test", "site"]) != {"PUCPR"}:
        raise AssertionError("Test split must contain only the held-out PUCPR site.")
    return frame.sort_values(["split", "site", "date", "path"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PKLot crops and a date-grouped, held-out-site split."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if not 0.05 <= args.val_fraction <= 0.40:
        raise ValueError("--val-fraction must be between 0.05 and 0.40")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = collect_segmented(dataset_root)
    source_mode = "PKLotSegmented"
    if not records:
        source_mode = "XML full-frame annotations"
        records = collect_from_xml(dataset_root, args.output_dir / "crops")
    if not records:
        raise RuntimeError(
            "No PKLot records found. Expected PKLotSegmented/.../Empty|Occupied images "
            "or full-frame JPG files with matching XML annotations."
        )
    frame = pd.DataFrame(records).drop_duplicates(subset=["path"])
    frame = assign_splits(frame, args.seed, args.val_fraction)
    manifest_path = args.output_dir / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    summary = {
        "source_mode": source_mode,
        "dataset_root": str(dataset_root),
        "total_crops": int(len(frame)),
        "seed": int(args.seed),
        "validation_fraction_by_date": float(args.val_fraction),
        "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts().items()},
        "class_counts": {
            str(split): {str(k): int(v) for k, v in group["class_name"].value_counts().items()}
            for split, group in frame.groupby("split")
        },
        "site_counts": {
            str(split): {str(k): int(v) for k, v in group["site"].value_counts().items()}
            for split, group in frame.groupby("split")
        },
        "unique_dates_by_split": {
            str(k): int(v) for k, v in frame.groupby("split")["date"].nunique().items()
        },
    }
    (args.output_dir / "preparation_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Source mode: {source_mode}")
    print(f"Manifest: {manifest_path}")
    print("\nCounts by split, site, and class:")
    print(frame.groupby(["split", "site", "class_name"]).size().to_string())
    print("\nUnique dates per split:")
    print(frame.groupby("split")["date"].nunique().to_string())
    print("\nDataset files are git-ignored and will not be pushed.")


if __name__ == "__main__":
    main()
