from __future__ import annotations

import argparse
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupShuffleSplit


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def metadata_from_path(path):
    parts = [part.upper() for part in path.parts]
    site = None
    # Official archive directory aliases: parking1a/parking1b are the two
    # views of the UFPR lot; parking2 is the PUCPR lot.
    aliases = {
        "UFPR04": "UFPR04", "PARKING1A": "UFPR04",
        "UFPR05": "UFPR05", "PARKING1B": "UFPR05",
        "PUCPR": "PUCPR", "PUC": "PUCPR", "PARKING2": "PUCPR",
    }
    for part in parts:
        if part in aliases:
            site = aliases[part]
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
    if dataset_root.is_dir() and dataset_root.name.lower() == "pklotsegmented":
        return dataset_root
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


def collect_from_xml(dataset_root, crop_dir, max_crops=10000, seed=42):
    """Crop only sampled source frames, stopping once the requested crop cap is reached."""
    grouped = {}
    for xml_path in dataset_root.rglob("*.xml"):
        metadata = metadata_from_path(xml_path)
        if metadata is None:
            continue
        site, weather, date = metadata
        grouped.setdefault((site, weather, date), []).append(xml_path)

    # Shuffle within each site/weather/date group, then round-robin the groups.
    # This avoids spending the whole crop budget on the first folder in the archive.
    rng = random.Random(seed)
    for paths in grouped.values():
        rng.shuffle(paths)
    xml_paths = []
    group_keys = sorted(grouped)
    while group_keys:
        remaining = []
        for key in group_keys:
            paths = grouped[key]
            if paths:
                xml_paths.append(paths.pop())
            if paths:
                remaining.append(key)
        group_keys = remaining

    records = []
    for xml_path in tqdm(xml_paths, desc="Cropping sampled PKLot frames"):
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
            points = []
            contour = space.find("contour")
            if contour is not None:
                for point in contour.findall("point"):
                    try:
                        points.append([float(point.get("x")), float(point.get("y"))])
                    except (TypeError, ValueError):
                        points = []
                        break

            if len(points) != 4:
                rotated = space.find("rotatedRect")
                try:
                    center = rotated.find("center")
                    size = rotated.find("size")
                    angle = rotated.find("angle")
                    rect = (
                        (float(center.get("x")), float(center.get("y"))),
                        (float(size.get("w")), float(size.get("h"))),
                        float(angle.get("d")),
                    )
                    points = cv2.boxPoints(rect).tolist()
                except (AttributeError, TypeError, ValueError):
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
                ok = cv2.imwrite(
                    str(destination), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not ok:
                    continue
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
            if len(records) >= max_crops:
                return records
    return records


def cap_records(frame, max_crops, seed):
    """Randomly sample no more than max_crops while preserving class proportions."""
    if max_crops <= 0 or len(frame) <= max_crops:
        return frame.reset_index(drop=True)
    labels = frame["label"].astype(int)
    proportions = labels.value_counts(normalize=True).sort_index()
    allocation = {
        int(label): int(round(float(frac) * max_crops))
        for label, frac in proportions.items()
    }
    # Correct rounding so the allocation sums exactly to max_crops.
    while sum(allocation.values()) < max_crops:
        label = int(proportions.idxmax())
        allocation[label] += 1
    while sum(allocation.values()) > max_crops:
        label = max(allocation, key=allocation.get)
        allocation[label] -= 1

    pieces = []
    for label, count in allocation.items():
        group = frame.loc[labels == label]
        count = min(count, len(group))
        if count:
            pieces.append(group.sample(n=count, random_state=seed + label))
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assign_splits(frame, seed):
    """Create approximately 70/15/15 splits without sharing a capture date."""
    if frame["date"].nunique() < 3:
        raise RuntimeError("At least three distinct capture dates are required for 70/15/15 splitting.")

    total = len(frame)
    overall_positive = float(frame["label"].mean())
    overall_sites = frame["site"].value_counts(normalize=True).to_dict()
    groups = frame["date"].astype(str)
    labels = frame["label"].astype(int)

    def score_part(part, target_fraction):
        if part.empty:
            return 100.0
        size_error = abs(len(part) / total - target_fraction)
        label_error = abs(float(part["label"].mean()) - overall_positive)
        site_mix = part["site"].value_counts(normalize=True).to_dict()
        site_error = sum(
            abs(site_mix.get(site, 0.0) - frac)
            for site, frac in overall_sites.items()
        )
        missing_class_penalty = 1.0 if part["label"].nunique() < 2 else 0.0
        return 4.0 * size_error + 2.0 * label_error + 0.5 * site_error + missing_class_penalty

    outer = GroupShuffleSplit(n_splits=40, test_size=0.30, random_state=seed)
    outer_candidates = []
    for train_idx, hold_idx in outer.split(frame, labels, groups):
        train_part = frame.iloc[train_idx]
        hold_part = frame.iloc[hold_idx]
        score = score_part(train_part, 0.70) + score_part(hold_part, 0.30)
        outer_candidates.append((score, train_idx, hold_idx))
    outer_candidates.sort(key=lambda item: item[0])

    best = None
    best_score = float("inf")
    for _, train_idx, hold_idx in outer_candidates[:5]:
        holdout = frame.iloc[hold_idx].reset_index(drop=True)
        inner = GroupShuffleSplit(
            n_splits=60, test_size=0.50, random_state=seed + 1
        )
        for val_idx, test_idx in inner.split(
            holdout, holdout["label"].astype(int), holdout["date"].astype(str)
        ):
            train_part = frame.iloc[train_idx]
            val_part = holdout.iloc[val_idx]
            test_part = holdout.iloc[test_idx]
            score = (
                score_part(train_part, 0.70)
                + score_part(val_part, 0.15)
                + score_part(test_part, 0.15)
            )
            if score < best_score:
                best_score = score
                best = (train_part.copy(), val_part.copy(), test_part.copy())

    if best is None:
        raise RuntimeError("Could not create train/validation/test splits.")
    train_part, val_part, test_part = best
    train_part["split"] = "train"
    val_part["split"] = "val"
    test_part["split"] = "test"
    result = pd.concat([train_part, val_part, test_part], ignore_index=True)

    date_sets = {
        name: set(group["date"].astype(str))
        for name, group in result.groupby("split")
    }
    if date_sets["train"] & date_sets["val"]:
        raise AssertionError("Capture-date leakage between train and validation.")
    if date_sets["train"] & date_sets["test"]:
        raise AssertionError("Capture-date leakage between train and test.")
    if date_sets["val"] & date_sets["test"]:
        raise AssertionError("Capture-date leakage between validation and test.")
    return result.sort_values(["split", "site", "date", "path"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a maximum of 10,000 PKLot crops with a date-grouped 70/15/15 split."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-crops", type=int, default=10000)
    args = parser.parse_args()

    if args.max_crops < 3:
        raise ValueError("--max-crops must be at least 3")
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_segmented(dataset_root)
    source_mode = "PKLotSegmented"
    if not records:
        source_mode = "XML full-frame annotations (sampled until crop limit)"
        records = collect_from_xml(
            dataset_root, args.output_dir / "crops",
            max_crops=args.max_crops, seed=args.seed,
        )
    if not records:
        raise RuntimeError(
            "No PKLot records found. Expected PKLotSegmented/.../Empty|Occupied images "
            "or full-frame JPG files with matching XML annotations."
        )

    frame = pd.DataFrame(records).drop_duplicates(subset=["path"])
    collected_before_cap = int(len(frame))
    frame = cap_records(frame, args.max_crops, args.seed)
    frame = assign_splits(frame, args.seed)

    manifest_path = args.output_dir / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    split_counts = {
        str(key): int(value)
        for key, value in frame["split"].value_counts().items()
    }
    split_ratios = {
        key: float(value / len(frame)) for key, value in split_counts.items()
    }
    summary = {
        "source_mode": source_mode,
        "dataset_root": str(dataset_root),
        "crops_collected_before_cap": collected_before_cap,
        "selected_crops": int(len(frame)),
        "max_crops": int(args.max_crops),
        "seed": int(args.seed),
        "target_split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "actual_split_counts": split_counts,
        "actual_split_ratios": split_ratios,
        "class_counts": {
            str(split): {
                str(key): int(value)
                for key, value in group["class_name"].value_counts().items()
            }
            for split, group in frame.groupby("split")
        },
        "site_counts": {
            str(split): {
                str(key): int(value)
                for key, value in group["site"].value_counts().items()
            }
            for split, group in frame.groupby("split")
        },
        "weather_counts": {
            str(split): {
                str(key): int(value)
                for key, value in group["weather"].value_counts().items()
            }
            for split, group in frame.groupby("split")
        },
        "unique_dates_by_split": {
            str(key): int(value)
            for key, value in frame.groupby("split")["date"].nunique().items()
        },
    }
    (args.output_dir / "preparation_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Source mode: {source_mode}")
    print(f"Crops collected: {collected_before_cap:,}")
    print(f"Selected crops (maximum {args.max_crops:,}): {len(frame):,}")
    print("\nCounts by split, site, and class:")
    print(frame.groupby(["split", "site", "class_name"]).size().to_string())
    print("\nActual split counts and ratios:")
    print(pd.DataFrame({"count": split_counts, "ratio": split_ratios}).to_string())
    print("\nUnique dates per split:")
    print(frame.groupby("split")["date"].nunique().to_string())
    print("\nCapture-date leakage check passed. Dataset files are not tracked by Git.")


if __name__ == "__main__":
    main()
