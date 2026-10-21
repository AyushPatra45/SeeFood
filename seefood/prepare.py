"""Inspect data and prevent detected exact/near duplicates from crossing split boundaries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from seefood.data import open_rgb
from seefood.model import CLASS_TO_IDX


def fingerprints(image):
    digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    thumb = np.array(image.convert("L").resize((9, 8)))
    bits = (thumb[:, 1:] > thumb[:, :-1]).flatten()
    dhash = sum(int(bit) << i for i, bit in enumerate(bits))
    return digest, dhash


def prepare(root, output, seed=42):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # This Kaggle ZIP packages the same corpus at the root AND under seefood/.
    # Pick one train/test tree rather than counting both copies as examples.
    if (root / "train").is_dir() and (root / "test").is_dir():
        dataset_root = root
    else:
        candidates = [p.parent for p in sorted(root.rglob("train")) if (p.parent / "test").is_dir()]
        if len(candidates) != 1:
            raise ValueError("Could not identify one train/test dataset tree")
        dataset_root = candidates[0]
    rows, invalid = [], []
    for path in sorted(dataset_root.rglob("*")):
        if path.relative_to(dataset_root).parts[0] not in {"train", "test"}:
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        class_name = next((p for p in reversed(path.parts[:-1]) if p in CLASS_TO_IDX), None)
        split = next((p for p in reversed(path.parts[:-1]) if p in {"train", "test"}), None)
        if class_name is None or split is None:
            continue
        try:
            image = open_rgb(path)
            sha, dhash = fingerprints(image)
            rows.append({"path": path.relative_to(root).as_posix(), "label": CLASS_TO_IDX[class_name],
                         "original_split": split, "sha256_pixels": sha, "dhash": dhash})
        except Exception as exc:
            invalid.append({"path": path.relative_to(root).as_posix(), "error": str(exc)})
    if not rows or not any(r["original_split"] == "test" for r in rows):
        raise ValueError("Expected Kaggle train/{hot_dog,not_hot_dog} and test/{...} folders")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    near_pairs = 0
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            if row["sha256_pixels"] == other["sha256_pixels"] or (row["dhash"] ^ other["dhash"]).bit_count() <= 2:
                parents[find(i)] = find(j)
                near_pairs += 1
    for i, row in enumerate(rows):
        row["group"] = find(i)
    test_groups = {r["group"] for r in rows if r["original_split"] == "test"}
    excluded = [r for r in rows if r["original_split"] == "train" and r["group"] in test_groups]
    training = [r for r in rows if r["original_split"] == "train" and r["group"] not in test_groups]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(training, [r["label"] for r in training], [r["group"] for r in training]))
    for indices, split in [(train_idx, "train"), (val_idx, "val")]:
        for index in indices:
            training[index]["split"] = split
    test = [dict(r, split="test") for r in rows if r["original_split"] == "test"]
    manifest = training + test
    groups = {s: {r["group"] for r in manifest if r["split"] == s} for s in ["train", "val", "test"]}
    assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
    audit = {"source": "https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog",
             "dataset_subdirectory": dataset_root.relative_to(root).as_posix(),
             "seed": seed, "class_to_idx": CLASS_TO_IDX,
             "counts": {s: dict(Counter("hot_dog" if r["label"] else "not_hot_dog" for r in manifest if r["split"] == s)) for s in groups},
             "invalid_files": invalid, "near_duplicate_pairs_detected": near_pairs,
             "excluded_training_paths_overlapping_test": [r["path"] for r in excluded],
             "duplicate_policy": "Decoded RGB SHA256 or 64-bit dHash distance <=2 groups; preserve official test; exclude overlapping train images; group-aware stratified train/val split.",
             "limitations": "Perceptual hashing is a heuristic, not proof that every semantic duplicate is detected. Official test may contain within-test duplicates."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    download = root.parent / "download.json"
    if download.exists():
        audit["download"] = json.loads(download.read_text())
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--output", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data, args.output, args.seed)
