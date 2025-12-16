#!/usr/bin/env python3
import os
import random
import shutil
from typing import Dict, List, Tuple

SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "frames")
DST_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "frames_split")
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def gather_images(root: str) -> Dict[str, List[str]]:
    label_to_files: Dict[str, List[str]] = {}
    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort()
        if files:
            label_to_files[label] = files
    return label_to_files


def split_list(items: List[str], ratios: Tuple[float, float, float]) -> Tuple[List[str], List[str], List[str]]:
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def copy_files(files: List[str], dst_label_dir: str) -> None:
    ensure_dir(dst_label_dir)
    for src in files:
        fname = os.path.basename(src)
        dst = os.path.join(dst_label_dir, fname)
        shutil.copy2(src, dst)


def main() -> int:
    random.seed(SEED)
    ensure_dir(DST_ROOT)
    mapping = gather_images(SRC_ROOT)
    if not mapping:
        print(f"No images found under {SRC_ROOT}")
        return 1
    total = 0
    for label, files in mapping.items():
        items = files[:]
        random.shuffle(items)
        train, val, test = split_list(items, (SPLITS["train"], SPLITS["val"], SPLITS["test"]))
        copy_files(train, os.path.join(DST_ROOT, "train", label))
        copy_files(val, os.path.join(DST_ROOT, "val", label))
        copy_files(test, os.path.join(DST_ROOT, "test", label))
        print(f"{label}: train={len(train)} val={len(val)} test={len(test)} total={len(items)}")
        total += len(items)
    print(f"Done. Total images: {total}. Output: {DST_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





