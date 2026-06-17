from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .config import RNG


def load_image(path: str | Path, color: bool = True) -> np.ndarray:
    import cv2

    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_dataset(cfg: dict, split: str | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    import fnmatch

    base = cfg["abs_path"]
    if split is not None:
        base = base / split

    pattern = cfg["glob"].lower()
    images: list[np.ndarray] = []
    labels: list[str] = []
    for subfolder, label in cfg["classes"].items():
        folder = base / subfolder
        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern)
        )
        if not paths:
            raise FileNotFoundError(f"no files matching {cfg['glob']!r} in {folder}")
        for p in tqdm(paths, desc=f"{cfg['name']}/{label}"):
            images.append(load_image(p))
            labels.append(label)

    return images, np.array(labels)


def load_presplit(cfg: dict) -> tuple[list, np.ndarray, list, np.ndarray]:
    X_train, y_train = load_dataset(cfg, split="train")
    X_test, y_test = load_dataset(cfg, split="test")
    return X_train, y_train, X_test, y_test


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    presplit: tuple | None = None,
) -> dict:
    if presplit is not None:
        X_test, y_test = presplit
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, stratify=y, random_state=RNG
        )
    else:
        X_tr, X_test, y_tr, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=RNG
        )
        rel_val = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=rel_val, stratify=y_tr, random_state=RNG
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
