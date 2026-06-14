from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .config import RNG


def load_image(path: str | Path, color: bool = True) -> np.ndarray:
    """
    Load a single image from disk as a ``uint8`` ndarray.
    """
    import cv2

    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:  # cv2 returns None for unreadable/corrupt files (no exception)
        raise FileNotFoundError(f"could not read image: {path}")
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # whole project assumes RGB
    return img


def load_dataset(cfg: dict, split: str | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Load every image of a flat (or one split of a pre-split) dataset.

    Parameters
    ----------
    cfg : dict
        A resolved config from :func:`cvproject.config.get_config`.
    split : {"train", "test", None}
        For pre-split datasets, which subfolder to read. ``None`` for flat sets.

    Returns
    -------
    (list[np.ndarray], np.ndarray)
        Variable-size images (a list, NOT a stacked array) and string labels.
    """
    import fnmatch

    base = cfg["abs_path"]
    if split is not None:
        base = base / split

    pattern = cfg["glob"].lower()  # case-insensitive: files may be .jpg/.JPG/.jpeg
    images: list[np.ndarray] = []
    labels: list[str] = []
    for subfolder, label in cfg["classes"].items():
        folder = base / subfolder
        # sorted -> reproducible splits; match case-insensitively across extensions
        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern)
        )
        if not paths:
            raise FileNotFoundError(
                f"no files matching {cfg['glob']!r} in {folder}"
            )
        for p in tqdm(paths, desc=f"{cfg['name']}/{label}"):
            images.append(load_image(p, color=cfg["color"]))
            labels.append(label)

    return images, np.array(labels)


def load_presplit(cfg: dict) -> tuple[list, np.ndarray, list, np.ndarray]:
    """
    Load a dataset that ships its own train/ and test/ folders.

    Only ``apples_tomatoes`` has ``presplit=True``. Using the provided split is
    important: it gives the "separate test set" the TASK explicitly requires,
    with no leakage from our own splitting.
    """
    assert cfg.get("presplit"), "load_presplit only for pre-split datasets"
    X_train, y_train = load_dataset(cfg, split="train")
    X_test, y_test = load_dataset(cfg, split="test")
    return X_train, y_train, X_test, y_test


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integer ids.
    
    The returned ``le`` is needed later: ``le.classes_`` gives the human names
    (in id order) for confusion-matrix axes and ``classification_report``, and
    ``le.transform`` re-encodes the pre-split test labels with the *same* mapping.
    """
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
    """
    Produce a train / validation / test split (stratified, reproducible).

    Why three sets: tune hyper-parameters on train+val (the TASK's
    "train/validation"), and report final metrics ONCE on the untouched test set.

    Parameters
    ----------
    presplit : optional ``(X_test, y_test)``
        If given (pre-split dataset), it is used directly as the test set and a
        validation set is carved only out of ``(X, y)``.

    Returns
    -------
    dict
        ``{"X_train","y_train","X_val","y_val","X_test","y_test"}``.
    """
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
