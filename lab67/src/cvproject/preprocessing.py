from __future__ import annotations

import numpy as np
import cv2


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return _uint8(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    out = img.astype(np.float32)
    if out.max() <= 1.0:
        out = out * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_contrast(
    img: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if img.ndim == 2:
            return clahe.apply(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if method == "hist_eq":
        if img.ndim == 2:
            return cv2.equalizeHist(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    raise ValueError(f"unknown method {method!r}; expected 'clahe' or 'hist_eq'")


def white_balance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img

    out = img.astype(np.float32)
    means = out.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    scale = np.divide(gray, means, out=np.ones_like(means), where=means > 0)
    out *= scale
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_artifacts(img: np.ndarray, method: str = "median", ksize: int = 3) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1  # must be odd

    if method == "median":
        return cv2.medianBlur(img, ksize)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=75, sigmaSpace=75)
    if method == "nlmeans":
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    raise ValueError(f"unknown method {method!r}; expected 'median'/'bilateral'/'nlmeans'")


_HSV_RANGES = {
    "green": [((25, 40, 40), (95, 255, 255))],
    "warm": [((0, 60, 40), (25, 255, 255)), ((160, 60, 40), (179, 255, 255))],
}


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def segment_foreground(
    img: np.ndarray,
    method: str = "otsu",
    hsv_target: str = "warm",
) -> np.ndarray:
    if method == "otsu":
        g = to_gray(img)
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask.mean() > 127:
            mask = cv2.bitwise_not(mask)
    elif method == "hsv":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = np.zeros(img.shape[:2], np.uint8)
        for low, high in _HSV_RANGES[hsv_target]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(low), np.array(high)))
    elif method == "grabcut":
        h, w = img.shape[:2]
        gc = np.zeros((h, w), np.uint8)
        rect = (int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h))
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'otsu'/'hsv'/'grabcut'")

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return _largest_component(mask)


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.where(mask > 0, img, 0).astype(np.uint8)
    return cv2.bitwise_and(img, img, mask=mask)


def preprocess(
    img: np.ndarray,
    steps: tuple[str, ...] = ("remove_artifacts", "white_balance", "enhance_contrast", "segment"),
    segment_method: str = "hsv",
    hsv_target: str = "warm",
) -> tuple[np.ndarray, np.ndarray | None]:
    out = img.copy()
    mask: np.ndarray | None = None

    for step in steps:
        if step == "remove_artifacts":
            out = remove_artifacts(out)
        elif step == "white_balance":
            out = white_balance(out)  # no-op for grayscale
        elif step == "enhance_contrast":
            out = enhance_contrast(out)
        elif step == "segment":
            mask = segment_foreground(out, method=segment_method, hsv_target=hsv_target)
        else:
            raise ValueError(f"unknown preprocessing step {step!r}")

    return out, mask
