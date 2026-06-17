from __future__ import annotations

import numpy as np

LEVELS = 32
DISTANCES = [1, 3]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0/45/90/135
PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]


def quantize(img: np.ndarray, levels: int = LEVELS) -> np.ndarray:
    import cv2

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    q = (img.astype(np.float32) / 256.0 * levels).astype(np.int32)
    q = np.clip(q, 0, levels - 1)
    return q.astype(np.uint8)


def glcm_features(
    img: np.ndarray,
    levels: int = LEVELS,
    distances: list = DISTANCES,
    angles: list = ANGLES,
) -> np.ndarray:
    from skimage.feature import graycomatrix, graycoprops

    q = quantize(img, levels)
    glcm = graycomatrix(
        q, distances=distances, angles=angles,
        levels=levels, symmetric=True, normed=True,
    )
    feats = [graycoprops(glcm, prop).ravel() for prop in PROPS]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1)).ravel()
    feats.append(entropy)
    return np.concatenate(feats)


def _build_glcm_names() -> list[str]:
    angle_degs = [int(round(np.degrees(a))) for a in ANGLES]
    names: list[str] = []
    for prop in PROPS + ["entropy"]:
        for d in DISTANCES:
            for deg in angle_degs:
                names.append(f"{prop}_d{d}_a{deg}")
    return names


GLCM_FEATURE_NAMES: list[str] = _build_glcm_names()


def color_features(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    import cv2
    from scipy.stats import skew

    n_out = len(COLOR_FEATURE_NAMES)

    def _stats(ch: np.ndarray) -> list[float]:
        sk = 0.0 if ch.std() == 0 else float(skew(ch))
        return [float(ch.mean()), float(ch.std()), sk]

    if img.ndim == 2:
        px = img[mask > 0] if mask is not None else img.ravel()
        if px.size == 0:
            return np.zeros(n_out)
        return np.tile(_stats(px.astype(np.float64)), n_out // 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    stacked = np.concatenate([img, hsv], axis=2)
    px = stacked[mask > 0] if mask is not None else stacked.reshape(-1, 6)
    if px.size == 0:
        return np.zeros(n_out)

    feats: list[float] = []
    for c in range(6):
        feats += _stats(px[:, c].astype(np.float64))
    return np.array(feats)


COLOR_FEATURE_NAMES: list[str] = [
    f"{space}_{ch}_{stat}"
    for space, chans in (("rgb", "rgb"), ("hsv", "hsv"))
    for ch in chans
    for stat in ("mean", "std", "skew")
]


def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, b = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)
    if b.mean() > 127:
        b = 255 - b
    return b.astype(np.uint8)


def shape_features(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    import cv2

    b = mask if mask is not None else binarize(img)
    b = (b > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(len(SHAPE_FEATURE_NAMES))

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    eq_diameter = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h) if w * h > 0 else 0.0
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    if len(cnt) >= 5:  # fitEllipse needs >=5 pts
        (_, (minor, major), _) = cv2.fitEllipse(cnt)
        eccentricity = (minor / major) if major > 0 else 0.0
    else:
        eccentricity = 0.0

    m = cv2.moments(b)
    if m["m00"] > 0:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        pts = cnt.reshape(-1, 2).astype(np.float64)
        radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        max_r, min_r, mean_r = float(radii.max()), float(radii.min()), float(radii.mean())
    else:
        max_r = min_r = mean_r = 0.0

    return np.array([
        area, perimeter, eq_diameter, float(w), float(h), extent,
        compactness, eccentricity, max_r, min_r, mean_r,
    ])


SHAPE_FEATURE_NAMES: list[str] = [
    "area", "perimeter", "eq_diameter", "bbox_w", "bbox_h", "extent",
    "compactness", "eccentricity", "max_radius", "min_radius", "mean_radius",
]


def hu_moments(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    import cv2

    b = mask if mask is not None else binarize(img)
    hu = cv2.HuMoments(cv2.moments((b > 0).astype(np.uint8) * 255)).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def fourier_descriptor(img: np.ndarray, n_coeffs: int = 32) -> np.ndarray:
    import cv2

    b = binarize(img)
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n_coeffs)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    z = cnt[:, 0] + 1j * cnt[:, 1]
    coeffs = np.fft.fft(z)[1:]
    if np.abs(coeffs[0]) > 0:
        coeffs = coeffs / np.abs(coeffs[0])
    fd = np.abs(coeffs[:n_coeffs])
    if len(fd) < n_coeffs:
        fd = np.pad(fd, (0, n_coeffs - len(fd)))
    return fd


HIST_BINS = 16
HIST_FEATURE_NAMES: list[str] = [
    f"hist_{ch}_{b}" for ch in ("h", "s", "v") for b in range(HIST_BINS)
]


def color_histogram(img: np.ndarray, mask: np.ndarray | None = None, bins: int = HIST_BINS) -> np.ndarray:
    import cv2

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    px = hsv[mask > 0] if mask is not None else hsv.reshape(-1, 3)
    if px.size == 0:
        return np.zeros(3 * bins)

    ranges = ((0, 180), (0, 256), (0, 256))
    out: list[np.ndarray] = []
    for c in range(3):
        h, _ = np.histogram(px[:, c], bins=bins, range=ranges[c])
        h = h.astype(float)
        total = h.sum()
        if total > 0:
            h /= total
        out.append(h)
    return np.concatenate(out)


LBP_P = 8
LBP_FEATURE_NAMES: list[str] = [f"lbp_{i}" for i in range(LBP_P + 2)]


def lbp_features(img: np.ndarray, mask: np.ndarray | None = None, P: int = LBP_P, R: float = 1.0) -> np.ndarray:
    import cv2
    from skimage.feature import local_binary_pattern

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    lbp = local_binary_pattern(g, P, R, method="uniform")
    vals = lbp[mask > 0] if mask is not None else lbp.ravel()
    if vals.size == 0:
        return np.zeros(P + 2)
    h, _ = np.histogram(vals, bins=P + 2, range=(0, P + 2))
    h = h.astype(float)
    total = h.sum()
    if total > 0:
        h /= total
    return h


def feature_names(use: tuple[str, ...] = ("glcm", "color", "shape")) -> list[str]:
    families = {
        "glcm": GLCM_FEATURE_NAMES,
        "color": COLOR_FEATURE_NAMES,
        "hist": HIST_FEATURE_NAMES,
        "lbp": LBP_FEATURE_NAMES,
        "shape": SHAPE_FEATURE_NAMES,
    }
    names: list[str] = []
    for key in use:
        if key not in families:
            raise KeyError(f"unknown feature family {key!r}; choose from {list(families)}")
        names.extend(families[key])
    return names


def extract_all(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    use: tuple[str, ...] = ("glcm", "color", "shape"),
) -> np.ndarray:
    extractors = {
        "glcm": lambda: glcm_features(img),
        "color": lambda: color_features(img, mask),
        "hist": lambda: color_histogram(img, mask),
        "lbp": lambda: lbp_features(img, mask),
        "shape": lambda: shape_features(img, mask),
    }
    parts: list[np.ndarray] = []
    for key in use:
        if key not in extractors:
            raise KeyError(f"unknown feature family {key!r}; choose from {list(extractors)}")
        parts.append(np.asarray(extractors[key](), dtype=float))
    return np.concatenate(parts)


def build_feature_matrix(
    images: list[np.ndarray],
    masks: list | None = None,
    use: tuple[str, ...] = ("glcm", "color", "shape"),
) -> tuple[np.ndarray, list[str]]:
    from tqdm import tqdm

    rows: list[np.ndarray] = []
    for i, img in enumerate(tqdm(images, desc="features")):
        mask = masks[i] if masks is not None else None
        rows.append(extract_all(img, mask, use=use))
    X = np.vstack(rows)
    return X, feature_names(use)
