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


# --------------------------------------------------------------------------- #
# Colour: per-channel statistics over RGB + HSV + Lab, plus a hue histogram
# --------------------------------------------------------------------------- #
_COLOR_STATS = ("mean", "std", "skew", "median", "kurt", "p25", "p75")
COLOR_FEATURE_NAMES: list[str] = [
    f"{space}_{ch}_{stat}"
    for space in ("rgb", "hsv", "lab")
    for ch in space
    for stat in _COLOR_STATS
]


def color_features(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    import cv2
    from scipy.stats import skew, kurtosis

    n_out = len(COLOR_FEATURE_NAMES)
    n_stats = len(_COLOR_STATS)

    def _stats(ch: np.ndarray) -> list[float]:
        ch = ch.astype(np.float64)
        if ch.std() == 0:
            sk = kt = 0.0  # undefined for a uniform channel
        else:
            sk = float(skew(ch))
            kt = float(kurtosis(ch))
        p25, p75 = np.percentile(ch, [25, 75])
        return [float(ch.mean()), float(ch.std()), sk,
                float(np.median(ch)), kt, float(p25), float(p75)]

    if img.ndim == 2:
        px = img[mask > 0] if mask is not None else img.ravel()
        if px.size == 0:
            return np.zeros(n_out)
        return np.tile(_stats(px), n_out // n_stats)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    stacked = np.concatenate([img, hsv, lab], axis=2)  # 9 channels
    px = stacked[mask > 0] if mask is not None else stacked.reshape(-1, 9)
    if px.size == 0:
        return np.zeros(n_out)

    feats: list[float] = []
    for c in range(9):
        feats += _stats(px[:, c])
    return np.array(feats)


HUE_BINS = 12
HUE_FEATURE_NAMES: list[str] = [f"hue_bin_{i:02d}" for i in range(HUE_BINS)]


def hue_histogram(img: np.ndarray, mask: np.ndarray | None = None, bins: int = HUE_BINS) -> np.ndarray:
    import cv2

    if img.ndim == 2:
        return np.zeros(bins)
    h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 0]
    vals = h[mask > 0] if mask is not None else h.ravel()
    if vals.size == 0:
        return np.zeros(bins)
    hist, _ = np.histogram(vals, bins=bins, range=(0, 180))
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


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
    hist, _ = np.histogram(vals, bins=P + 2, range=(0, P + 2))
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def binarize(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, b = cv2.threshold(g, threshold, 255, cv2.THRESH_BINARY)
    if b.mean() > 127:
        b = 255 - b
    return b.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Shape: contour geometry, Hu moments, Fourier descriptors, fractal dimension,
# convex-hull metrics, radial signature, Zernike moments
# --------------------------------------------------------------------------- #
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


HU_FEATURE_NAMES: list[str] = [f"hu_{i}" for i in range(7)]


def hu_moments(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    import cv2

    b = mask if mask is not None else binarize(img)
    hu = cv2.HuMoments(cv2.moments((b > 0).astype(np.uint8) * 255)).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


FOURIER_COEFFS = 15
FOURIER_FEATURE_NAMES: list[str] = [f"fd_{i:02d}" for i in range(FOURIER_COEFFS)]


def fourier_descriptor(img: np.ndarray, mask: np.ndarray | None = None, n_coeffs: int = FOURIER_COEFFS) -> np.ndarray:
    import cv2

    b = mask if mask is not None else binarize(img)
    b = (b > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n_coeffs)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    z = cnt[:, 0] + 1j * cnt[:, 1]
    coeffs = np.fft.fft(z)[1:]  # drop DC -> translation invariance
    if np.abs(coeffs[0]) > 0:
        coeffs = coeffs / np.abs(coeffs[0])  # scale/rotation normalisation
    fd = np.abs(coeffs[:n_coeffs])
    if len(fd) < n_coeffs:
        fd = np.pad(fd, (0, n_coeffs - len(fd)))
    return fd


def fractal_dimension(mask: np.ndarray) -> float:
    """Box-counting dimension of the object boundary (edge raggedness)."""
    import cv2

    b = (mask > 0).astype(np.uint8)
    if b.sum() == 0:
        return 0.0
    edges = cv2.Canny(b * 255, 50, 150) > 0
    Z = edges if edges.any() else (b > 0)
    p = min(Z.shape)
    if p < 2:
        return 0.0

    sizes, counts = [], []
    s = 2 ** int(np.floor(np.log2(p)))
    while s >= 2:
        ny, nx = Z.shape[0] // s, Z.shape[1] // s
        if ny and nx:
            blocks = Z[:ny * s, :nx * s].reshape(ny, s, nx, s)
            c = int(np.count_nonzero(blocks.any(axis=(1, 3))))
            if c > 0:
                sizes.append(s)
                counts.append(c)
        s //= 2
    if len(sizes) < 2:
        return 0.0
    slope = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(np.array(counts)), 1)[0]
    return float(slope)


GEOM_FEATURE_NAMES: list[str] = ["fractal_dimension", "solidity", "convexity", "aspect_ratio"]


def geom_features(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    import cv2

    b = mask if mask is not None else binarize(img)
    b = (b > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(len(GEOM_FEATURE_NAMES))

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perim = float(cv2.arcLength(hull, True))
    solidity = area / hull_area if hull_area > 0 else 0.0
    convexity = hull_perim / perim if perim > 0 else 0.0
    _, _, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h > 0 else 0.0
    return np.array([fractal_dimension(b), solidity, convexity, aspect])


SIGNATURE_SAMPLES = 64
SIGNATURE_FEATURE_NAMES: list[str] = [f"sig_{i:02d}" for i in range(SIGNATURE_SAMPLES)]


def shape_signature(img: np.ndarray, mask: np.ndarray | None = None, n_samples: int = SIGNATURE_SAMPLES) -> np.ndarray:
    import cv2

    b = mask if mask is not None else binarize(img)
    b = (b > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(n_samples)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    m = cv2.moments(b)
    if m["m00"] == 0:
        return np.zeros(n_samples)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    ang = np.arctan2(cnt[:, 1] - cy, cnt[:, 0] - cx)
    rad = np.sqrt((cnt[:, 0] - cx) ** 2 + (cnt[:, 1] - cy) ** 2)
    order = np.argsort(ang)
    grid = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
    sig = np.interp(grid, ang[order], rad[order], period=2 * np.pi)
    mean_r = sig.mean()
    if mean_r > 0:
        sig = sig / mean_r  # scale invariance
    return sig


def _zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    from math import factorial

    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * factorial(n - k)) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
        R = R + c * rho ** (n - 2 * k)
    return R


def _zernike_orders(degree: int) -> list[tuple[int, int]]:
    return [(n, m) for n in range(degree + 1) for m in range(n + 1) if (n - m) % 2 == 0]


ZERNIKE_DEGREE = 8
ZERNIKE_ORDERS = _zernike_orders(ZERNIKE_DEGREE)
ZERNIKE_FEATURE_NAMES: list[str] = [f"zernike_n{n}_m{m}_abs" for (n, m) in ZERNIKE_ORDERS]


def zernike_features(img: np.ndarray, mask: np.ndarray | None = None, degree: int = ZERNIKE_DEGREE) -> np.ndarray:
    import cv2

    orders = _zernike_orders(degree)
    b = mask if mask is not None else binarize(img)
    b = (b > 0).astype(np.uint8)
    if b.sum() == 0:
        return np.zeros(len(orders))

    ys, xs = np.where(b)
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    N = 64
    b = cv2.resize(b, (N, N), interpolation=cv2.INTER_NEAREST).astype(np.float64)

    g = (np.arange(N) - (N - 1) / 2) / ((N - 1) / 2)
    X, Y = np.meshgrid(g, g)
    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)
    inside = rho <= 1.0
    mass = b[inside].sum()
    if mass == 0:
        return np.zeros(len(orders))

    feats = []
    for (n, m) in orders:
        V = _zernike_radial(n, m, rho) * np.exp(-1j * m * theta)
        z = (n + 1) / np.pi * np.sum(b[inside] * V[inside]) / mass
        feats.append(abs(z))
    return np.array(feats, dtype=float)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_FAMILIES = {
    "glcm": GLCM_FEATURE_NAMES,
    "color": COLOR_FEATURE_NAMES,
    "huehist": HUE_FEATURE_NAMES,
    "lbp": LBP_FEATURE_NAMES,
    "shape": SHAPE_FEATURE_NAMES,
    "hu": HU_FEATURE_NAMES,
    "fourier": FOURIER_FEATURE_NAMES,
    "geom": GEOM_FEATURE_NAMES,
    "zernike": ZERNIKE_FEATURE_NAMES,
    "signature": SIGNATURE_FEATURE_NAMES,
}

DEFAULT_USE = ("glcm", "color", "huehist", "lbp", "shape", "hu", "fourier", "geom", "zernike")


def feature_names(use: tuple[str, ...] = DEFAULT_USE) -> list[str]:
    names: list[str] = []
    for key in use:
        if key not in _FAMILIES:
            raise KeyError(f"unknown feature family {key!r}; choose from {list(_FAMILIES)}")
        names.extend(_FAMILIES[key])
    return names


def extract_all(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    use: tuple[str, ...] = DEFAULT_USE,
) -> np.ndarray:
    extractors = {
        "glcm": lambda: glcm_features(img),
        "color": lambda: color_features(img, mask),
        "huehist": lambda: hue_histogram(img, mask),
        "lbp": lambda: lbp_features(img, mask),
        "shape": lambda: shape_features(img, mask),
        "hu": lambda: hu_moments(img, mask),
        "fourier": lambda: fourier_descriptor(img, mask),
        "geom": lambda: geom_features(img, mask),
        "zernike": lambda: zernike_features(img, mask),
        "signature": lambda: shape_signature(img, mask),
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
    use: tuple[str, ...] = DEFAULT_USE,
) -> tuple[np.ndarray, list[str]]:
    from tqdm import tqdm

    rows: list[np.ndarray] = []
    for i, img in enumerate(tqdm(images, desc="features")):
        mask = masks[i] if masks is not None else None
        rows.append(extract_all(img, mask, use=use))
    X = np.vstack(rows)
    return X, feature_names(use)
