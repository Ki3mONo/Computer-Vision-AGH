"""Display helpers, in the style of the previous labs.

All plotting is inline matplotlib: single-channel images use ``cmap='gray'``,
image axes are turned off, and every figure ends with ``tight_layout()``.
Functions return the created ``Figure`` so the caller can save it if needed.

Unlike the rest of the package these helpers are fully implemented — they are
the reusable "boilerplate" requested for the project.
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt


def _is_gray(img: np.ndarray) -> bool:
    """True for 2-D (single channel) images."""
    return img.ndim == 2


def _auto_cmap(img: np.ndarray, cmap: str | None) -> str | None:
    """Pick a cmap: explicit value wins, else 'gray' for single-channel."""
    if cmap is not None:
        return cmap
    return "gray" if _is_gray(img) else None


def show_image(img, title=None, ax=None, cmap=None):
    """Display a single image. Returns the Figure.

    Parameters
    ----------
    img : ndarray (H, W) or (H, W, 3)
    title : optional str
    ax : optional existing Axes to draw into
    cmap : optional colormap override (defaults to gray for single-channel)
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.imshow(img, cmap=_auto_cmap(img, cmap))
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")

    if created:
        fig.tight_layout()
    return fig


def show_grid(images, titles=None, ncols=6, cmap=None, suptitle=None, figsize=None):
    """Display a list of images in a grid. Returns the Figure.

    Empty trailing cells are hidden. ``titles`` (if given) must align with
    ``images``.
    """
    n = len(images)
    if n == 0:
        raise ValueError("show_grid: no images to display")

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (2.5 * ncols, 2.7 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], cmap=_auto_cmap(images[i], cmap))
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
        ax.axis("off")  # also hides the empty trailing axes

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


def show_class_samples(images, labels, n_per_class=6, cmap=None, suptitle=None):
    """One row per class, ``n_per_class`` evenly-sampled images per row.

    Mirrors the lab4 dataset-preview grid. ``images`` and ``labels`` are parallel
    sequences (labels are class names or ids). Returns the Figure.
    """
    labels = np.asarray(labels)
    classes = sorted(np.unique(labels).tolist(), key=str)
    nrows = len(classes)

    fig, axes = plt.subplots(
        nrows, n_per_class,
        figsize=(2.3 * n_per_class, 2.4 * nrows),
        squeeze=False,
    )

    for r, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        # evenly spaced samples across the class
        step = max(1, len(idx) // n_per_class)
        picks = idx[::step][:n_per_class]
        for col in range(n_per_class):
            ax = axes[r][col]
            if col < len(picks):
                ax.imshow(images[picks[col]], cmap=_auto_cmap(images[picks[col]], cmap))
            ax.axis("off")
        axes[r][0].set_title(str(c), fontsize=11, loc="left")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def show_pair(before, after, titles=("Original", "Processed"), cmap=None, suptitle=None):
    """Side-by-side comparison of two images (e.g. raw vs preprocessed).

    Returns the Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, img, title in zip(axes, (before, after), titles):
        ax.imshow(img, cmap=_auto_cmap(img, cmap))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def show_mask_overlay(img, mask, alpha=0.4, color=(255, 0, 0), title=None, ax=None):
    """Overlay a binary ``mask`` (tinted ``color``) on ``img``. Returns the Figure.

    ``mask`` may be {0,255} or boolean. Works for gray or RGB ``img``.
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    # Build an RGB base so we can blend a coloured overlay.
    base = img if img.ndim == 3 else np.stack([img] * 3, axis=-1)
    base = base.astype(np.float32)

    m = mask.astype(bool)
    overlay = base.copy()
    overlay[m] = (1 - alpha) * base[m] + alpha * np.array(color, dtype=np.float32)

    ax.imshow(overlay.clip(0, 255).astype(np.uint8))
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")

    if created:
        fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names, ax=None, normalize=False, title=None, cmap="Blues"):
    """Plot a confusion matrix (wraps ``ConfusionMatrixDisplay``). Returns the Figure.

    ``cm`` is a precomputed matrix (see :func:`cvproject.metrics.confusion`).
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = np.asarray(cm)
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        values_format = ".2f"
    else:
        # keep integer counts so the "d" format code is valid (cm may arrive as float)
        cm = cm.astype(int)
        values_format = "d"

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.figure

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(
        ax=ax, cmap=cmap, colorbar=False, xticks_rotation=45,
        values_format=values_format,
    )
    ax.set_title(title or ("Confusion matrix" + (" (normalized)" if normalize else "")), fontsize=11)

    if created:
        fig.tight_layout()
    return fig


def plot_feature_distributions(X, y, names, feature_idx=None, max_features=6, suptitle=None):
    """Per-class boxplots for selected features (lab4 feature-boxplot pattern).

    Parameters
    ----------
    X : (n_samples, n_features) array
    y : (n_samples,) class labels
    names : feature names aligned with X columns
    feature_idx : optional iterable of column indices to plot
                  (defaults to the first ``max_features``)

    Returns the Figure.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if feature_idx is None:
        feature_idx = list(range(min(max_features, X.shape[1])))

    classes = sorted(np.unique(y).tolist(), key=str)
    n = len(feature_idx)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax_i, fi in enumerate(feature_idx):
        ax = axes[ax_i]
        data = [X[y == c, fi] for c in classes]
        # set tick labels manually: matplotlib renamed boxplot's `labels` kwarg to
        # `tick_labels` in 3.9, so neither name is portable across versions.
        ax.boxplot(data)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels([str(c) for c in classes])
        ax.set_title(names[fi] if fi < len(names) else f"feat {fi}", fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    for ax in axes[n:]:  # hide unused panels
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig
