from __future__ import annotations

from pathlib import Path

RNG = 7312
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets"
DATASETS: dict[str, dict] = {
    "potato_simple": {
        "path": "Task 1 - Simple",
        "classes": {
            "Potato___Early_blight": "early_blight",
            "Potato___healthy": "healthy",
        },
        "presplit": False,
        "color": True,
        "glob": "*.jpg",
        "description": "Potato leaves: early blight vs healthy (~305 imgs, easy).",
    },
    "potato_hard": {
        "path": "Task 2 - Hard",
        "classes": {
            "Potato___Early_blight": "early_blight",
            "Potato___Late_blight": "late_blight",
        },
        "presplit": False,
        "color": True,
        "glob": "*.jpg",
        "description": "Potato leaves: early vs late blight (2000 imgs, hard).",
    },
    "apples_tomatoes": {
        "path": "apples_tomatoes",
        "classes": {
            "apples": "apple",
            "tomatoes": "tomato",
        },
        "presplit": True,
        "color": True,
        "glob": "*.jp*g",
        "description": "Fruit: apples vs tomatoes (391 imgs, pre-split train/test).",
    },
}
ACTIVE = "potato_simple"


def get_config(name: str | None = None) -> dict:
    """
    Return a resolved, self-contained copy of a dataset config.

    The returned dict is what every other module consumes, so it must be fully
    resolved (absolute path included) and safe to mutate by the caller.
    
    Returns
    -------
    dict
        The registry entry plus ``"name"`` and ``"abs_path"`` (absolute ``Path``).
    """
    if name is None:
        name = ACTIVE
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")

    cfg = dict(DATASETS[name])
    cfg["name"] = name
    cfg["abs_path"] = DATA_ROOT / cfg["path"]

    if not cfg["abs_path"].exists():
        import warnings

        warnings.warn(
            f"dataset folder not found: {cfg['abs_path']} "
            "(wrong DATA_ROOT or data not downloaded yet?)",
            stacklevel=2,
        )
    return cfg
