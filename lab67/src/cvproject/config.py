from __future__ import annotations

from pathlib import Path

RNG = 7312
DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets"
DATASETS: dict[str, dict] = {
    "apples_tomatoes": {
        "path": "apples_tomatoes",
        "classes": {
            "apples": "apple",
            "tomatoes": "tomato",
        },
        "glob": "*.jp*g",
        "description": "Fruit: apples vs tomatoes (391 imgs, train/test).",
    },
}
ACTIVE = "apples_tomatoes"


def get_config(name: str | None = None) -> dict:
    if name is None:
        name = ACTIVE
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {list(DATASETS)}")

    cfg = dict(DATASETS[name])
    cfg["name"] = name
    cfg["abs_path"] = DATA_ROOT / cfg["path"]

    if not cfg["abs_path"].exists():
        import warnings

        warnings.warn(f"dataset folder not found: {cfg['abs_path']}", stacklevel=2)
    return cfg
