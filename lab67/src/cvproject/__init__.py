from __future__ import annotations

__version__ = "0.1.0"

from . import config, data, preprocessing, features, classification, metrics, viz
from .config import RNG, DATASETS, ACTIVE, get_config

__all__ = [
    "__version__",
    "RNG",
    "DATASETS",
    "ACTIVE",
    "get_config",
    "config",
    "data",
    "preprocessing",
    "features",
    "classification",
    "metrics",
    "viz",
]
