from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> dict:
    """
    Compute the headline metrics for one set of predictions.

    Why both averages: macro exposes poor minority-class performance (the
    interesting failure mode for blight detection); weighted tracks overall
    accuracy. Discuss the gap between them in the report.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
    }


def report(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> str:
    """
    Per-class precision/recall/f1/support text table.
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Confusion matrix (rows = true class, cols = predicted).
    """
    return confusion_matrix(y_true, y_pred)


def compare_models(predictions: dict[str, np.ndarray], y_true: np.ndarray) -> pd.DataFrame:
    """Build a one-row-per-model comparison table.

    Parameters
    ----------
    predictions : ``{model_name: y_pred}``  (e.g. from ``classification.predict_all``)
    """
    rows = {name: evaluate(y_true, y_pred) for name, y_pred in predictions.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.sort_values("f1_macro", ascending=False)
