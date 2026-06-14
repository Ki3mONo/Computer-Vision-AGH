from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RNG


def build_models(random_state: int = RNG) -> dict[str, Pipeline]:
    """
    Return a name -> ``Pipeline`` dict of candidate classifiers.

    Each value pairs a ``StandardScaler`` with one classical estimator. Scaling
    matters: SVM and kNN are distance-based and the feature families live on very
    different magnitudes (GLCM in [0,1] vs shape ``area`` in the thousands), so an
    unscaled SVM would be dominated by the large-valued columns. Wrapping the
    scaler INSIDE the pipeline means it is re-fit on each CV training fold, so no
    information leaks from validation into the scaling. SVM and the decision tree
    are the two named in the TASK; the rest are allowed baselines.
    """
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    estimators = {
        "svm": SVC(random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "knn": KNeighborsClassifier(),
        "logreg": LogisticRegression(max_iter=2000, random_state=random_state),
    }
    return {
        name: Pipeline([("scaler", StandardScaler()), ("clf", est)])
        for name, est in estimators.items()
    }


# Hyper-parameter search spaces for GridSearchCV (the TASK's "select parameters
# to maximise results on train/validation"). Keys are prefixed ``clf__`` so they
# target the classifier step inside the pipeline (``step__param`` convention).
# Starting points — widen/refine around the winner. Total fits == product of list
# lengths x cv folds, so keep each grid modest.
PARAM_GRIDS: dict[str, dict] = {
    "svm": {
        "clf__C": [0.1, 1, 10, 100],            # regularisation: low = smoother boundary
        "clf__gamma": ["scale", 0.01, 0.001],   # rbf kernel width (ignored for linear)
        "clf__kernel": ["rbf", "linear"],
    },
    "decision_tree": {
        "clf__max_depth": [None, 5, 10, 20],     # cap depth to fight overfitting
        "clf__min_samples_leaf": [1, 2, 5],      # larger -> simpler tree
        "clf__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__max_features": ["sqrt", "log2"],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"],
    },
    "logreg": {
        "clf__C": [0.1, 1, 10],
    },
}


def tune(
    model: Pipeline,
    grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "f1_macro",
) -> "object":
    """
    Grid-search a single pipeline's hyper-parameters with cross-validation.

    Returns the fitted ``GridSearchCV``; read ``.best_estimator_`` (re-trained on
    the full training set via ``refit=True``), ``.best_params_`` and
    ``.best_score_``. Stratified folds keep class ratios balanced; the default
    ``scoring="f1_macro"`` optimises balanced performance, not raw accuracy. An
    empty ``grid`` simply CV-scores the default model.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RNG)
    search = GridSearchCV(model, grid, scoring=scoring, cv=skf, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search


def fit_all(models: dict[str, Pipeline], X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Fit every pipeline on the training set (no tuning — defaults / pre-tuned).

    Use this for a quick baseline pass; use :func:`tune` per model when selecting
    hyper-parameters.
    """
    from sklearn.base import clone

    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        fitted[name] = clone(model).fit(X_train, y_train)
    return fitted


def predict_all(models: dict[str, Pipeline], X: np.ndarray) -> dict[str, np.ndarray]:
    """
    Predict with every fitted model.

    Call on ``X_test`` and feed the result to
    :func:`cvproject.metrics.compare_models` for the final comparison table.
    Pre-condition: every ``model`` must already be fitted (from :func:`tune` or
    :func:`fit_all`).
    """
    return {name: model.predict(X) for name, model in models.items()}
