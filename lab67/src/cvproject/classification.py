from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RNG


def build_models(random_state: int = RNG) -> dict[str, Pipeline]:
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


PARAM_GRIDS: dict[str, dict] = {
    "svm": {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
        "clf__kernel": ["rbf", "linear"],
    },
    "decision_tree": {
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
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
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RNG)
    search = GridSearchCV(model, grid, scoring=scoring, cv=skf, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return search


def fit_all(models: dict[str, Pipeline], X_train: np.ndarray, y_train: np.ndarray) -> dict:
    from sklearn.base import clone

    fitted: dict[str, Pipeline] = {}
    for name, model in models.items():
        fitted[name] = clone(model).fit(X_train, y_train)
    return fitted


def predict_all(models: dict[str, Pipeline], X: np.ndarray) -> dict[str, np.ndarray]:
    return {name: model.predict(X) for name, model in models.items()}
