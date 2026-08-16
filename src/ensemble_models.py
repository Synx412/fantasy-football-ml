from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.pipeline import Pipeline


def _fit_estimator(model: Any, X, y, sample_weight=None):
    if sample_weight is None:
        model.fit(X, y)
        return model
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=sample_weight)
    else:
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            model.fit(X, y)
    return model


@dataclass
class BlendedClassifier:
    models: list[Any]
    weights: list[float]

    def __post_init__(self) -> None:
        if len(self.models) != len(self.weights) or not self.models:
            raise ValueError("models and weights must have the same non-zero length")
        weights = np.asarray(self.weights, dtype=float)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("weights must be non-negative and sum to > 0")
        self.weights = (weights / weights.sum()).tolist()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        for model in self.models:
            _fit_estimator(model, X, y, sample_weight)
        classes = getattr(self.models[0], "classes_", None)
        if classes is None and isinstance(self.models[0], Pipeline):
            classes = getattr(self.models[0].named_steps.get("model"), "classes_", None)
        if classes is not None:
            self.classes_ = np.asarray(classes)
        return self

    def predict_proba(self, X) -> np.ndarray:
        total = None
        for weight, model in zip(self.weights, self.models):
            values = np.asarray(model.predict_proba(X), dtype=float)
            total = weight * values if total is None else total + weight * values
        return np.asarray(total, dtype=float)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        if probabilities.shape[1] == 1:
            return np.zeros(len(probabilities), dtype=int)
        return (probabilities[:, 1] >= 0.5).astype(int)


@dataclass
class BlendedRegressor:
    models: list[Any]
    weights: list[float]

    def __post_init__(self) -> None:
        if len(self.models) != len(self.weights) or not self.models:
            raise ValueError("models and weights must have the same non-zero length")
        weights = np.asarray(self.weights, dtype=float)
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("weights must be non-negative and sum to > 0")
        self.weights = (weights / weights.sum()).tolist()

    def fit(self, X, y, sample_weight=None):
        for model in self.models:
            _fit_estimator(model, X, y, sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        total = None
        for weight, model in zip(self.weights, self.models):
            values = np.asarray(model.predict(X), dtype=float)
            total = weight * values if total is None else total + weight * values
        return np.asarray(total, dtype=float)


def set_xgboost_device(obj: Any, device: str = "cpu") -> None:
    """Recursively switch fitted XGBoost estimators to the desired inference device."""
    if isinstance(obj, (BlendedClassifier, BlendedRegressor)):
        for model in obj.models:
            set_xgboost_device(model, device)
        return
    if isinstance(obj, Pipeline):
        model = obj.named_steps.get("model")
        if model is not None and model.__class__.__module__.startswith("xgboost"):
            model.set_params(device=device)
