"""Predict a meal's 2 h peak glucose rise and measure it honestly.

Evaluation is chronological: train on earlier meals, test on later ones,
never shuffled (a shuffled split leaks next week's habits into training).
The headline is the improvement over a carbs-only baseline, with a paired
bootstrap interval because one person's meal count is small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, Row

MIN_TRAIN = 20
MIN_TEST = 8


def _matrix(rows: Sequence[Row], names: Sequence[str]) -> np.ndarray:
    return np.array([[r.features[n] for n in names] for r in rows], dtype=float)


def _targets(rows: Sequence[Row]) -> np.ndarray:
    return np.array([r.target for r in rows], dtype=float)


class CarbOnlyBaseline:
    """Peak rise as a straight line in net carbs. What a carb counter assumes."""

    names = ("net_carbs_g",)

    def fit(self, rows: Sequence[Row]) -> "CarbOnlyBaseline":
        self._m = LinearRegression().fit(_matrix(rows, self.names), _targets(rows))
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        return self._m.predict(_matrix(rows, self.names))


class ResponseModel:
    """Ridge regression on all features. Linear on purpose: with a few hundred
    meals, a model you can read the coefficients of beats one you cannot."""

    names = FEATURE_NAMES

    def __init__(self, alpha: float = 3.0) -> None:
        self.alpha = alpha

    def fit(self, rows: Sequence[Row]) -> "ResponseModel":
        self._m = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._m.fit(_matrix(rows, self.names), _targets(rows))
        return self

    def predict(self, rows: Sequence[Row]) -> np.ndarray:
        return self._m.predict(_matrix(rows, self.names))

    def predict_features(self, features: dict[str, float]) -> float:
        x = np.array([[features[n] for n in self.names]], dtype=float)
        return float(self._m.predict(x)[0])

    def coefficients(self) -> dict[str, float]:
        """Effect of +1 standard deviation of each feature, in mg/dL."""
        ridge = self._m.named_steps["ridge"]
        return dict(zip(self.names, (float(c) for c in ridge.coef_)))


@dataclass(frozen=True)
class EvalResult:
    n_train: int
    n_test: int
    baseline_mae: float
    model_mae: float
    improvement_pct: float
    improvement_ci95: tuple[float, float]
    test_start: str

    def as_dict(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "baseline_mae_mgdl": round(self.baseline_mae, 1),
            "model_mae_mgdl": round(self.model_mae, 1),
            "improvement_pct": round(self.improvement_pct, 1),
            "improvement_ci95_pct": [round(x, 1) for x in self.improvement_ci95],
            "test_start": self.test_start,
        }


def chronological_split(
    rows: Sequence[Row], test_fraction: float = 0.25
) -> tuple[list[Row], list[Row]]:
    ordered = sorted(rows, key=lambda r: r.meal.eaten_at)
    cut = int(round(len(ordered) * (1 - test_fraction)))
    return ordered[:cut], ordered[cut:]


def _bootstrap_improvement(
    base_err: np.ndarray, model_err: np.ndarray, seed: int = 0, n: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    k = len(base_err)
    samples = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        b = base_err[idx].mean()
        samples.append(0.0 if b == 0 else 100 * (b - model_err[idx].mean()) / b)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def grouped_chronological_split(
    groups: dict[str, Sequence[Row]], test_fraction: float = 0.25
) -> tuple[list[Row], list[Row]]:
    """For many people: each person's later meals are test, earlier are train.
    No person's future meals are ever used to train on their past."""
    train: list[Row] = []
    test: list[Row] = []
    for rows in groups.values():
        a, b = chronological_split(rows, test_fraction)
        train += a
        test += b
    return train, test


def evaluate(rows: Sequence[Row], test_fraction: float = 0.25) -> EvalResult:
    train, test = chronological_split(rows, test_fraction)
    return _score(train, test)


def evaluate_grouped(groups: dict[str, Sequence[Row]], test_fraction: float = 0.25) -> EvalResult:
    train, test = grouped_chronological_split(groups, test_fraction)
    return _score(train, sorted(test, key=lambda r: r.meal.eaten_at))


def _score(train: Sequence[Row], test: Sequence[Row]) -> EvalResult:
    if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
        raise ValueError(
            f"not enough clean meals: {len(train)} train / {len(test)} test, "
            f"need at least {MIN_TRAIN} / {MIN_TEST}. Keep logging."
        )
    y = _targets(test)
    base_err = np.abs(CarbOnlyBaseline().fit(train).predict(test) - y)
    model_err = np.abs(ResponseModel().fit(train).predict(test) - y)
    b, m = float(base_err.mean()), float(model_err.mean())
    return EvalResult(
        n_train=len(train),
        n_test=len(test),
        baseline_mae=b,
        model_mae=m,
        improvement_pct=0.0 if b == 0 else 100 * (b - m) / b,
        improvement_ci95=_bootstrap_improvement(base_err, model_err),
        test_start=test[0].meal.eaten_at.isoformat(),
    )
