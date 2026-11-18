"""Insulin on board (IOB) for rapid-acting insulin.

Uses the exponential insulin action curve popularised by the open-source
Loop project: a curve defined by the time of peak activity and the total
duration of insulin action (DIA). Defaults (peak 75 min, DIA 6 h) are the
common adult rapid-acting settings, not personal ones.

This is used only as a *feature* that explains glucose responses. It never
produces a dose and must not be used to choose one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .models import InsulinDose, InsulinKind


@dataclass(frozen=True)
class ExponentialInsulinModel:
    peak_minutes: float = 75.0
    duration_minutes: float = 360.0

    def __post_init__(self) -> None:
        if not 0 < self.peak_minutes < self.duration_minutes / 2:
            raise ValueError("peak must be positive and under half of the duration")

    @property
    def _params(self) -> tuple[float, float, float]:
        tp, td = self.peak_minutes, self.duration_minutes
        tau = tp * (1 - tp / td) / (1 - 2 * tp / td)
        a = 2 * tau / td
        s = 1 / (1 - a + (1 + a) * math.exp(-td / tau))
        return tau, a, s

    def fraction_remaining(self, minutes_since_dose: float) -> float:
        """Share of a dose still to act, 1.0 at the dose and 0.0 at DIA."""
        t = minutes_since_dose
        if t <= 0:
            return 1.0
        if t >= self.duration_minutes:
            return 0.0
        tau, a, s = self._params
        td = self.duration_minutes
        value = 1 - s * (1 - a) * (
            (t * t / (tau * td * (1 - a)) - t / tau - 1) * math.exp(-t / tau) + 1
        )
        return min(max(value, 0.0), 1.0)


DEFAULT_MODEL = ExponentialInsulinModel()


def insulin_on_board(
    doses: Iterable[InsulinDose],
    at: datetime,
    model: ExponentialInsulinModel = DEFAULT_MODEL,
) -> float:
    """Units of rapid-acting insulin still active at `at`.

    Doses after `at` are ignored (they have not happened yet). Long-acting
    (basal) insulin is excluded: it is a flat background, not a meal effect.
    """
    total = 0.0
    for dose in doses:
        if dose.kind is not InsulinKind.RAPID:
            continue
        minutes = (at - dose.time).total_seconds() / 60
        if minutes < 0:
            continue
        total += dose.units * model.fraction_remaining(minutes)
    return total
