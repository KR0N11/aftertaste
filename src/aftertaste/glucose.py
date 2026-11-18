"""Measure what a meal did to glucose, from the CGM trace alone.

Deterministic maths, no model involved. Every number here is reproducible
from the raw readings.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from statistics import median
from typing import Sequence

import numpy as np

from .models import GlucoseReading, MealResponse

BASELINE_BEFORE_MIN = 15
BASELINE_AFTER_MIN = 5
WINDOW_MIN = 180
IAUC_MIN = 120
MAX_GAP_MIN = 20
GRID_STEP_MIN = 5


def _minutes(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60


def readings_between(
    readings: Sequence[GlucoseReading], start: datetime, end: datetime
) -> list[GlucoseReading]:
    """Readings with start <= time <= end. `readings` must be sorted by time."""
    times = [r.time for r in readings]
    return list(readings[bisect_left(times, start) : bisect_right(times, end)])


def measure_response(
    readings: Sequence[GlucoseReading], meal_time: datetime
) -> MealResponse:
    """Baseline, peak rise, time to peak, 2 h incremental AUC and rise at 60 min.

    The response is marked invalid (with reasons) instead of guessed when the
    trace has holes: a sensor warm-up or signal loss after a meal would
    otherwise read as "this meal did nothing".
    """
    problems: list[str] = []
    pre = readings_between(
        readings,
        meal_time - timedelta(minutes=BASELINE_BEFORE_MIN),
        meal_time + timedelta(minutes=BASELINE_AFTER_MIN),
    )
    post = readings_between(
        readings, meal_time - timedelta(minutes=BASELINE_BEFORE_MIN),
        meal_time + timedelta(minutes=WINDOW_MIN),
    )
    if not pre:
        return MealResponse(0, 0, 0, 0, 0, valid=False, problems=("no baseline reading",))
    baseline = median(r.mgdl for r in pre)

    t = np.array([_minutes(meal_time, r.time) for r in post])
    v = np.array([r.mgdl for r in post])

    in_window = t[(t >= 0) & (t <= IAUC_MIN)]
    if len(in_window) == 0 or t.max() < IAUC_MIN - GRID_STEP_MIN:
        problems.append("trace ends before 2 h")
    edges = np.concatenate(([max(t.min(), 0.0)], in_window, [min(t.max(), IAUC_MIN)]))
    if len(edges) > 1 and np.diff(np.sort(edges)).max() > MAX_GAP_MIN:
        problems.append(f"gap over {MAX_GAP_MIN} min in the first 2 h")
    if problems:
        return MealResponse(baseline, 0, 0, 0, 0, valid=False, problems=tuple(problems))

    after = t >= 0
    peak_idx = int(np.argmax(np.where(after, v, -np.inf)))
    peak_rise = float(v[peak_idx] - baseline)

    grid = np.arange(0, IAUC_MIN + GRID_STEP_MIN, GRID_STEP_MIN, dtype=float)
    on_grid = np.interp(grid, t, v)
    increments = np.clip(on_grid - baseline, 0, None)
    iauc = float(np.trapezoid(increments, grid))
    rise_60 = float(np.interp(60.0, t, v) - baseline)

    return MealResponse(
        baseline_mgdl=float(baseline),
        peak_rise_mgdl=peak_rise,
        minutes_to_peak=float(t[peak_idx]),
        iauc_2h=iauc,
        rise_at_60_mgdl=rise_60,
    )


def trend_before(
    readings: Sequence[GlucoseReading], at: datetime, minutes: int = 30
) -> float | None:
    """Least-squares slope (mg/dL per min) over the `minutes` before `at`."""
    window = readings_between(readings, at - timedelta(minutes=minutes), at)
    if len(window) < 3:
        return None
    x = np.array([_minutes(at, r.time) for r in window])
    y = np.array([r.mgdl for r in window])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)
