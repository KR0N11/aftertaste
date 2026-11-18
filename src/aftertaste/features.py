"""Turn (meal, CGM trace, insulin log) into one training row.

The target is the 2 h peak rise in mg/dL. Meals that overlap another meal
within the response window are excluded by default: their responses mix.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo
from typing import Sequence

from .glucose import measure_response, trend_before
from .insulin import insulin_on_board
from .models import GlucoseReading, InsulinDose, InsulinKind, Meal, MealResponse

LOCAL_TZ = ZoneInfo(os.environ.get("AFTERTASTE_TZ", "America/Toronto"))
BOLUS_WINDOW_MIN = 30
OVERLAP_MIN = 120

FEATURE_NAMES = (
    "net_carbs_g",
    "fiber_g",
    "fat_g",
    "protein_g",
    "bolus_units",
    "iob_before_units",
    "baseline_mgdl",
    "trend_mgdl_per_min",
    "hour_sin",
    "hour_cos",
)


@dataclass(frozen=True)
class Row:
    meal: Meal
    response: MealResponse
    features: dict[str, float]

    @property
    def target(self) -> float:
        return self.response.peak_rise_mgdl


def meal_bolus(doses: Sequence[InsulinDose], meal: Meal) -> float:
    """Rapid insulin taken within 30 min either side of the meal."""
    lo = meal.eaten_at - timedelta(minutes=BOLUS_WINDOW_MIN)
    hi = meal.eaten_at + timedelta(minutes=BOLUS_WINDOW_MIN)
    return sum(
        d.units for d in doses if d.kind is InsulinKind.RAPID and lo <= d.time <= hi
    )


def is_isolated(meal: Meal, meals: Sequence[Meal]) -> bool:
    for other in meals:
        if other is meal:
            continue
        gap = abs((other.eaten_at - meal.eaten_at).total_seconds()) / 60
        if gap < OVERLAP_MIN:
            return False
    return True


def build_row(
    meal: Meal,
    readings: Sequence[GlucoseReading],
    doses: Sequence[InsulinDose],
) -> Row | None:
    """One row, or None when the meal has no nutrition or no clean response."""
    if meal.nutrition is None:
        return None
    response = measure_response(readings, meal.eaten_at)
    if not response.valid:
        return None
    n = meal.nutrition
    bolus = meal_bolus(doses, meal)
    before = meal.eaten_at - timedelta(minutes=BOLUS_WINDOW_MIN)
    iob_before = insulin_on_board(doses, before)
    trend = trend_before(readings, meal.eaten_at)
    local = meal.eaten_at.astimezone(LOCAL_TZ)
    local_hour = local.hour + local.minute / 60
    angle = 2 * math.pi * local_hour / 24
    features = {
        "net_carbs_g": n.net_carbs_g,
        "fiber_g": n.fiber_g,
        "fat_g": n.fat_g,
        "protein_g": n.protein_g,
        "bolus_units": bolus,
        "iob_before_units": iob_before,
        "baseline_mgdl": response.baseline_mgdl,
        "trend_mgdl_per_min": trend if trend is not None else 0.0,
        "hour_sin": math.sin(angle),
        "hour_cos": math.cos(angle),
    }
    return Row(meal=meal, response=response, features=features)


def build_dataset(
    meals: Sequence[Meal],
    readings: Sequence[GlucoseReading],
    doses: Sequence[InsulinDose],
    require_isolated: bool = True,
) -> tuple[list[Row], dict[str, int]]:
    """Rows sorted by meal time, plus counts of why meals were skipped."""
    readings = sorted(readings, key=lambda r: r.time)
    skipped = {"no_nutrition": 0, "overlapping_meal": 0, "bad_trace": 0}
    rows: list[Row] = []
    for meal in sorted(meals, key=lambda m: m.eaten_at):
        if meal.nutrition is None:
            skipped["no_nutrition"] += 1
            continue
        if require_isolated and not is_isolated(meal, meals):
            skipped["overlapping_meal"] += 1
            continue
        row = build_row(meal, readings, doses)
        if row is None:
            skipped["bad_trace"] += 1
            continue
        rows.append(row)
    return rows, skipped
