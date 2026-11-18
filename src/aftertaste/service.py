"""Use cases shared by the CLI and the HTTP API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .dexcom import DexcomClient
from .features import build_dataset, build_row
from .glucose import measure_response
from .models import Meal, utc
from .nutrition import MealParser
from .predict import EvalResult, ResponseModel, evaluate
from .store import Store

FEATURE_LABELS = {
    "net_carbs_g": "net carbs",
    "fiber_g": "fibre",
    "fat_g": "fat",
    "protein_g": "protein",
    "bolus_units": "meal bolus",
    "iob_before_units": "insulin already on board",
    "baseline_mgdl": "starting glucose",
    "trend_mgdl_per_min": "glucose trend before eating",
    "hour_sin": "time of day (sin)",
    "hour_cos": "time of day (cos)",
}


def log_meal(
    store: Store,
    parser: MealParser,
    eaten_at: datetime,
    text: str | None = None,
    photo: str | Path | None = None,
) -> Meal:
    nutrition = parser.parse(text=text, photo=photo)
    meal = Meal(
        eaten_at=utc(eaten_at),
        description=text or (Path(photo).name if photo else ""),
        nutrition=nutrition,
        photo_path=str(photo) if photo else None,
    )
    store.add_meal(meal)
    return meal


def meal_report(store: Store, meal: Meal) -> dict:
    """Measured response and, when a model can be trained, the expected one."""
    readings = store.readings(meal.eaten_at - timedelta(hours=1), meal.eaten_at + timedelta(hours=4))
    response = measure_response(readings, meal.eaten_at)
    out = {
        "id": meal.id,
        "eaten_at": meal.eaten_at.isoformat(),
        "description": meal.description,
        "nutrition": None if meal.nutrition is None else {
            "carbs_g": meal.nutrition.carbs_g,
            "net_carbs_g": meal.nutrition.net_carbs_g,
            "fat_g": meal.nutrition.fat_g,
            "protein_g": meal.nutrition.protein_g,
            "fiber_g": meal.nutrition.fiber_g,
            "confidence": meal.nutrition.confidence,
            "items": list(meal.nutrition.items),
        },
        "measured": None if not response.valid else {
            "peak_rise_mgdl": round(response.peak_rise_mgdl),
            "minutes_to_peak": round(response.minutes_to_peak),
            "iauc_2h": round(response.iauc_2h),
        },
        "problems": list(response.problems),
        "expected_peak_rise_mgdl": None,
    }
    model = _trained_model(store, exclude_id=meal.id)
    if model is not None:
        row = build_row(meal, store.readings(), store.doses())
        if row is not None:
            out["expected_peak_rise_mgdl"] = round(model.predict_features(row.features))
    return out


def _trained_model(store: Store, exclude_id: int | None = None) -> ResponseModel | None:
    meals = [m for m in store.meals() if m.id != exclude_id]
    rows, _ = build_dataset(meals, store.readings(), store.doses())
    if len(rows) < 20:
        return None
    return ResponseModel().fit(rows)


def insights(store: Store) -> dict:
    rows, skipped = build_dataset(store.meals(), store.readings(), store.doses())
    out: dict = {"clean_meals": len(rows), "skipped": skipped, "evaluation": None, "effects": None}
    try:
        result: EvalResult = evaluate(rows)
        out["evaluation"] = result.as_dict()
    except ValueError as e:
        out["evaluation_error"] = str(e)
        return out
    model = ResponseModel().fit(rows)
    out["effects"] = sorted(
        (
            {"feature": FEATURE_LABELS[k], "mgdl_per_sd": round(v, 1)}
            for k, v in model.coefficients().items()
        ),
        key=lambda e: -abs(e["mgdl_per_sd"]),
    )
    return out


def sync_dexcom(store: Store, client: DexcomClient, days: int = 30) -> dict:
    """Pull glucose and insulin since the last reading (or `days` back)."""
    end = datetime.now(timezone.utc)
    latest = store.latest_reading_time()
    start = latest if latest and end - latest < timedelta(days=days) else end - timedelta(days=days)
    readings = client.glucose(start, end)
    doses = client.insulin(start, end)
    store.add_readings(readings)
    store.add_doses(doses)
    store.set_meta("last_sync", end.isoformat())
    return {"readings": len(readings), "insulin_doses": len(doses), "from": start.isoformat()}
