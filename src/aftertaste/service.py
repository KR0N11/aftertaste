"""Use cases shared by the CLI and the HTTP API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .dexcom import DexcomClient
from .features import build_dataset, build_row
from .glucose import measure_response
from .models import Meal, utc
from .nutrition import MealParser, ParseError
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


def import_photos(store: Store, parser: MealParser, folder: str | Path, tz: str = "America/Toronto") -> dict:
    """Log every new photo in `folder` as a meal at the time it was taken.
    A photo the model cannot read stays un-imported and is retried next run."""
    from .photos import find_new, to_model_jpeg

    imported, failed = [], []
    for photo in find_new(folder, store.imported_hashes("photo"), tz):
        jpeg = to_model_jpeg(photo, Path("data/photos"))
        try:
            meal = log_meal(store, parser, photo.taken_at, photo=jpeg)
        except ParseError as e:
            failed.append({"file": photo.source.name, "error": str(e)})
            continue
        store.mark_imported(photo.sha256, "photo", meal.id)
        imported.append(
            {"file": photo.source.name, "eaten_at": meal.eaten_at.isoformat(),
             "time_from": photo.time_source, "carbs_g": meal.nutrition.carbs_g,
             "confidence": meal.nutrition.confidence}
        )
    return {"imported": imported, "failed": failed}


def find_clarity_exports(folder: str | Path) -> list[Path]:
    """CSV files in `folder` whose header looks like a Clarity export."""
    found = []
    for path in sorted(Path(folder).glob("*.csv"), key=lambda p: p.stat().st_mtime):
        try:
            with open(path, encoding="utf-8-sig") as f:
                header = f.readline()
        except (OSError, UnicodeDecodeError):
            continue
        if "Event Type" in header and "Glucose Value" in header:
            found.append(path)
    return found


def import_new_clarity(store: Store, folder: str | Path, tz: str = "America/Toronto") -> dict:
    from .clarity import import_clarity_csv
    from .photos import sha256_of

    done = store.imported_hashes("clarity")
    counts = {"files": 0, "readings": 0, "insulin_doses": 0}
    for path in find_clarity_exports(folder):
        digest = sha256_of(path)
        if digest in done:
            continue
        readings, doses = import_clarity_csv(path, tz=tz)
        store.add_readings(readings)
        store.add_doses(doses)
        store.mark_imported(digest, "clarity")
        counts["files"] += 1
        counts["readings"] += len(readings)
        counts["insulin_doses"] += len(doses)
    return counts


NO_ACCESS = (
    "no permission to read {folder}. macOS blocks background jobs from Downloads and "
    "iCloud Drive: System Settings > Privacy & Security > Full Disk Access > add "
    "~/.local/bin/uv (the program the scheduled job runs)."
)


def _readable(folder: Path) -> str | None:
    """None if the folder can be listed, else the reason. Path.glob hides
    permission errors as 'no files', so check explicitly."""
    import os

    try:
        os.listdir(folder)
    except FileNotFoundError:
        return f"{folder} does not exist"
    except PermissionError:
        return NO_ACCESS.format(folder=folder)
    return None


def daily(store: Store, parser: MealParser, photo_folder: str | Path, clarity_folder: str | Path,
          tz: str = "America/Toronto") -> dict:
    """The scheduled job: new Clarity exports, new meal photos, then insights."""
    photo_folder, clarity_folder = Path(photo_folder), Path(clarity_folder)
    problems = [p for p in (_readable(clarity_folder), _readable(photo_folder)) if p]
    empty_photos = {"imported": [], "failed": []}
    empty_clarity = {"files": 0, "readings": 0, "insulin_doses": 0}
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "problems": problems,
        "clarity": empty_clarity if _readable(clarity_folder) else import_new_clarity(store, clarity_folder, tz),
        "photos": empty_photos if _readable(photo_folder) else import_photos(store, parser, photo_folder, tz),
        "meals_logged_total": len(store.meals()),
    }
    report["insights"] = insights(store)
    return report
