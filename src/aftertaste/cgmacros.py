"""CGMacros (PhysioNet, CC BY-NC-SA 4.0): 45 adults, Dexcom G6 + labelled meals.

Gutierrez-Osuna et al., CGMacros v1.0.0, https://doi.org/10.13026/3z8q-x658
Download it yourself; it is not redistributed here.

What the real files look like (checked against v1.0.0, not just the data
dictionary): column sets differ between participants, "Amount Consumed" has
a trailing space in some files and values like 100, 1.0, 300, 0 or blank,
meal types mix case ("dinner", "Dinner", "Snacks", "snack 1"), timestamps are
date-shifted local wall-clock time, and glucose is at 1-minute resolution.

Only meals marked 100 (% consumed) are used, so a photo and its label
describe the same food. Everything skipped is counted, never dropped silently.
These participants do not have type 1 diabetes and take no insulin, so
results here are about the method on public data, not about type 1.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

from .features import FEATURE_NAMES, LOCAL_TZ, Row, build_dataset
from .models import GlucoseReading, Meal, Nutrition
from .predict import (
    CarbOnlyBaseline, EvalResult, ResponseModel, _bootstrap_improvement, chronological_split,
    evaluate_grouped,
)

FULL_MEAL = 100.0


def _get(row: dict, name: str) -> str:
    for key, value in row.items():
        if key and key.strip().lower() == name.lower():
            return (value or "").strip()
    return ""


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class Participant:
    id: str
    folder: Path
    readings: list[GlucoseReading]
    meals: list[Meal]
    photos: dict[int, Path]  # meal index -> photo path
    skipped: Counter = field(default_factory=Counter)


def load_participant(folder: str | Path) -> Participant:
    folder = Path(folder)
    pid = folder.name
    csv_path = folder / f"{pid}.csv"
    readings: list[GlucoseReading] = []
    meals: list[Meal] = []
    photos: dict[int, Path] = {}
    skipped: Counter = Counter()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ts = _get(row, "Timestamp")
            if not ts:
                continue
            when = datetime.fromisoformat(ts).replace(tzinfo=LOCAL_TZ)
            dex = _num(_get(row, "Dexcom GL"))
            if dex is not None:
                readings.append(GlucoseReading(when, dex))
            if not _get(row, "Meal Type"):
                continue
            amount = _num(_get(row, "Amount Consumed"))
            if amount != FULL_MEAL:
                skipped["not_fully_eaten_or_unknown"] += 1
                continue
            carbs = _num(_get(row, "Carbs"))
            if carbs is None:
                skipped["no_carbs_label"] += 1
                continue
            nutrition = Nutrition(
                carbs_g=carbs,
                fiber_g=_num(_get(row, "Fiber")) or 0.0,
                fat_g=_num(_get(row, "Fat")) or 0.0,
                protein_g=_num(_get(row, "Protein")) or 0.0,
                confidence="label",
            )
            meals.append(Meal(when, _get(row, "Meal Type").lower(), nutrition))
            image = _get(row, "Image path")
            if image and (folder / image).exists():
                photos[len(meals) - 1] = folder / image
    readings.sort(key=lambda r: r.time)
    return Participant(pid, folder, readings, meals, photos, skipped)


def participants(root: str | Path) -> list[Participant]:
    root = Path(root)
    folders = sorted(p for p in root.glob("CGMacros-0*") if p.is_dir())
    if not folders:
        raise FileNotFoundError(f"no CGMacros-0XX folders under {root}")
    return [load_participant(p) for p in folders]


def evaluate_public(root: str | Path) -> dict:
    """Glucose-rise prediction on every participant: each person's later
    meals are held out; one model is trained on everyone's earlier meals."""
    people = participants(root)
    groups = {}
    skipped: Counter = Counter()
    n_meals = 0
    for p in people:
        skipped.update(p.skipped)
        n_meals += len(p.meals) + sum(p.skipped.values())
        rows, why = build_dataset(p.meals, p.readings, doses=[])
        skipped.update(why)
        if rows:
            groups[p.id] = rows
    result: EvalResult = evaluate_grouped(groups)
    return {
        "dataset": "CGMacros v1.0.0 (PhysioNet), 45 adults without type 1 diabetes, no insulin",
        "participants_used": len(groups),
        "meal_rows_in_files": n_meals,
        "clean_meals": sum(len(r) for r in groups.values()),
        "skipped": dict(skipped),
        "evaluation": result.as_dict(),
    }


def write_carb_labels(root: str | Path, out_csv: str | Path) -> int:
    """Photo-only carb eval file: id, description (empty), photo, true_carbs_g.
    Photo paths are written relative to the output CSV."""
    out_csv = Path(out_csv).resolve()
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "description", "photo", "true_carbs_g"])
        for p in participants(root):
            for i, photo in sorted(p.photos.items()):
                meal = p.meals[i]
                rel = os.path.relpath(Path(photo).resolve(), out_csv.parent)
                stamp = meal.eaten_at.strftime("%Y%m%dT%H%M")
                w.writerow([f"{p.id}-{stamp}", "", str(rel), meal.nutrition.carbs_g])
                n += 1
    return n


PERSONAL_FEATURES = ("person_rise_per_g", "person_mean_rise", "a1c")


def read_a1c(root: str | Path) -> dict[str, float]:
    """A1c (%) per participant from bio.csv, keyed like 'CGMacros-001'."""
    out = {}
    with open(Path(root) / "bio.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            subject = _get(row, "subject")
            a1c = _num(_get(row, "A1c PDL (Lab)"))
            if subject and a1c is not None:
                out[f"CGMacros-{int(float(subject)):03d}"] = a1c
    return out


def _with(row: Row, extra: dict[str, float]) -> Row:
    return Row(meal=row.meal, response=row.response, features={**row.features, **extra})


def personalize(
    train: list[Row], test: list[Row], a1c: float, fallback_rate: float, fallback_mean: float
) -> tuple[list[Row], list[Row]]:
    """Add what this person's EARLIER meals say about them.

    Training rows use leave-one-out (a meal never sees its own rise); test
    rows use all of the person's training meals. Nothing from test leaks in.
    """
    rises = np.array([r.target for r in train])
    carbs = np.array([r.features["net_carbs_g"] for r in train])
    n = len(train)

    def stats(total_rise: float, total_carbs: float, count: int) -> dict[str, float]:
        if count == 0 or total_carbs <= 0:
            return {"person_rise_per_g": fallback_rate, "person_mean_rise": fallback_mean, "a1c": a1c}
        return {
            "person_rise_per_g": total_rise / total_carbs,
            "person_mean_rise": total_rise / count,
            "a1c": a1c,
        }

    new_train = [
        _with(r, stats(rises.sum() - rises[i], carbs.sum() - carbs[i], n - 1))
        for i, r in enumerate(train)
    ]
    new_test = [_with(r, stats(rises.sum(), carbs.sum(), n)) for r in test]
    return new_train, new_test


def evaluate_personalized(root: str | Path, test_fraction: float = 0.25) -> dict:
    """Four predictors on the same held-out meals (each person's later 25%):
    shared carbs-only, personal carbs-only, shared model, personalized model."""
    root = Path(root)
    a1c = read_a1c(root)
    groups: dict[str, tuple[list[Row], list[Row]]] = {}
    for p in participants(root):
        rows, _ = build_dataset(p.meals, p.readings, doses=[])
        if len(rows) >= 4 and p.id in a1c:
            groups[p.id] = chronological_split(rows, test_fraction)

    all_train = [r for tr, _ in groups.values() for r in tr]
    rate = sum(r.target for r in all_train) / max(sum(r.features["net_carbs_g"] for r in all_train), 1e-9)
    mean_rise = float(np.mean([r.target for r in all_train]))

    train, test, personal_pred = [], [], []
    shared_carbs = CarbOnlyBaseline().fit(all_train)
    for pid, (tr, te) in groups.items():
        ptr, pte = personalize(tr, te, a1c[pid], rate, mean_rise)
        train += ptr
        test += pte
        if len(tr) >= 3:
            x = np.array([[r.features["net_carbs_g"]] for r in tr])
            m = LinearRegression().fit(x, [r.target for r in tr])
            personal_pred += list(m.predict(np.array([[r.features["net_carbs_g"]] for r in te])))
        else:
            personal_pred += list(shared_carbs.predict(te))

    y = np.array([r.target for r in test])
    errs = {
        "shared_carbs_only": np.abs(shared_carbs.predict(test) - y),
        "personal_carbs_only": np.abs(np.array(personal_pred) - y),
        "shared_model": np.abs(ResponseModel().fit(train).predict(test) - y),
        "personalized_model": np.abs(
            ResponseModel(names=FEATURE_NAMES + PERSONAL_FEATURES).fit(train).predict(test) - y
        ),
    }
    mae = {k: round(float(v.mean()), 1) for k, v in errs.items()}

    def vs(base: str) -> dict:
        b, m = errs[base], errs["personalized_model"]
        lo, hi = _bootstrap_improvement(b, m)
        return {
            "improvement_pct": round(100 * (b.mean() - m.mean()) / b.mean(), 1),
            "ci95_pct": [round(lo, 1), round(hi, 1)],
        }

    return {
        "dataset": "CGMacros v1.0.0 (PhysioNet), adults without type 1 diabetes, no insulin",
        "participants": len(groups),
        "n_train": len(train),
        "n_test": len(test),
        "mae_mgdl": mae,
        "personalized_vs_shared_carbs_only": vs("shared_carbs_only"),
        "personalized_vs_personal_carbs_only": vs("personal_carbs_only"),
    }
