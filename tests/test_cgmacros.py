"""Tests against a tiny file in the real CGMacros shape (columns copied from
v1.0.0, values made up)."""

import csv
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from aftertaste.cgmacros import evaluate_personalized, load_participant, read_a1c, write_carb_labels

HEADER_A = ["Unnamed: 0", "Timestamp", "Libre GL", "Dexcom GL", "HR", "Calories (Activity)", "METs",
            "Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed ", "Image path"]
HEADER_B = ["Timestamp", "Libre GL", "Dexcom GL", "HR", "Calories (Activity)", "Intensity",
            "Meal Type", "Calories", "Carbs", "Protein", "Fat", "Fiber", "Amount Consumed", "Image path"]


def write_person(folder: Path, header, sensitivity: float, days: int = 12, photo: bool = False):
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 5, 1, 6, 0)
    meals = {}
    for d in range(days):
        for hour, carbs in ((8, 30 + 5 * d % 40), (13, 60), (19, 45 + d)):
            meals[start + timedelta(days=d, hours=hour - 6)] = carbs
    rows = []
    for minute in range(days * 24 * 60):
        t = start + timedelta(minutes=minute)
        g = 95.0
        for when, carbs in meals.items():
            m = (t - when).total_seconds() / 60
            if 0 < m < 180:
                g += sensitivity * carbs * (m / 45 if m < 45 else max(0, (180 - m) / 135))
        row = dict.fromkeys(header, "")
        row["Timestamp"] = t.strftime("%Y-%m-%d %H:%M:%S")
        row["Dexcom GL"] = f"{g:.1f}"
        if t in meals:
            row["Meal Type"] = "Lunch" if t.hour == 13 else "dinner"
            row["Carbs"], row["Protein"], row["Fat"], row["Fiber"] = meals[t], 20, 10, 2
            amount_key = [k for k in header if k.strip() == "Amount Consumed"][0]
            row[amount_key] = "100.0" if t.hour != 8 else "1.0"  # breakfasts excluded
            if photo:
                name = f"photos/{t:%Y%m%d%H%M}.jpg"
                (folder / name).write_bytes(b"\xff\xd8\xff")
                row["Image path"] = name
        rows.append(row)
    with open(folder / f"{folder.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("cgm") / "CGMacros"
    write_person(r / "CGMacros-001", HEADER_A, sensitivity=0.8)
    write_person(r / "CGMacros-002", HEADER_B, sensitivity=2.0, photo=True)
    (r / "bio.csv").write_text("subject,Age,A1c PDL (Lab)\n1,30,5.2\n2,55,7.1\n")
    return r


def test_tolerates_column_differences_and_filters_partial_meals(root):
    p = load_participant(root / "CGMacros-001")
    assert len(p.meals) == 24  # 12 days x lunch + dinner; breakfasts were "1.0"
    assert p.skipped["not_fully_eaten_or_unknown"] == 12
    assert p.meals[0].description in ("lunch", "dinner")
    assert p.meals[0].eaten_at.tzinfo is not None


def test_reads_a1c(root):
    assert read_a1c(root) == {"CGMacros-001": 5.2, "CGMacros-002": 7.1}


def test_personalization_helps_when_people_differ(root):
    result = evaluate_personalized(root)
    mae = result["mae_mgdl"]
    assert result["participants"] == 2
    assert mae["personalized_model"] < mae["shared_carbs_only"]


def test_carb_labels_point_at_real_photos(root, tmp_path):
    out = tmp_path / "labels.csv"
    n = write_carb_labels(root, out)
    rows = list(csv.DictReader(open(out)))
    assert n == len(rows) == 24
    assert all((out.parent / r["photo"]).exists() for r in rows)
