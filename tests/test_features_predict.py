from datetime import timedelta

import pytest

from aftertaste.features import build_dataset, is_isolated, meal_bolus
from aftertaste.models import InsulinDose, InsulinKind, Meal, Nutrition
from aftertaste.predict import chronological_split, evaluate
from aftertaste.simulate import simulate

N = Nutrition(carbs_g=50, fiber_g=5, fat_g=10, protein_g=20, confidence="high")


def test_meal_bolus_window(t0):
    meal = Meal(t0, "x", N)
    doses = [
        InsulinDose(t0 - timedelta(minutes=20), 3),
        InsulinDose(t0 + timedelta(minutes=25), 1),
        InsulinDose(t0 + timedelta(minutes=45), 9),  # correction, not the meal bolus
        InsulinDose(t0, 18, InsulinKind.LONG),
    ]
    assert meal_bolus(doses, meal) == 4


def test_overlapping_meals_are_not_isolated(t0):
    a, b, c = Meal(t0, "a", N), Meal(t0 + timedelta(minutes=90), "b", N), Meal(t0 + timedelta(hours=5), "c", N)
    meals = [a, b, c]
    assert not is_isolated(a, meals)
    assert is_isolated(c, meals)


def test_skip_reasons_are_counted(t0):
    readings, doses, meals = simulate(days=3, seed=1)
    meals.append(Meal(meals[0].eaten_at + timedelta(hours=7), "no nutrition"))
    rows, skipped = build_dataset(meals, readings, doses)
    assert skipped["no_nutrition"] == 1
    assert len(rows) + sum(skipped.values()) == len(meals)


def test_split_is_chronological():
    readings, doses, meals = simulate(days=20, seed=2)
    rows, _ = build_dataset(meals, readings, doses)
    train, test = chronological_split(rows)
    assert max(r.meal.eaten_at for r in train) < min(r.meal.eaten_at for r in test)


def test_refuses_to_score_tiny_datasets():
    readings, doses, meals = simulate(days=5, seed=3)
    rows, _ = build_dataset(meals, readings, doses)
    with pytest.raises(ValueError, match="Keep logging"):
        evaluate(rows)


def test_pipeline_recovers_simulated_effects():
    """On SYNTHETIC data where fat and timing matter, the model must beat
    carbs-only with an interval above zero. Guards the pipeline, not a claim."""
    readings, doses, meals = simulate(days=60, seed=7)
    rows, _ = build_dataset(meals, readings, doses)
    result = evaluate(rows)
    assert result.model_mae < result.baseline_mae
    assert result.improvement_ci95[0] > 0
    lo, hi = result.improvement_ci95
    assert lo <= result.improvement_pct <= hi
