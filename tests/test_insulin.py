from datetime import timedelta

import pytest

from aftertaste.insulin import DEFAULT_MODEL, ExponentialInsulinModel, insulin_on_board
from aftertaste.models import InsulinDose, InsulinKind


def test_full_at_dose_and_empty_at_duration():
    assert DEFAULT_MODEL.fraction_remaining(0) == 1.0
    assert DEFAULT_MODEL.fraction_remaining(360) == 0.0
    assert DEFAULT_MODEL.fraction_remaining(359.9) == pytest.approx(0.0, abs=1e-3)


def test_decreases_monotonically():
    values = [DEFAULT_MODEL.fraction_remaining(t) for t in range(0, 361, 5)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_about_half_left_after_two_hours_for_default_curve():
    # sanity range for a 75 min peak / 6 h curve, not a personal value
    assert 0.4 < DEFAULT_MODEL.fraction_remaining(120) < 0.65


def test_rejects_impossible_curve():
    with pytest.raises(ValueError):
        ExponentialInsulinModel(peak_minutes=200, duration_minutes=360)


def test_iob_sums_rapid_doses_and_ignores_future_and_basal(t0):
    doses = [
        InsulinDose(t0 - timedelta(minutes=60), 4.0),
        InsulinDose(t0 - timedelta(minutes=10), 2.0),
        InsulinDose(t0 + timedelta(minutes=10), 5.0),  # not taken yet
        InsulinDose(t0 - timedelta(minutes=30), 20.0, InsulinKind.LONG),
    ]
    expected = 4 * DEFAULT_MODEL.fraction_remaining(60) + 2 * DEFAULT_MODEL.fraction_remaining(10)
    assert insulin_on_board(doses, t0) == pytest.approx(expected)
