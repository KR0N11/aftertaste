from datetime import timedelta

import pytest

from aftertaste.glucose import measure_response, trend_before
from conftest import trace


def test_flat_trace_has_no_rise(t0):
    r = measure_response(trace([120] * 50), t0)
    assert r.valid
    assert r.peak_rise_mgdl == 0
    assert r.iauc_2h == 0


def test_known_bump(t0):
    # baseline 100 for 30 min before; +2 mg/dL per 5 min for 60 min, then back down
    pre = [100] * 7  # -30 .. 0 min
    up = [100 + 10 * i for i in range(1, 13)]  # 5 .. 60 min -> 220
    down = [220 - 10 * i for i in range(1, 13)]  # 65 .. 120 -> 100
    tail = [100] * 13
    r = measure_response(trace(pre + up + down + tail), t0)
    assert r.valid
    assert r.baseline_mgdl == 100
    assert r.peak_rise_mgdl == 120
    assert r.minutes_to_peak == 60
    assert r.rise_at_60_mgdl == pytest.approx(120)
    # triangle: 0.5 * 120 min * 120 mg/dL
    assert r.iauc_2h == pytest.approx(7200)


def test_gap_after_meal_is_invalid_not_zero(t0):
    readings = trace([110] * 10) + trace([150] * 30, start=t0 + timedelta(minutes=60))
    r = measure_response(readings, t0)
    assert not r.valid
    assert any("gap" in p for p in r.problems)


def test_no_baseline_is_invalid(t0):
    r = measure_response(trace([150] * 40, start=t0 + timedelta(minutes=30)), t0)
    assert not r.valid


def test_trace_ending_early_is_invalid(t0):
    r = measure_response(trace([120] * 20), t0)  # ends ~65 min after the meal
    assert not r.valid


def test_trend_slope(t0):
    readings = trace([100 + 2 * i for i in range(7)])  # +2 per 5 min up to t0
    assert trend_before(readings, t0) == pytest.approx(0.4)
