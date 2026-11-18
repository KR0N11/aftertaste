from datetime import timedelta

from aftertaste.models import GlucoseReading, InsulinDose, Meal, Nutrition
from aftertaste.store import Store


def test_readings_and_doses_are_idempotent(t0):
    s = Store(":memory:")
    readings = [GlucoseReading(t0 + timedelta(minutes=5 * i), 100 + i) for i in range(10)]
    doses = [InsulinDose(t0, 3.5)]
    s.add_readings(readings)
    s.add_readings(readings)
    s.add_doses(doses)
    s.add_doses(doses)
    assert len(s.readings()) == 10
    assert len(s.doses()) == 1
    assert s.latest_reading_time() == readings[-1].time


def test_meal_round_trip(t0):
    s = Store(":memory:")
    n = Nutrition(60, 4, 12, 25, "medium", ("pad thai (1 plate)",), "restaurant portion")
    meal_id = s.add_meal(Meal(t0, "pad thai", n))
    back = s.meal(meal_id)
    assert back.nutrition == n
    assert back.eaten_at == t0


def test_reading_range_query(t0):
    s = Store(":memory:")
    s.add_readings([GlucoseReading(t0 + timedelta(hours=h), 100) for h in range(10)])
    assert len(s.readings(t0 + timedelta(hours=2), t0 + timedelta(hours=4))) == 3
