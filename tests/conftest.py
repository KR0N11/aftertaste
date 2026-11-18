from datetime import datetime, timedelta, timezone

import pytest

from aftertaste.models import GlucoseReading

T0 = datetime(2026, 3, 2, 17, 0, tzinfo=timezone.utc)


def trace(values, start=None, step=5):
    """Readings every `step` minutes starting at `start` (default T0 - 30 min)."""
    start = start or T0 - timedelta(minutes=30)
    return [GlucoseReading(start + timedelta(minutes=step * i), float(v)) for i, v in enumerate(values)]


@pytest.fixture
def t0():
    return T0
