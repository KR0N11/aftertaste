"""Plain data types shared by every module.

All timestamps are timezone-aware UTC. Glucose is stored in mg/dL internally;
mmol/L is converted on the way in and can be converted back for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

MGDL_PER_MMOL = 18.0182


def to_mgdl(value: float, unit: str) -> float:
    unit = unit.strip().lower()
    if unit in ("mg/dl", "mgdl"):
        return float(value)
    if unit in ("mmol/l", "mmol"):
        return float(value) * MGDL_PER_MMOL
    raise ValueError(f"unknown glucose unit: {unit!r}")


def to_mmol(mgdl: float) -> float:
    return mgdl / MGDL_PER_MMOL


def utc(dt: datetime) -> datetime:
    """Return dt as aware UTC. Naive datetimes are rejected on purpose:
    a silent local-time guess shifts every meal against the CGM trace."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class GlucoseReading:
    time: datetime
    mgdl: float
    source_id: str | None = None  # Dexcom recordId, for idempotent sync


class InsulinKind(str, Enum):
    RAPID = "rapid"
    LONG = "long"


@dataclass(frozen=True)
class InsulinDose:
    time: datetime
    units: float
    kind: InsulinKind = InsulinKind.RAPID
    source_id: str | None = None


@dataclass(frozen=True)
class Nutrition:
    carbs_g: float
    fiber_g: float
    fat_g: float
    protein_g: float
    confidence: str  # "high" | "medium" | "low"
    items: tuple[str, ...] = ()
    notes: str = ""

    @property
    def net_carbs_g(self) -> float:
        return max(self.carbs_g - self.fiber_g, 0.0)


@dataclass
class Meal:
    eaten_at: datetime
    description: str
    nutrition: Nutrition | None = None
    photo_path: str | None = None
    id: int | None = None


@dataclass(frozen=True)
class MealResponse:
    """What a meal actually did, measured from the CGM trace."""

    baseline_mgdl: float
    peak_rise_mgdl: float
    minutes_to_peak: float
    iauc_2h: float  # incremental area above baseline, mg/dL x min, 0-120 min
    rise_at_60_mgdl: float
    valid: bool = True
    problems: tuple[str, ...] = field(default_factory=tuple)
