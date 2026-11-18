"""Import a Dexcom Clarity CSV export (clarity.dexcom.com -> Export).

The quickest way to get months of history without API approval. Clarity
timestamps are local wall-clock time with no offset, so the timezone must be
given. Columns are found by header prefix, which covers both the mg/dL and
the mmol/L (Canadian) exports. Written against the documented column layout;
check it against your own export the first time.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import GlucoseReading, InsulinDose, InsulinKind, to_mgdl

# Clarity writes "Low" / "High" outside the sensor range (40-400 mg/dL).
SENSOR_LOW_MGDL = 40.0
SENSOR_HIGH_MGDL = 400.0


def _column(header: list[str], prefix: str) -> int | None:
    for i, name in enumerate(header):
        if name.strip().lower().startswith(prefix.lower()):
            return i
    return None


def _unit_from_header(name: str) -> str:
    return "mmol/L" if "mmol" in name.lower() else "mg/dL"


def import_clarity_csv(
    path: str | Path, tz: str = "America/Toronto"
) -> tuple[list[GlucoseReading], list[InsulinDose]]:
    zone = ZoneInfo(tz)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header = rows[0]
    i_time = _column(header, "Timestamp")
    i_type = _column(header, "Event Type")
    i_sub = _column(header, "Event Subtype")
    i_glu = _column(header, "Glucose Value")
    i_ins = _column(header, "Insulin Value")
    i_id = _column(header, "Index")
    if None in (i_time, i_type, i_glu):
        raise ValueError("not a Clarity export: missing Timestamp / Event Type / Glucose Value columns")
    unit = _unit_from_header(header[i_glu])

    readings: list[GlucoseReading] = []
    doses: list[InsulinDose] = []
    for row in rows[1:]:
        if len(row) <= max(i_time, i_type, i_glu) or not row[i_time].strip():
            continue  # patient/device info rows have no timestamp
        when = datetime.fromisoformat(row[i_time].strip()).replace(tzinfo=zone)
        kind = row[i_type].strip().upper()
        source = row[i_id].strip() if i_id is not None else None
        if kind == "EGV":
            raw = row[i_glu].strip()
            if raw.lower() == "low":
                mgdl = SENSOR_LOW_MGDL
            elif raw.lower() == "high":
                mgdl = SENSOR_HIGH_MGDL
            elif raw:
                mgdl = to_mgdl(float(raw), unit)
            else:
                continue
            readings.append(GlucoseReading(when, mgdl, f"clarity:{source}" if source else None))
        elif kind == "INSULIN" and i_ins is not None and row[i_ins].strip():
            sub = row[i_sub].strip().lower() if i_sub is not None else ""
            ins_kind = InsulinKind.LONG if sub.startswith("long") else InsulinKind.RAPID
            doses.append(InsulinDose(when, float(row[i_ins]), ins_kind, f"clarity:{source}" if source else None))
    readings.sort(key=lambda r: r.time)
    doses.sort(key=lambda d: d.time)
    return readings, doses
