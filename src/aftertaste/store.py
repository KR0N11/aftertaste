"""SQLite storage. One file, stays on your machine.

Writes are idempotent: syncing the same Dexcom window twice, or importing
the same Clarity export twice, does not duplicate anything.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .models import GlucoseReading, InsulinDose, InsulinKind, Meal, Nutrition, utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    time      TEXT PRIMARY KEY,          -- UTC ISO 8601
    mgdl      REAL NOT NULL,
    source_id TEXT
);
CREATE TABLE IF NOT EXISTS doses (
    id        INTEGER PRIMARY KEY,
    time      TEXT NOT NULL,
    units     REAL NOT NULL,
    kind      TEXT NOT NULL,
    source_id TEXT,
    UNIQUE (time, units, kind)
);
CREATE TABLE IF NOT EXISTS meals (
    id          INTEGER PRIMARY KEY,
    eaten_at    TEXT NOT NULL,
    description TEXT NOT NULL,
    photo_path  TEXT,
    nutrition   TEXT                      -- JSON, null until parsed
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _ts(dt: datetime) -> str:
    return utc(dt).isoformat()


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _nutrition_to_json(n: Nutrition | None) -> str | None:
    if n is None:
        return None
    return json.dumps(
        {
            "carbs_g": n.carbs_g,
            "fiber_g": n.fiber_g,
            "fat_g": n.fat_g,
            "protein_g": n.protein_g,
            "confidence": n.confidence,
            "items": list(n.items),
            "notes": n.notes,
        }
    )


def _nutrition_from_json(s: str | None) -> Nutrition | None:
    if not s:
        return None
    d = json.loads(s)
    return Nutrition(
        carbs_g=d["carbs_g"],
        fiber_g=d["fiber_g"],
        fat_g=d["fat_g"],
        protein_g=d["protein_g"],
        confidence=d["confidence"],
        items=tuple(d.get("items", ())),
        notes=d.get("notes", ""),
    )


class Store:
    def __init__(self, path: str | Path = "data/aftertaste.db") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)

    # readings -------------------------------------------------------------
    def add_readings(self, readings: Iterable[GlucoseReading]) -> int:
        cur = self.db.executemany(
            "INSERT INTO readings (time, mgdl, source_id) VALUES (?, ?, ?) "
            "ON CONFLICT(time) DO UPDATE SET mgdl = excluded.mgdl",
            ((_ts(r.time), r.mgdl, r.source_id) for r in readings),
        )
        self.db.commit()
        return cur.rowcount

    def readings(self, start: datetime | None = None, end: datetime | None = None) -> list[GlucoseReading]:
        q, args = "SELECT time, mgdl, source_id FROM readings", []
        if start or end:
            q += " WHERE time >= ? AND time <= ?"
            args = [_ts(start) if start else "", _ts(end) if end else "9999"]
        q += " ORDER BY time"
        return [GlucoseReading(_dt(t), m, s) for t, m, s in self.db.execute(q, args)]

    def latest_reading_time(self) -> datetime | None:
        row = self.db.execute("SELECT MAX(time) FROM readings").fetchone()
        return _dt(row[0]) if row and row[0] else None

    # doses ----------------------------------------------------------------
    def add_doses(self, doses: Iterable[InsulinDose]) -> int:
        cur = self.db.executemany(
            "INSERT OR IGNORE INTO doses (time, units, kind, source_id) VALUES (?, ?, ?, ?)",
            ((_ts(d.time), d.units, d.kind.value, d.source_id) for d in doses),
        )
        self.db.commit()
        return cur.rowcount

    def doses(self) -> list[InsulinDose]:
        rows = self.db.execute("SELECT time, units, kind, source_id FROM doses ORDER BY time")
        return [InsulinDose(_dt(t), u, InsulinKind(k), s) for t, u, k, s in rows]

    # meals ----------------------------------------------------------------
    def add_meal(self, meal: Meal) -> int:
        cur = self.db.execute(
            "INSERT INTO meals (eaten_at, description, photo_path, nutrition) VALUES (?, ?, ?, ?)",
            (_ts(meal.eaten_at), meal.description, meal.photo_path, _nutrition_to_json(meal.nutrition)),
        )
        self.db.commit()
        meal.id = cur.lastrowid
        return meal.id

    def set_nutrition(self, meal_id: int, nutrition: Nutrition) -> None:
        self.db.execute(
            "UPDATE meals SET nutrition = ? WHERE id = ?", (_nutrition_to_json(nutrition), meal_id)
        )
        self.db.commit()

    def meals(self) -> list[Meal]:
        rows = self.db.execute(
            "SELECT id, eaten_at, description, photo_path, nutrition FROM meals ORDER BY eaten_at"
        )
        return [
            Meal(eaten_at=_dt(e), description=d, photo_path=p, nutrition=_nutrition_from_json(n), id=i)
            for i, e, d, p, n in rows
        ]

    def meal(self, meal_id: int) -> Meal | None:
        return next((m for m in self.meals() if m.id == meal_id), None)

    # meta -----------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.db.commit()
