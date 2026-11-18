"""How good are the carb estimates? Scored against meals with known carbs.

Input CSV columns: id, description, photo, true_carbs_g
  (photo is a path relative to the CSV, may be empty; description may be empty)

Estimates are cached per meal, so re-running the report does not pay again.
Every meal in the CSV is scored; errors count as failures, not silently dropped.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from .models import Nutrition
from .nutrition import MealParser, ParseError
from .store import _nutrition_from_json, _nutrition_to_json

WITHIN_G = 10.0


@dataclass(frozen=True)
class Case:
    id: str
    description: str
    photo: Path | None
    true_carbs_g: float


@dataclass(frozen=True)
class Scored:
    case: Case
    estimate: Nutrition | None
    error: str | None = None

    @property
    def abs_error(self) -> float | None:
        if self.estimate is None:
            return None
        return abs(self.estimate.carbs_g - self.case.true_carbs_g)


def load_cases(csv_path: str | Path) -> list[Case]:
    base = Path(csv_path).parent
    out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            photo = (row.get("photo") or "").strip()
            out.append(
                Case(
                    id=row["id"].strip(),
                    description=(row.get("description") or "").strip(),
                    photo=base / photo if photo else None,
                    true_carbs_g=float(row["true_carbs_g"]),
                )
            )
    return out


def _cache_key(case: Case) -> str:
    h = hashlib.sha256(case.description.encode())
    if case.photo:
        h.update(case.photo.read_bytes())
    return f"{case.id}-{h.hexdigest()[:16]}"


def run(cases: list[Case], parser: MealParser, cache_dir: str | Path = "eval_cache") -> list[Scored]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        path = cache / f"{_cache_key(case)}.json"
        if path.exists():
            results.append(Scored(case, _nutrition_from_json(path.read_text())))
            continue
        try:
            n = parser.parse(text=case.description or None, photo=case.photo)
        except ParseError as e:
            results.append(Scored(case, None, str(e)))
            continue
        path.write_text(_nutrition_to_json(n))
        results.append(Scored(case, n))
    return results


def summarize(results: list[Scored]) -> dict:
    ok = [r for r in results if r.abs_error is not None]
    errors = [r.abs_error for r in ok]
    by_conf: dict[str, dict] = {}
    for level in ("high", "medium", "low"):
        group = [r.abs_error for r in ok if r.estimate.confidence == level]
        by_conf[level] = {
            "n": len(group),
            "mae_g": round(mean(group), 1) if group else None,
        }
    worst = sorted(ok, key=lambda r: r.abs_error, reverse=True)[:5]
    return {
        "n_cases": len(results),
        "n_scored": len(ok),
        "n_failed": len(results) - len(ok),
        "mae_g": round(mean(errors), 1) if errors else None,
        "median_ae_g": round(median(errors), 1) if errors else None,
        f"within_{int(WITHIN_G)}g_pct": round(100 * sum(e <= WITHIN_G for e in errors) / len(errors), 1)
        if errors
        else None,
        "by_confidence": by_conf,
        "worst": [
            {
                "id": r.case.id,
                "true_g": r.case.true_carbs_g,
                "estimated_g": r.estimate.carbs_g,
                "confidence": r.estimate.confidence,
            }
            for r in worst
        ],
    }


def to_markdown(summary: dict) -> str:
    lines = [
        "# Carb estimation eval",
        "",
        f"- Meals: {summary['n_cases']} ({summary['n_failed']} failed to parse, counted below as unscored)",
        f"- Mean absolute error: {summary['mae_g']} g",
        f"- Median absolute error: {summary['median_ae_g']} g",
        f"- Within ±{int(WITHIN_G)} g: {summary[f'within_{int(WITHIN_G)}g_pct']}%",
        "",
        "| Confidence | Meals | MAE (g) |",
        "|---|---|---|",
    ]
    for level, d in summary["by_confidence"].items():
        lines.append(f"| {level} | {d['n']} | {d['mae_g']} |")
    lines += ["", "Worst 5:", ""]
    for w in summary["worst"]:
        lines.append(f"- {w['id']}: true {w['true_g']} g, estimated {w['estimated_g']} g ({w['confidence']})")
    return "\n".join(lines) + "\n"


def write_report(summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "carb_eval.json").write_text(json.dumps(summary, indent=2))
    (out / "carb_eval.md").write_text(to_markdown(summary))
