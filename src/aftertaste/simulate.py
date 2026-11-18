"""A synthetic type 1 patient, for tests and the demo. NOT real data.

Every number produced from this module is synthetic. It exists to check the
pipeline end to end: if the simulated physiology makes fat slow absorption
and mornings spike harder, the model must find that and beat carbs-only.
It says nothing about how well the model works on a real person.

Physiology (deliberately simple):
  * carbs raise glucose by CSF mg/dL per gram, absorbed over a curve that
    stretches with fat; mornings raise it 30% more (dawn effect)
  * protein adds a small late rise
  * rapid insulin lowers glucose by ISF mg/dL per unit along the same
    exponential action curve used in insulin.py
  * boluses roughly match carbs but vary: sometimes late, sometimes short
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from .insulin import DEFAULT_MODEL
from .models import GlucoseReading, InsulinDose, InsulinKind, Meal, Nutrition

STEP_MIN = 5
EFFECT_MIN = 900  # a meal's effect is followed for 15 h


@dataclass(frozen=True)
class Patient:
    csf: float = 4.0  # mg/dL per gram of carbs
    isf: float = 40.0  # mg/dL per unit of insulin
    icr: float = 10.0  # grams per unit, what the patient uses to bolus
    target: float = 115.0
    morning_factor: float = 1.3
    noise_sd: float = 6.0


def _carb_absorbed(minutes: float, fat_g: float) -> float:
    if minutes <= 0:
        return 0.0
    duration = 100 + 2.5 * fat_g  # fat slows absorption
    tau = duration / 4
    return 1 - (1 + minutes / tau) * math.exp(-minutes / tau)


def _protein_absorbed(minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    tau = 90.0
    return 1 - (1 + minutes / tau) * math.exp(-minutes / tau)


def simulate(
    days: int = 60,
    seed: int = 7,
    patient: Patient = Patient(),
    start: datetime = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc),
    estimate_noise_g: float = 0.0,
) -> tuple[list[GlucoseReading], list[InsulinDose], list[Meal]]:
    """Return (cgm readings, rapid insulin doses, meals) for `days` days.

    `estimate_noise_g` adds error to the carbs stored on each meal, to mimic
    an imperfect carb estimate while the body still reacts to the true carbs.
    """
    rng = np.random.default_rng(seed)
    meals: list[Meal] = []
    truth: list[tuple[datetime, Nutrition, bool]] = []
    doses: list[InsulinDose] = []

    # local meal times expressed in UTC (Montreal winter = UTC-5); `start` is
    # 05:00 UTC = midnight local, so hour + 5 lands on the local meal time
    for day in range(days):
        base = start + timedelta(days=day)
        for hour, is_morning in ((8.0, True), (12.5, False), (19.0, False)):
            if rng.random() < 0.08:  # skipped meal
                continue
            when = base + timedelta(hours=hour + 5 + rng.normal(0, 0.5))
            carbs = float(rng.uniform(20, 110))
            n = Nutrition(
                carbs_g=round(carbs, 1),
                fiber_g=round(float(rng.uniform(0, 10)), 1),
                fat_g=round(float(rng.uniform(3, 45)), 1),
                protein_g=round(float(rng.uniform(8, 45)), 1),
                confidence="high",
            )
            truth.append((when, n, is_morning))
            seen_carbs = max(n.carbs_g + rng.normal(0, estimate_noise_g), 0) if estimate_noise_g else n.carbs_g
            meals.append(
                Meal(
                    eaten_at=when,
                    description=f"synthetic meal {len(meals) + 1}",
                    nutrition=Nutrition(
                        carbs_g=round(float(seen_carbs), 1),
                        fiber_g=n.fiber_g,
                        fat_g=n.fat_g,
                        protein_g=n.protein_g,
                        confidence="high",
                    ),
                )
            )
            units = n.net_carbs_g / patient.icr * float(rng.uniform(0.7, 1.15))
            delay = float(rng.choice([-10, 0, 0, 0, 15, 25]))
            doses.append(
                InsulinDose(when + timedelta(minutes=delay), round(units * 2) / 2, InsulinKind.RAPID)
            )

    total_steps = days * 24 * 60 // STEP_MIN
    times = [start + timedelta(minutes=STEP_MIN * i) for i in range(total_steps)]
    glucose = np.full(total_steps, patient.target)
    horizon = EFFECT_MIN // STEP_MIN
    for (when, n, morning), dose in zip(truth, doses):
        first = max(int(math.ceil((when - start).total_seconds() / 60 / STEP_MIN)), 0)
        idx = np.arange(first, min(first + horizon, total_steps))
        minutes = np.array([(times[i] - when).total_seconds() / 60 for i in idx])
        dose_min = minutes - (dose.time - when).total_seconds() / 60
        factor = patient.morning_factor if morning else 1.0
        carbs = np.array([_carb_absorbed(m, n.fat_g) for m in minutes])
        protein = np.array([_protein_absorbed(m - 60) for m in minutes])
        acted = np.array([1 - DEFAULT_MODEL.fraction_remaining(m) if m > 0 else 0.0 for m in dose_min])
        effect = (
            patient.csf * n.net_carbs_g * factor * carbs
            + 0.5 * n.protein_g * protein
            - patient.isf * dose.units * acted
        )
        # whatever a meal leaves behind (over- or under-bolus) is corrected
        # over the following hours, as basal adjustments and corrections would
        effect *= np.exp(-np.clip(minutes - 240, 0, None) / 150)
        glucose[idx] += effect
    drift = np.zeros(total_steps)
    for i in range(1, total_steps):
        drift[i] = 0.97 * drift[i - 1] + rng.normal(0, 1.2)
    glucose = np.clip(glucose + drift + rng.normal(0, patient.noise_sd, total_steps), 40, 400)
    readings = [GlucoseReading(t, float(v)) for t, v in zip(times, glucose)]
    return readings, doses, meals
