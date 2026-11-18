"""Meal photo or text -> carbs, fibre, fat, protein, using Claude.

Design rules:
  * Claude only sees the meal (photo and/or text). Never glucose, insulin,
    or anything about the person.
  * Claude estimates per item; totals are summed here in code, so the model
    never does arithmetic that ends up in a number we report.
  * Low confidence is a first-class answer. A wrong carb count is worse than
    "not sure".
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Literal, Protocol

import anthropic
from pydantic import BaseModel, Field

from .models import Nutrition

MODEL = "claude-opus-5-5"

SYSTEM = """You estimate the nutrition of a meal from a photo and/or a short description.

For each distinct food item, give its name, the portion you assume, and grams of
total carbohydrate, fibre, fat and protein for that portion. Use standard
nutrition references (e.g. USDA FoodData Central, Canadian Nutrient File) and
typical portions when the size is not stated; say what you assumed.

Confidence:
- high: item and portion are clear (packaged food, measured amount, clear photo)
- medium: item is clear but portion is estimated
- low: the item itself is uncertain, or the photo/description is too vague

Do not add up totals; list items only. Do not give any medical, insulin or
dosing advice. If the input does not show food, return no items and low confidence."""


class Item(BaseModel):
    name: str
    portion: str = Field(description="assumed portion, e.g. '2 slices, 14 inch pizza'")
    carbs_g: float = Field(ge=0)
    fiber_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)
    protein_g: float = Field(ge=0)


class MealEstimate(BaseModel):
    items: list[Item]
    confidence: Literal["high", "medium", "low"]
    assumptions: str


class ParseError(RuntimeError):
    pass


class MealParser(Protocol):
    def parse(self, text: str | None = None, photo: str | Path | None = None) -> Nutrition: ...


def to_nutrition(estimate: MealEstimate) -> Nutrition:
    items = estimate.items
    return Nutrition(
        carbs_g=round(sum(i.carbs_g for i in items), 1),
        fiber_g=round(sum(i.fiber_g for i in items), 1),
        fat_g=round(sum(i.fat_g for i in items), 1),
        protein_g=round(sum(i.protein_g for i in items), 1),
        confidence=estimate.confidence if items else "low",
        items=tuple(f"{i.name} ({i.portion})" for i in items),
        notes=estimate.assumptions,
    )


def _image_block(photo: str | Path) -> dict:
    path = Path(photo)
    media_type = mimetypes.guess_type(path.name)[0]
    if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        raise ParseError(f"unsupported image type: {path.suffix}")
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def build_content(text: str | None, photo: str | Path | None) -> list[dict]:
    if not text and not photo:
        raise ParseError("give a photo, a description, or both")
    content: list[dict] = []
    if photo:
        content.append(_image_block(photo))
    content.append({"type": "text", "text": f"Meal: {text}" if text else "Estimate this meal."})
    return content


class ClaudeMealParser:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL) -> None:
        # Created on first use, so the app still starts (and shows existing
        # data) on a machine without Anthropic credentials.
        self._client = client
        self.model = model

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            try:
                self._client = anthropic.Anthropic()
            except anthropic.CredentialsError as e:
                raise ParseError(
                    "no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`"
                ) from e
        return self._client

    def estimate(self, text: str | None = None, photo: str | Path | None = None) -> MealEstimate:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_content(text, photo)}],
            output_format=MealEstimate,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise ParseError("the model declined this input; enter the meal by hand")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise ParseError(f"no usable estimate (stop_reason={response.stop_reason})")
        return response.parsed_output

    def parse(self, text: str | None = None, photo: str | Path | None = None) -> Nutrition:
        return to_nutrition(self.estimate(text, photo))
