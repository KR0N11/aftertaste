"""Meal photo or text -> carbs, fibre, fat, protein, using a vision LLM through
a small LangGraph workflow.

Two interchangeable model backends sit behind the same graph:
  * local (default): an open-weights vision model served by Ollama on this
    machine. No API key, and the meal photo never leaves the computer.
  * claude: Claude through the Anthropic API (set AFTERTASTE_LLM=claude).

    estimate ──► check ──► accept
       ▲           │
       └─ retry ◄──┘   (flagged, first pass only: re-estimate at higher effort)

`check` is plain code, not a model call: it flags estimates that are
physically impossible (fibre above total carbs), implausibly large, empty, or
low confidence. A flagged first pass is re-run once at higher effort with the
problems listed; if the second pass is still flagged, the meal is kept but
marked low confidence so it never drives a prediction silently.

Design rules:
  * Claude only sees the meal (photo and/or text) and, on a retry, the list of
    problems `check` found. Never glucose, insulin, or anything about the person.
  * Claude estimates per item; totals are summed here in code, so the model
    never does arithmetic that ends up in a number we report.
  * Low confidence is a first-class answer. A wrong carb count is worse than
    "not sure".
"""

from __future__ import annotations

import base64
import mimetypes
import os
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal, Protocol, TypedDict

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .models import Nutrition

MODEL = "claude-opus-5-5"
FIRST_EFFORT = "medium"
RETRY_EFFORT = "high"
MAX_ATTEMPTS = 2
MAX_ITEM_CARBS_G = 250.0
MAX_MEAL_CARBS_G = 400.0

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


def to_nutrition(estimate: MealEstimate, problems: list[str] | None = None) -> Nutrition:
    items = estimate.items
    confidence = estimate.confidence if items else "low"
    notes = estimate.assumptions
    if problems:
        confidence = "low"
        notes = f"flagged: {'; '.join(problems)}. {notes}".strip()
    return Nutrition(
        carbs_g=round(sum(i.carbs_g for i in items), 1),
        fiber_g=round(sum(i.fiber_g for i in items), 1),
        fat_g=round(sum(i.fat_g for i in items), 1),
        protein_g=round(sum(i.protein_g for i in items), 1),
        confidence=confidence,
        items=tuple(f"{i.name} ({i.portion})" for i in items),
        notes=notes,
    )


def check(estimate: MealEstimate) -> list[str]:
    """Deterministic sanity checks. An empty list means the estimate passes."""
    problems: list[str] = []
    if not estimate.items:
        problems.append("no food items found")
    for i in estimate.items:
        if i.fiber_g > i.carbs_g:
            problems.append(f"{i.name}: fibre ({i.fiber_g:g} g) exceeds total carbs ({i.carbs_g:g} g)")
        if i.carbs_g > MAX_ITEM_CARBS_G:
            problems.append(f"{i.name}: {i.carbs_g:g} g carbs is implausible for one item")
    total = sum(i.carbs_g for i in estimate.items)
    if total > MAX_MEAL_CARBS_G:
        problems.append(f"meal total of {total:g} g carbs is implausible")
    if estimate.confidence == "low" and estimate.items:
        problems.append("low confidence")
    return problems


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


def retry_content(content: list[dict], problems: list[str]) -> list[dict]:
    note = (
        "A first estimate of this meal was flagged by automatic checks: "
        + "; ".join(problems)
        + ". Look again carefully and give a corrected estimate."
    )
    return [*content, {"type": "text", "text": note}]


# --- the model call ---------------------------------------------------------

# (content blocks, effort) -> (parsed estimate or None, stop_reason)
Estimator = Callable[[list[dict], str], tuple[MealEstimate | None, str | None]]


class _ChatAnthropic(ChatAnthropic):
    """ChatAnthropic always hands the SDK an explicit api_key (empty when
    ANTHROPIC_API_KEY is unset), which bypasses the SDK's own credential
    lookup and so breaks `ant auth login`. Build the client the SDK's way."""

    @cached_property
    def _client(self) -> anthropic.Anthropic:
        try:
            return anthropic.Anthropic(max_retries=self.max_retries)
        except anthropic.CredentialsError as e:
            raise ParseError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`"
            ) from e


def chat_model(model: str = MODEL, effort: str = FIRST_EFFORT) -> ChatAnthropic:
    return _ChatAnthropic(
        model=model,
        max_tokens=16000,
        effort=effort,
        # On a safety decline the API re-runs the request on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


def langchain_estimator(model: str = MODEL) -> Estimator:
    chains = {}

    def estimate(content: list[dict], effort: str) -> tuple[MealEstimate | None, str | None]:
        if effort not in chains:
            chains[effort] = chat_model(model, effort).with_structured_output(
                MealEstimate, method="json_schema", include_raw=True
            )
        out = chains[effort].invoke([SystemMessage(SYSTEM), HumanMessage(content=content)])
        stop_reason = out["raw"].response_metadata.get("stop_reason")
        return out["parsed"], stop_reason

    return estimate


# --- local model (default): Ollama ------------------------------------------

LOCAL_MODEL = "qwen3-vl:8b-instruct"  # the default qwen3-vl:8b tag is the slower thinking variant


def local_model_name() -> str:
    return os.environ.get("AFTERTASTE_LOCAL_MODEL", LOCAL_MODEL)


def to_ollama_content(content: list[dict]) -> list[dict]:
    """Anthropic-style image blocks -> data-URL image parts LangChain's Ollama
    adapter understands. Text blocks pass through unchanged."""
    out = []
    for block in content:
        if block["type"] == "image":
            src = block["source"]
            out.append({"type": "image_url", "image_url": f"data:{src['media_type']};base64,{src['data']}"})
        else:
            out.append(block)
    return out


def ollama_estimator(model: str | None = None, base_url: str | None = None) -> Estimator:
    """Same contract as the Claude estimator. `effort` has no local equivalent,
    so a retry differs only by the list of problems the check found."""
    import httpx
    import ollama
    from langchain_ollama import ChatOllama

    model = model or local_model_name()
    chain = None

    def estimate(content: list[dict], effort: str) -> tuple[MealEstimate | None, str | None]:
        nonlocal chain
        if chain is None:
            # Qwen's recommended sampling for its VL instruct mode. Greedy decoding
            # (temperature 0) makes these models loop; the fixed seed keeps
            # eval runs reproducible.
            llm = ChatOllama(
                model=model, base_url=base_url, reasoning=False, num_ctx=8192, num_predict=2048,
                temperature=0.7, top_p=0.8, top_k=20, seed=0,
            )
            chain = llm.with_structured_output(MealEstimate, method="json_schema", include_raw=True)
        try:
            out = chain.invoke([SystemMessage(SYSTEM), HumanMessage(content=to_ollama_content(content))])
        except (ConnectionError, httpx.ConnectError) as e:
            raise ParseError("local model not reachable: start it with `ollama serve`") from e
        except ollama.ResponseError as e:
            raise ParseError(f"local model error ({model}): {e}. Try `ollama pull {model}`") from e
        return out["parsed"], out["raw"].response_metadata.get("done_reason")

    return estimate


def default_estimator() -> Estimator:
    backend = os.environ.get("AFTERTASTE_LLM", "local")
    if backend == "local":
        return ollama_estimator()
    if backend == "claude":
        return langchain_estimator()
    raise ParseError(f"unknown AFTERTASTE_LLM={backend!r}; use 'local' or 'claude'")


def model_label() -> str:
    """Which model the default parser uses, for eval reports and cache folders."""
    return local_model_name() if os.environ.get("AFTERTASTE_LLM", "local") == "local" else MODEL


# --- the graph --------------------------------------------------------------

class MealState(TypedDict, total=False):
    content: list[dict]
    attempts: int
    estimate: MealEstimate | None
    stop_reason: str | None
    problems: list[str]


def build_graph(estimator: Estimator):
    def estimate_node(state: MealState) -> MealState:
        attempts = state.get("attempts", 0)
        content = state["content"]
        effort = FIRST_EFFORT
        if attempts:
            content = retry_content(content, state["problems"])
            effort = RETRY_EFFORT
        estimate, stop_reason = estimator(content, effort)
        return {"estimate": estimate, "stop_reason": stop_reason, "attempts": attempts + 1}

    def check_node(state: MealState) -> MealState:
        estimate = state.get("estimate")
        return {"problems": check(estimate) if estimate is not None else []}

    def route(state: MealState) -> str:
        if state.get("estimate") is None:
            return END
        if state["problems"] and state["attempts"] < MAX_ATTEMPTS:
            return "estimate"
        return END

    graph = StateGraph(MealState)
    graph.add_node("estimate", estimate_node)
    graph.add_node("check", check_node)
    graph.add_edge(START, "estimate")
    graph.add_edge("estimate", "check")
    graph.add_conditional_edges("check", route, ["estimate", END])
    return graph.compile()


class LLMMealParser:
    def __init__(self, estimator: Estimator | None = None) -> None:
        # The model client is created on first use, so the app still starts
        # (and shows existing data) when no model is reachable.
        self.graph = build_graph(estimator or default_estimator())

    def run(self, text: str | None = None, photo: str | Path | None = None) -> MealState:
        return self.graph.invoke({"content": build_content(text, photo)})

    def parse(self, text: str | None = None, photo: str | Path | None = None) -> Nutrition:
        state = self.run(text, photo)
        if state["stop_reason"] == "refusal":
            raise ParseError("the model declined this input; enter the meal by hand")
        if state["estimate"] is None:
            raise ParseError(f"no usable estimate (stop_reason={state['stop_reason']})")
        return to_nutrition(state["estimate"], state["problems"])
