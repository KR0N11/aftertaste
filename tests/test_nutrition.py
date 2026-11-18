import json

import pytest
from anthropic.types.beta import BetaMessage

from aftertaste import nutrition
from aftertaste.nutrition import (
    FIRST_EFFORT, MODEL, RETRY_EFFORT, ClaudeMealParser, Item, MealEstimate, ParseError,
    build_content, check, langchain_estimator, to_nutrition,
)

PIZZA = MealEstimate(
    items=[
        Item(name="pepperoni pizza", portion="2 slices, 14 inch", carbs_g=72.4, fiber_g=4.1, fat_g=26, protein_g=24),
        Item(name="cola", portion="355 ml can", carbs_g=39, fiber_g=0, fat_g=0, protein_g=0),
    ],
    confidence="medium",
    assumptions="regular crust",
)
BAD_FIBRE = MealEstimate(
    items=[Item(name="oatmeal", portion="1 bowl", carbs_g=5, fiber_g=8, fat_g=3, protein_g=5)],
    confidence="medium",
    assumptions="",
)
OATMEAL = MealEstimate(
    items=[Item(name="oatmeal", portion="1 bowl", carbs_g=27, fiber_g=4, fat_g=3, protein_g=5)],
    confidence="high",
    assumptions="40 g dry oats",
)


class FakeEstimator:
    """Returns queued (estimate, stop_reason) pairs and records each call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, content, effort):
        self.calls.append((content, effort))
        return self.replies.pop(0)


# --- pure code ---------------------------------------------------------------

def test_totals_are_summed_in_code():
    n = to_nutrition(PIZZA)
    assert n.carbs_g == 111.4
    assert n.net_carbs_g == pytest.approx(107.3)
    assert n.fat_g == 26 and n.protein_g == 24
    assert n.items == ("pepperoni pizza (2 slices, 14 inch)", "cola (355 ml can)")


def test_no_items_means_low_confidence():
    n = to_nutrition(MealEstimate(items=[], confidence="high", assumptions="not food"))
    assert n.confidence == "low" and n.carbs_g == 0


def test_check_flags_impossible_and_implausible_estimates():
    assert check(PIZZA) == []
    assert "exceeds total carbs" in check(BAD_FIBRE)[0]
    huge = MealEstimate(
        items=[Item(name="cake", portion="whole", carbs_g=300, fiber_g=2, fat_g=90, protein_g=20),
               Item(name="juice", portion="2 l", carbs_g=200, fiber_g=0, fat_g=0, protein_g=0)],
        confidence="medium", assumptions="",
    )
    problems = check(huge)
    assert any("implausible for one item" in p for p in problems)
    assert any("meal total" in p for p in problems)
    assert check(MealEstimate(items=[], confidence="low", assumptions="")) == ["no food items found"]


def test_photo_is_sent_as_image(tmp_path):
    photo = tmp_path / "lunch.jpg"
    photo.write_bytes(b"\xff\xd8\xff fake jpeg")
    content = build_content(None, photo)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/jpeg"


def test_rejects_unsupported_files_and_empty_input(tmp_path):
    doc = tmp_path / "menu.pdf"
    doc.write_bytes(b"%PDF")
    with pytest.raises(ParseError):
        build_content(None, doc)
    with pytest.raises(ParseError):
        build_content(None, None)


# --- the graph -----------------------------------------------------------------

def test_clean_estimate_is_accepted_in_one_pass():
    fake = FakeEstimator((PIZZA, "end_turn"))
    n = ClaudeMealParser(fake).parse(text="2 slices pepperoni pizza and a coke")
    assert len(fake.calls) == 1 and fake.calls[0][1] == FIRST_EFFORT
    assert n.carbs_g == 111.4 and n.confidence == "medium"


def test_flagged_estimate_is_retried_once_at_higher_effort_with_the_problems():
    fake = FakeEstimator((BAD_FIBRE, "end_turn"), (OATMEAL, "end_turn"))
    n = ClaudeMealParser(fake).parse(text="bowl of oatmeal")
    assert [effort for _, effort in fake.calls] == [FIRST_EFFORT, RETRY_EFFORT]
    retry_text = json.dumps(fake.calls[1][0])
    assert "exceeds total carbs" in retry_text and "oatmeal" in retry_text
    assert n.carbs_g == 27 and n.confidence == "high" and "flagged" not in n.notes


def test_still_flagged_after_retry_is_kept_but_marked_low_confidence():
    fake = FakeEstimator((BAD_FIBRE, "end_turn"), (BAD_FIBRE, "end_turn"))
    n = ClaudeMealParser(fake).parse(text="bowl of oatmeal")
    assert len(fake.calls) == 2
    assert n.confidence == "low" and n.notes.startswith("flagged: oatmeal: fibre")


def test_refusal_and_truncation_raise():
    with pytest.raises(ParseError, match="declined"):
        ClaudeMealParser(FakeEstimator((None, "refusal"))).parse(text="x")
    with pytest.raises(ParseError, match="max_tokens"):
        ClaudeMealParser(FakeEstimator((None, "max_tokens"))).parse(text="x")


# --- the LangChain call, against a fake API response -----------------------------

def fake_api(monkeypatch, message: dict):
    """Patch the HTTP call under ChatAnthropic; everything above it runs for real."""
    payloads = []

    class Raw:
        def parse(self):
            return BetaMessage.model_validate(message)

    def create(self, payload):
        payloads.append(payload)
        return Raw()

    monkeypatch.setattr(nutrition._ChatAnthropic, "_create", create)
    return payloads


def api_message(content, stop_reason="end_turn"):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": MODEL,
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def test_langchain_request_shape_and_only_meal_content_sent(monkeypatch):
    payloads = fake_api(monkeypatch, api_message([
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": PIZZA.model_dump_json()},
    ]))
    estimate, stop_reason = langchain_estimator()(build_content("2 slices pepperoni pizza and a coke", None), "high")
    assert estimate == PIZZA and stop_reason == "end_turn"
    p = payloads[0]
    assert p["model"] == MODEL
    assert p["fallbacks"] == "default"
    assert p["betas"] == ["server-side-fallback-2026-07-01"]
    assert p["output_config"]["effort"] == "high"
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert "items" in p["output_config"]["format"]["schema"]["properties"]
    sent = json.dumps(p["messages"])
    assert "pepperoni" in sent
    assert "mg/dL" not in sent and "insulin" not in sent.lower()


def test_langchain_refusal_is_reported_not_parsed(monkeypatch):
    fake_api(monkeypatch, api_message([], stop_reason="refusal"))
    estimate, stop_reason = langchain_estimator()(build_content("x", None), FIRST_EFFORT)
    assert estimate is None and stop_reason == "refusal"


def test_missing_credentials_fail_at_parse_time_not_startup(monkeypatch):
    import anthropic

    def no_creds(*a, **k):
        raise anthropic.CredentialsError("none")

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", no_creds)
    parser = ClaudeMealParser()  # must not raise
    with pytest.raises(ParseError, match="no Anthropic credentials"):
        parser.parse(text="toast")
