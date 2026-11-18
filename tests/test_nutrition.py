import json
from types import SimpleNamespace

import pytest

from aftertaste.nutrition import (
    MODEL, ClaudeMealParser, Item, MealEstimate, ParseError, build_content, to_nutrition,
)

PIZZA = MealEstimate(
    items=[
        Item(name="pepperoni pizza", portion="2 slices, 14 inch", carbs_g=72.4, fiber_g=4.1, fat_g=26, protein_g=24),
        Item(name="cola", portion="355 ml can", carbs_g=39, fiber_g=0, fat_g=0, protein_g=0),
    ],
    confidence="medium",
    assumptions="regular crust",
)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_client(parsed=PIZZA, stop_reason="end_turn"):
    messages = FakeMessages(SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason))
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_totals_are_summed_in_code():
    n = to_nutrition(PIZZA)
    assert n.carbs_g == 111.4
    assert n.net_carbs_g == pytest.approx(107.3)
    assert n.fat_g == 26 and n.protein_g == 24
    assert n.items == ("pepperoni pizza (2 slices, 14 inch)", "cola (355 ml can)")


def test_no_items_means_low_confidence():
    n = to_nutrition(MealEstimate(items=[], confidence="high", assumptions="not food"))
    assert n.confidence == "low" and n.carbs_g == 0


def test_request_shape_and_only_meal_content_sent():
    client, messages = fake_client()
    ClaudeMealParser(client).parse(text="2 slices pepperoni pizza and a coke")
    call = messages.calls[0]
    assert call["model"] == MODEL
    assert call["output_format"] is MealEstimate
    assert call["fallbacks"] == "default"
    sent = json.dumps(call["messages"])
    assert "pepperoni" in sent
    assert "mg/dL" not in sent and "insulin" not in sent.lower()


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


def test_refusal_and_truncation_raise():
    client, _ = fake_client(stop_reason="refusal")
    with pytest.raises(ParseError, match="declined"):
        ClaudeMealParser(client).parse(text="x")
    client, _ = fake_client(parsed=None, stop_reason="max_tokens")
    with pytest.raises(ParseError):
        ClaudeMealParser(client).parse(text="x")


def test_missing_credentials_fail_at_parse_time_not_startup(monkeypatch, tmp_path):
    import anthropic

    def no_creds(*a, **k):
        raise anthropic.CredentialsError("none")

    monkeypatch.setattr(anthropic, "Anthropic", no_creds)
    parser = ClaudeMealParser()  # must not raise
    with pytest.raises(ParseError, match="no Anthropic credentials"):
        parser.parse(text="toast")
