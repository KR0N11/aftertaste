import json

from fastapi.testclient import TestClient

from aftertaste.api import create_app
from aftertaste.cli import main
from aftertaste.models import Nutrition
from aftertaste.simulate import simulate
from aftertaste.store import Store


class FakeParser:
    def parse(self, text=None, photo=None):
        return Nutrition(60, 5, 15, 20, "medium", ("test meal (1 plate)",))


def seeded_store():
    store = Store(":memory:")
    readings, doses, meals = simulate(days=40, seed=11)
    store.add_readings(readings)
    store.add_doses(doses)
    for m in meals:
        store.add_meal(m)
    return store, readings


def test_log_meal_returns_measured_and_expected_rise():
    store, readings = seeded_store()
    client = TestClient(create_app(store, FakeParser()))
    # a time inside the synthetic trace, between two simulated meals
    when = readings[len(readings) // 2].time.isoformat()
    r = client.post("/meals", data={"eaten_at": when, "description": "test meal"})
    assert r.status_code == 201
    body = r.json()
    assert body["nutrition"]["carbs_g"] == 60
    assert body["measured"] is not None
    assert isinstance(body["expected_peak_rise_mgdl"], int)


def test_rejects_times_without_offset_and_empty_meals():
    client = TestClient(create_app(Store(":memory:"), FakeParser()))
    assert client.post("/meals", data={"eaten_at": "2026-03-02T12:00:00", "description": "x"}).status_code == 422
    assert client.post("/meals", data={"eaten_at": "2026-03-02T12:00:00-05:00"}).status_code == 422


def test_insights_and_unconfigured_dexcom():
    store, _ = seeded_store()
    client = TestClient(create_app(store, FakeParser()))
    body = client.get("/insights").json()
    assert body["evaluation"]["n_test"] >= 8
    assert {e["feature"] for e in body["effects"]} >= {"net carbs", "fat", "meal bolus"}
    assert client.post("/sync").status_code == 503
    assert client.get("/dexcom/login").status_code == 503


def test_no_dose_suggestion_endpoint_exists():
    app = create_app(Store(":memory:"), FakeParser())
    paths = {route.path for route in app.routes}
    assert not any("dose" in p or "bolus" in p for p in paths)


def test_cli_demo_and_import(tmp_path, capsys):
    assert main(["--db", str(tmp_path / "demo.db"), "demo", "--days", "30"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out
    db = tmp_path / "real.db"
    fixture = __file__.replace("test_api_cli.py", "fixtures/clarity_mmol.csv")
    assert main(["--db", str(db), "import-clarity", fixture]) == 0
    assert json.loads(capsys.readouterr().out) == {"readings": 3, "insulin_doses": 2}
