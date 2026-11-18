import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from aftertaste.dexcom import (
    DexcomClient, DexcomError, Token, TokenFile, parse_egvs, parse_insulin_events, windows,
)
from aftertaste.models import InsulinKind


def test_windows_never_exceed_30_days():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    chunks = list(windows(start, start + timedelta(days=75)))
    assert len(chunks) == 3
    assert all(hi - lo <= timedelta(days=30) for lo, hi in chunks)
    assert chunks[0][0] == start and chunks[-1][1] == start + timedelta(days=75)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))


def test_parse_egvs_skips_gaps_and_converts_mmol():
    records = [
        {"recordId": "a", "systemTime": "2025-01-30T23:49:55Z", "value": 120, "unit": "mg/dL"},
        {"recordId": "b", "systemTime": "2025-01-30T23:54:55Z", "value": None, "unit": "mg/dL"},
        {"recordId": "c", "systemTime": "2025-01-30T23:59:55", "value": 7, "unit": "mmol/L"},
    ]
    out = parse_egvs(records)
    assert [r.source_id for r in out] == ["a", "c"]
    assert out[1].mgdl == pytest.approx(7 * 18.0182)
    assert out[1].time.tzinfo is not None


def test_parse_insulin_events():
    records = [
        {"recordId": "1", "systemTime": "2025-02-06T09:12:35+00:00", "eventType": "insulin",
         "eventSubType": "fastActing", "value": "4.5", "unit": "units", "eventStatus": "created"},
        {"recordId": "2", "systemTime": "2025-02-06T21:00:00+00:00", "eventType": "insulin",
         "eventSubType": "longActing", "value": "18", "unit": "units", "eventStatus": "created"},
        {"recordId": "3", "systemTime": "2025-02-06T10:00:00+00:00", "eventType": "insulin",
         "eventSubType": "fastActing", "value": "3", "unit": "units", "eventStatus": "deleted"},
        {"recordId": "4", "systemTime": "2025-02-06T10:00:00+00:00", "eventType": "carbs",
         "eventSubType": None, "value": "40", "unit": "grams", "eventStatus": "created"},
    ]
    out = parse_insulin_events(records)
    assert [(d.source_id, d.units, d.kind) for d in out] == [
        ("1", 4.5, InsulinKind.RAPID), ("2", 18.0, InsulinKind.LONG),
    ]


def test_login_url(tmp_path):
    c = DexcomClient("cid", "secret", "http://localhost:8000/cb", region="sandbox",
                     tokens=TokenFile(tmp_path / "t.json"))
    url, state = c.login_url()
    q = parse_qs(urlparse(url).query)
    assert url.startswith("https://sandbox-api.dexcom.com/v3/oauth2/login?")
    assert q["scope"] == ["offline_access"] and q["response_type"] == ["code"]
    assert q["state"] == [state]


def test_expired_token_is_refreshed_and_saved_before_use(tmp_path):
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        if request.url.path == "/v3/oauth2/token":
            form = parse_qs(request.content.decode())
            assert form["grant_type"] == ["refresh_token"]
            assert form["refresh_token"] == ["old-refresh"]
            return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh",
                                             "expires_in": 7200, "token_type": "Bearer"})
        assert request.headers["Authorization"] == "Bearer new-access"
        assert "startDate" in request.url.params
        return httpx.Response(200, json={"recordType": "egv", "records": [
            {"recordId": "x", "systemTime": "2026-01-01T00:00:00Z", "value": 99, "unit": "mg/dL"}]})

    tokens = TokenFile(tmp_path / "t.json")
    tokens.save(Token("old-access", "old-refresh", time.time() - 10))
    c = DexcomClient("cid", "secret", "http://cb", tokens=tokens,
                     http=httpx.Client(transport=httpx.MockTransport(handler)))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    readings = c.glucose(start, start + timedelta(days=1))
    assert [r.mgdl for r in readings] == [99]
    saved = json.loads((tmp_path / "t.json").read_text())
    assert saved["refresh_token"] == "new-refresh"  # the single-use token was replaced


def test_not_connected_is_a_clear_error(tmp_path):
    c = DexcomClient("cid", "secret", "http://cb", tokens=TokenFile(tmp_path / "none.json"),
                     http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(DexcomError, match="not connected"):
        c.glucose(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))
