"""Dexcom API v3 client: OAuth 2.0 + glucose (EGV) and insulin events.

Reference: developer.dexcom.com (Authentication, Endpoint Overview, egvs,
events). Facts this code relies on:
  * login  GET  {base}/v3/oauth2/login   (response_type=code, scope=offline_access)
  * token  POST {base}/v3/oauth2/token   (form-encoded)
  * access tokens last 2 h; each refresh token works ONCE, so the new one
    is saved before anything else happens
  * data   GET  {base}/v3/users/self/egvs | /events, startDate/endDate in UTC,
    start inclusive, end exclusive, at most 30 days per request
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode

import httpx

from .models import GlucoseReading, InsulinDose, InsulinKind, to_mgdl, utc

BASE_URLS = {
    "us": "https://api.dexcom.com",
    "ous": "https://api.dexcom.eu",  # Dexcom: "EU and outside US"
    "jp": "https://api.dexcom.jp",
    "sandbox": "https://sandbox-api.dexcom.com",
}
MAX_WINDOW = timedelta(days=30)
DATE_FMT = "%Y-%m-%dT%H:%M:%S"


class DexcomError(RuntimeError):
    pass


@dataclass
class Token:
    access_token: str
    refresh_token: str
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at - 60

    @classmethod
    def from_response(cls, data: dict) -> "Token":
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + float(data.get("expires_in", 7200)),
        )


class TokenFile:
    """Token persisted as JSON (git-ignored). Written with 0600 permissions."""

    def __init__(self, path: str | Path = "data/dexcom_token.json") -> None:
        self.path = Path(path)

    def load(self) -> Token | None:
        if not self.path.exists():
            return None
        return Token(**json.loads(self.path.read_text()))

    def save(self, token: Token) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(token.__dict__))
        os.chmod(self.path, 0o600)


class DexcomClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        region: str = "sandbox",
        tokens: TokenFile | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        if region not in BASE_URLS:
            raise ValueError(f"region must be one of {sorted(BASE_URLS)}")
        self.base = BASE_URLS[region]
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.tokens = tokens or TokenFile()
        self.http = http or httpx.Client(timeout=30)

    @classmethod
    def from_env(cls, **kwargs) -> "DexcomClient":
        try:
            return cls(
                client_id=os.environ["DEXCOM_CLIENT_ID"],
                client_secret=os.environ["DEXCOM_CLIENT_SECRET"],
                redirect_uri=os.environ.get("DEXCOM_REDIRECT_URI", "http://localhost:8000/dexcom/callback"),
                region=os.environ.get("DEXCOM_REGION", "sandbox"),
                **kwargs,
            )
        except KeyError as e:
            raise DexcomError(f"missing environment variable {e.args[0]}") from None

    # OAuth ----------------------------------------------------------------
    def login_url(self, state: str | None = None) -> tuple[str, str]:
        state = state or secrets.token_urlsafe(16)
        query = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "response_type": "code",
                "scope": "offline_access",
                "state": state,
            }
        )
        return f"{self.base}/v3/oauth2/login?{query}", state

    def _token_request(self, form: dict) -> Token:
        resp = self.http.post(
            f"{self.base}/v3/oauth2/token",
            data={**form, "client_id": self.client_id, "client_secret": self.client_secret},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if resp.status_code != 200:
            raise DexcomError(f"token request failed: HTTP {resp.status_code}")
        token = Token.from_response(resp.json())
        self.tokens.save(token)  # refresh tokens are single-use: save first
        return token

    def exchange_code(self, code: str) -> Token:
        return self._token_request(
            {"grant_type": "authorization_code", "code": code, "redirect_uri": self.redirect_uri}
        )

    def _access_token(self) -> str:
        token = self.tokens.load()
        if token is None:
            raise DexcomError("not connected: run `aftertaste dexcom-login` first")
        if token.expired:
            token = self._token_request(
                {"grant_type": "refresh_token", "refresh_token": token.refresh_token}
            )
        return token.access_token

    # data -----------------------------------------------------------------
    def _get_records(self, path: str, start: datetime, end: datetime) -> Iterator[dict]:
        for lo, hi in windows(start, end):
            resp = self.http.get(
                f"{self.base}/v3/users/self/{path}",
                params={"startDate": lo.strftime(DATE_FMT), "endDate": hi.strftime(DATE_FMT)},
                headers={"Authorization": f"Bearer {self._access_token()}"},
            )
            if resp.status_code != 200:
                raise DexcomError(f"GET {path} failed: HTTP {resp.status_code}")
            yield from resp.json().get("records", [])

    def glucose(self, start: datetime, end: datetime) -> list[GlucoseReading]:
        return parse_egvs(self._get_records("egvs", start, end))

    def insulin(self, start: datetime, end: datetime) -> list[InsulinDose]:
        return parse_insulin_events(self._get_records("events", start, end))


def windows(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Split [start, end) into chunks Dexcom accepts (30 days max), in UTC."""
    lo, end = utc(start), utc(end)
    while lo < end:
        hi = min(lo + MAX_WINDOW, end)
        yield lo, hi
        lo = hi


def _parse_time(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:  # systemTime is UTC per Dexcom docs
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_egvs(records) -> list[GlucoseReading]:
    out = []
    for r in records:
        if r.get("value") is None:  # sensor gap; never invent a value
            continue
        out.append(
            GlucoseReading(
                time=_parse_time(r["systemTime"]),
                mgdl=to_mgdl(r["value"], r.get("unit") or "mg/dL"),
                source_id=r.get("recordId"),
            )
        )
    return sorted(out, key=lambda g: g.time)


def parse_insulin_events(records) -> list[InsulinDose]:
    kinds = {"fastActing": InsulinKind.RAPID, "longActing": InsulinKind.LONG}
    out = []
    for r in records:
        if r.get("eventType") != "insulin" or r.get("eventStatus") == "deleted":
            continue
        kind = kinds.get(r.get("eventSubType") or "")
        if kind is None:
            continue
        try:
            units = float(r["value"])
        except (TypeError, ValueError):
            continue
        out.append(InsulinDose(_parse_time(r["systemTime"]), units, kind, r.get("recordId")))
    return sorted(out, key=lambda d: d.time)
