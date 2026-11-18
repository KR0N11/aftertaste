"""HTTP API. Run with `aftertaste serve`.

There is deliberately no "what if I take N units" endpoint: expected rises
are only shown for meals already eaten, with the insulin actually taken.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse

from . import service
from .dexcom import DexcomClient, DexcomError
from .nutrition import MealParser, ParseError
from .store import Store

PHOTO_DIR = Path("data/photos")


def create_app(store: Store, parser: MealParser, dexcom: DexcomClient | None = None) -> FastAPI:
    app = FastAPI(title="Aftertaste", description="Insight only. No dosing advice.")
    pending_states: set[str] = set()

    @app.post("/meals", status_code=201)
    async def add_meal(
        eaten_at: datetime = Form(..., description="ISO 8601 with offset, e.g. 2026-11-18T12:30:00-05:00"),
        description: str | None = Form(None),
        photo: UploadFile | None = File(None),
    ):
        if eaten_at.tzinfo is None:
            raise HTTPException(422, "eaten_at needs a UTC offset")
        if not description and photo is None:
            raise HTTPException(422, "send a description, a photo, or both")
        saved = None
        if photo is not None:
            PHOTO_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(photo.filename or "meal.jpg").suffix.lower() or ".jpg"
            with tempfile.NamedTemporaryFile(dir=PHOTO_DIR, suffix=suffix, delete=False) as out:
                shutil.copyfileobj(photo.file, out)
                saved = out.name
        try:
            meal = service.log_meal(store, parser, eaten_at, text=description, photo=saved)
        except ParseError as e:
            raise HTTPException(422, str(e))
        return service.meal_report(store, meal)

    @app.get("/meals")
    def list_meals():
        return [service.meal_report(store, m) for m in store.meals()]

    @app.get("/meals/{meal_id}")
    def get_meal(meal_id: int):
        meal = store.meal(meal_id)
        if meal is None:
            raise HTTPException(404, "no such meal")
        return service.meal_report(store, meal)

    @app.get("/insights")
    def get_insights():
        return service.insights(store)

    @app.get("/dexcom/login")
    def dexcom_login():
        if dexcom is None:
            raise HTTPException(503, "Dexcom is not configured")
        url, state = dexcom.login_url()
        pending_states.add(state)
        return RedirectResponse(url)

    @app.get("/dexcom/callback")
    def dexcom_callback(code: str, state: str):
        if dexcom is None:
            raise HTTPException(503, "Dexcom is not configured")
        if state not in pending_states:
            raise HTTPException(400, "unknown OAuth state")
        pending_states.discard(state)
        try:
            dexcom.exchange_code(code)
        except DexcomError as e:
            raise HTTPException(502, str(e))
        return {"connected": True}

    @app.post("/sync")
    def sync(days: int = 30):
        if dexcom is None:
            raise HTTPException(503, "Dexcom is not configured")
        try:
            return service.sync_dexcom(store, dexcom, days=days)
        except DexcomError as e:
            raise HTTPException(502, str(e))

    return app
