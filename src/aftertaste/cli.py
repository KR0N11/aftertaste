"""Command line: aftertaste <command> ...  (run `aftertaste -h`)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_DB = "data/aftertaste.db"
DEFAULT_TZ = "America/Toronto"


def _when(s: str | None, tz: str) -> datetime:
    if not s:
        return datetime.now(ZoneInfo(tz))
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(tz))


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aftertaste", description="Meal -> glucose insight. No dosing advice.")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--tz", default=DEFAULT_TZ, help="timezone for times without an offset")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("import-clarity", help="import a Dexcom Clarity CSV export")
    c.add_argument("csv")

    c = sub.add_parser("log", help="log a meal from text and/or a photo")
    c.add_argument("text", nargs="?")
    c.add_argument("--photo")
    c.add_argument("--at", help="when you ate, e.g. 2026-11-18T12:30 (default: now)")

    sub.add_parser("meals", help="list meals with measured and expected rises")
    sub.add_parser("insights", help="evaluate the model and show what drives your rises")

    c = sub.add_parser("sync", help="pull glucose and insulin from the Dexcom API")
    c.add_argument("--days", type=int, default=30)

    c = sub.add_parser("carb-eval", help="score carb estimates against a labelled CSV")
    c.add_argument("csv")
    c.add_argument("--out", default="eval/results")

    c = sub.add_parser("demo", help="fill a database with SYNTHETIC data and evaluate")
    c.add_argument("--days", type=int, default=60)

    c = sub.add_parser("serve", help="run the HTTP API")
    c.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    from .store import Store  # imported late so `-h` stays fast

    if args.cmd == "demo":
        from .simulate import simulate
        from . import service

        db = args.db if args.db != DEFAULT_DB else "data/demo.db"
        store = Store(db)
        readings, doses, meals = simulate(days=args.days)
        store.add_readings(readings)
        store.add_doses(doses)
        for m in meals:
            store.add_meal(m)
        print(f"SYNTHETIC data written to {db}. These numbers describe the simulator, not a person.")
        _print(service.insights(store))
        return 0

    store = Store(args.db)

    if args.cmd == "import-clarity":
        from .clarity import import_clarity_csv

        readings, doses = import_clarity_csv(args.csv, tz=args.tz)
        store.add_readings(readings)
        store.add_doses(doses)
        _print({"readings": len(readings), "insulin_doses": len(doses)})
    elif args.cmd == "log":
        from .nutrition import ClaudeMealParser, ParseError
        from . import service

        if not args.text and not args.photo:
            p.error("log needs a description, --photo, or both")
        try:
            meal = service.log_meal(store, ClaudeMealParser(), _when(args.at, args.tz), args.text, args.photo)
        except ParseError as e:
            print(f"could not estimate this meal: {e}", file=sys.stderr)
            return 1
        _print(service.meal_report(store, meal))
    elif args.cmd == "meals":
        from . import service

        _print([service.meal_report(store, m) for m in store.meals()])
    elif args.cmd == "insights":
        from . import service

        _print(service.insights(store))
    elif args.cmd == "sync":
        from .dexcom import DexcomClient
        from . import service

        _print(service.sync_dexcom(store, DexcomClient.from_env(), days=args.days))
    elif args.cmd == "carb-eval":
        from .carb_eval import load_cases, run, summarize, write_report
        from .nutrition import ClaudeMealParser

        summary = summarize(run(load_cases(args.csv), ClaudeMealParser()))
        write_report(summary, args.out)
        _print(summary)
    elif args.cmd == "serve":
        import uvicorn

        from .api import create_app
        from .dexcom import DexcomClient, DexcomError
        from .nutrition import ClaudeMealParser

        try:
            dexcom = DexcomClient.from_env()
        except DexcomError:
            dexcom = None
        uvicorn.run(create_app(store, ClaudeMealParser(), dexcom), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
