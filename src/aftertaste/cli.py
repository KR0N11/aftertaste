"""Command line: aftertaste <command> ...  (run `aftertaste -h`)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_DB = "data/aftertaste.db"
DEFAULT_TZ = "America/Toronto"
ICLOUD_MEALS = "~/Library/Mobile Documents/com~apple~CloudDocs/Aftertaste/Meals"


def _when(s: str | None, tz: str) -> datetime:
    if not s:
        return datetime.now(ZoneInfo(tz))
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(tz))


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _summary(report: dict) -> str:
    if report.get("problems"):
        return "Needs attention: " + report["problems"][0]
    photos = report["photos"]
    parts = [f"{len(photos['imported'])} meal(s) logged"]
    if photos["failed"]:
        parts.append(f"{len(photos['failed'])} photo(s) failed")
    if report["clarity"]["files"]:
        parts.append(f"{report['clarity']['readings']} glucose readings imported")
    ev = report["insights"].get("evaluation")
    if ev:
        parts.append(f"model {ev['improvement_pct']}% vs carbs-only on {ev['n_test']} meals")
    else:
        parts.append(f"{report['insights']['clean_meals']} clean meals so far")
    return ", ".join(parts)


def _notify(text: str) -> None:
    import subprocess

    script = f'display notification {json.dumps(text)} with title "Aftertaste"'
    subprocess.run(["osascript", "-e", script], check=False)


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

    c = sub.add_parser("import-photos", help="log every new meal photo in a folder (time = when taken)")
    c.add_argument("folder", nargs="?", default=ICLOUD_MEALS)

    c = sub.add_parser("daily", help="scheduled job: new Clarity exports + new meal photos + insights")
    c.add_argument("--photos", default=ICLOUD_MEALS)
    c.add_argument("--clarity", default="~/Downloads", help="folder to look for Clarity CSV exports")
    c.add_argument("--notify", action="store_true", help="macOS notification with the summary")

    sub.add_parser("meals", help="list meals with measured and expected rises")
    sub.add_parser("insights", help="evaluate the model and show what drives your rises")

    c = sub.add_parser("sync", help="pull glucose and insulin from the Dexcom API")
    c.add_argument("--days", type=int, default=30)

    c = sub.add_parser("carb-eval", help="score carb estimates against a labelled CSV")
    c.add_argument("csv")
    c.add_argument("--out", default="eval/results")

    c = sub.add_parser("cgmacros-eval", help="glucose-rise prediction on the public CGMacros dataset")
    c.add_argument("root", help="folder containing CGMacros-0XX folders and bio.csv")
    c.add_argument("--out", default="eval/results")

    c = sub.add_parser("cgmacros-labels", help="write a photo-only carb eval CSV from CGMacros")
    c.add_argument("root")
    c.add_argument("--out", default="data/cgmacros/labels.csv")

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

    if args.cmd == "cgmacros-eval":
        from pathlib import Path

        from .cgmacros import evaluate_personalized

        result = evaluate_personalized(args.root)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "cgmacros.json").write_text(json.dumps(result, indent=2) + "\n")
        _print(result)
        return 0
    if args.cmd == "cgmacros-labels":
        from .cgmacros import write_carb_labels

        print(f"{write_carb_labels(args.root, args.out)} photo-labelled meals written to {args.out}")
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
    elif args.cmd == "import-photos":
        from pathlib import Path

        from .nutrition import ClaudeMealParser
        from . import service

        _print(service.import_photos(store, ClaudeMealParser(), Path(args.folder).expanduser(), args.tz))
    elif args.cmd == "daily":
        from pathlib import Path

        from .nutrition import ClaudeMealParser
        from . import service

        report = service.daily(
            store, ClaudeMealParser(), Path(args.photos).expanduser(), Path(args.clarity).expanduser(), args.tz
        )
        out = Path("data/reports") / f"{datetime.now(ZoneInfo(args.tz)):%Y-%m-%d}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        _print(report)
        if args.notify:
            _notify(_summary(report))
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
