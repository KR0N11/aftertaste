# Aftertaste

What will this meal do to my glucose?

I have type 1 diabetes. Carb counting treats every 60 g of carbs the same, but
a bowl of rice at dinner and a slice of pizza at lunch do very different things
to me. Aftertaste learns *my* responses: log a meal as a photo or a sentence,
and it lines the meal up with my Dexcom CGM trace and the insulin I took,
measures what actually happened, and learns what drives the rise.

**Insight only.** It never suggests an insulin dose, and there is no endpoint
that takes one as input. Talk to your care team before changing anything.

## How it works

```
meal photo / text ──► Claude (LangGraph) ──► per-item carbs, fibre, fat, protein ──► totals summed in code
                                                                              │
Dexcom API v3 / Clarity CSV ──► glucose readings + insulin doses ─────────────┤
                                                                              ▼
                       per-meal response (baseline, peak rise, time to peak, 2 h iAUC)
                                                                              ▼
              ridge model vs carbs-only baseline, tested on later meals only
```

- **The LLM only reads food.** Claude gets the meal photo and/or description,
  nothing else: no glucose, no insulin, nothing about me. It returns a
  validated schema of items; totals are added in code, so no reported number
  comes from model arithmetic. Low confidence is an allowed answer.
- **Meal parsing is a small LangGraph workflow.** `estimate` calls Claude
  through LangChain (`ChatAnthropic`, JSON-schema structured output);
  `check` is plain code that flags impossible or implausible estimates (fibre
  above total carbs, over 250 g carbs in one item or 400 g in a meal, no items,
  low confidence). A flagged first pass is re-run once at higher effort with
  the problems listed; if it is still flagged, the meal is kept but marked low
  confidence.

  ```
  estimate ──► check ──► done
     ▲           │
     └── retry ◄─┘  (flagged, first pass only)
  ```
- **All glucose maths is deterministic and tested.** A meal whose trace has a
  gap over 20 min in the first 2 h is marked invalid with a reason, never
  scored as "no rise".
- **Insulin on board** uses the exponential action curve from the open-source
  Loop project (peak 75 min, 6 h duration by default) as a feature only.
- **Evaluation is chronological.** Train on earlier meals, test on later ones,
  never shuffled. The headline is improvement over a carbs-only baseline with
  a paired bootstrap 95% interval, because one person's meal count is small.
- **Privacy.** Data stays in a local SQLite file. Dexcom tokens are saved with
  0600 permissions; refresh tokens are single-use, so the new one is saved
  before anything else runs.

## Status

| Part | State |
|---|---|
| Meal parsing (Claude via LangChain + LangGraph, structured output, check-and-retry) | built, tested with a fake model and a fake API response through LangChain; live run needs an API key |
| Dexcom API v3 OAuth + sync, Clarity CSV import (mg/dL and mmol/L) | built, tested with mocked HTTP and a fixture export |
| Response maths, IOB, features, chronological eval | built, tested on known traces |
| HTTP API + CLI | built, tested; server run end to end on demo data |
| Glucose-rise prediction on public data (CGMacros) | **measured**, see Results |
| Carb estimation from photos (1,084 labelled CGMacros photos) | harness and labels ready; needs an API key to run |
| Daily import (iCloud photos with EXIF time, Clarity exports) | built, tested, scheduled |
| My own data | not collected yet |

Results are added here only when measured. The simulator numbers further down
are not results.

## Results

### Glucose-rise prediction on public data

[CGMacros](https://doi.org/10.13026/3z8q-x658) (PhysioNet, CC BY-NC-SA 4.0):
adults wearing a Dexcom G6 with labelled meals. **Nobody in it has type 1
diabetes or takes insulin**, so this tests the method, not type 1.

Only fully eaten meals (`Amount Consumed` = 100) with a clean trace and no
other meal within 2 h are used: 840 of 1,706 meal rows, 35 people. Each
person's later 25% of meals are held out; nothing from them is used in
training. Reproduce with `aftertaste cgmacros-eval <folder>`
([eval/results/cgmacros.json](eval/results/cgmacros.json)).

| Predictor, 210 held-out meals | Peak-rise error (MAE) |
|---|---|
| Shared carbs-only (carb counting) | 30.5 mg/dL |
| Shared model (carbs, fat, protein, fibre, timing) | 30.9 mg/dL |
| Personal carbs-only (one line per person) | 26.7 mg/dL |
| **Personalized model** | **26.1 mg/dL** |

- Personalized model vs carb counting: **14.6% lower error** (95% bootstrap CI 4.7% to 24.2%).
- Personalized model vs personal carbs-only: 2.3% lower (CI -7.9% to 11.8%), **not
  a real difference**.

What this says: the gain comes from learning the person (their typical rise
per gram, from earlier meals only, leave-one-out, plus A1c). Adding fat,
protein and fibre on top barely helps here. A model shared across people
did not beat carb counting at all, which was the first thing tried; the
personal features were added after that, and both results are reported.

### Pipeline check on synthetic data (not a result)

`aftertaste demo` generates a synthetic type 1 patient where fat slows
absorption and mornings spike harder, then evaluates. The model should find
those effects and beat carbs-only, and it does (60 days, 40 held-out meals:
mean absolute error 16.2 vs 31.6 mg/dL). With ±15 g noise added to the carb
estimates, the advantage shrinks and its interval crosses zero, which is why
the carb eval matters. This describes the simulator, not a person.

## Use

```sh
uv sync
uv run aftertaste import-clarity export.csv          # Clarity -> Export, times in America/Toronto by default
uv run aftertaste log "2 slices pepperoni pizza + coke" --at 2026-11-18T12:30
uv run aftertaste log --photo lunch.jpg
uv run aftertaste meals                               # measured vs expected rise per meal
uv run aftertaste insights                            # evaluation + what drives your rises
uv run aftertaste serve                               # HTTP API on 127.0.0.1:8000 (/docs)
uv run aftertaste demo                                # synthetic data, for trying it out
uv run aftertaste carb-eval labels.csv                # id,description,photo,true_carbs_g
```

Claude credentials come from `ANTHROPIC_API_KEY` or `ant auth login`. Dexcom
needs a developer app: `DEXCOM_CLIENT_ID`, `DEXCOM_CLIENT_SECRET`,
`DEXCOM_REDIRECT_URI`, `DEXCOM_REGION` (`sandbox`, `us`, `ous` for outside the
US, `jp`), then open `/dexcom/login` on the running server.

## Daily workflow

1. On the phone, save each meal photo to **Files > iCloud Drive > Aftertaste > Meals**
   before eating. The meal time is read from the photo itself, not the upload.
2. Now and then, export from Dexcom Clarity (Export, CSV) into `~/Downloads`.
3. Every night at 22:30 a launchd job runs `aftertaste daily --notify`: it imports
   new Clarity exports and new photos (each file once, by hash), retries photos
   that failed, writes `data/reports/<date>.json` and shows a notification.

Install: copy `scripts/aftertaste.daily.plist.example` to
`~/Library/LaunchAgents/`, replace `$HOME`, then
`launchctl bootstrap gui/$(id -u) <plist>`. The job runs `uv` directly so only
that one program needs folder access; if macOS blocks Downloads or iCloud
Drive, the notification says so instead of silently finding nothing.

## Tests

```sh
uv run pytest
```

## License

MIT for the code. Public datasets used for evaluation keep their own licences
and are not redistributed here.
