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
meal photo / text ──► Claude ──► per-item carbs, fibre, fat, protein ──► totals summed in code
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
| Meal parsing (Claude, structured output) | built, tested against a fake client; live run needs an API key |
| Dexcom API v3 OAuth + sync, Clarity CSV import (mg/dL and mmol/L) | built, tested with mocked HTTP and a fixture export |
| Response maths, IOB, features, chronological eval | built, tested on known traces |
| HTTP API + CLI | built, tested; server run end to end on demo data |
| Carb estimation eval on a public labelled dataset | harness built; results pending |
| My own data | not collected yet |

Results are added here only when measured. The simulator numbers below are
not results.

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

## Tests

```sh
uv run pytest
```

## License

MIT for the code. Public datasets used for evaluation keep their own licences
and are not redistributed here.
