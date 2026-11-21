# Carb estimation eval

- Model: qwen3-vl:8b-instruct (sample: 200 of csv, seed 0)
- Meals: 200 (0 failed to parse, counted below as unscored)
- Mean absolute error: 32.7 g
- Median absolute error: 28.5 g
- Within ±10 g: 23.5%

| Confidence | Meals | MAE (g) |
|---|---|---|
| high | 24 | 23.0 |
| medium | 165 | 32.5 |
| low | 11 | 56.1 |

Worst 5:

- CGMacros-004-20230914T1954: true 204.0 g, estimated 12.0 g (medium)
- CGMacros-011-20201008T1750: true 163.0 g, estimated 24.0 g (high)
- CGMacros-001-20200502T1637: true 9.0 g, estimated 130.0 g (medium)
- CGMacros-033-20220417T1918: true 92.0 g, estimated 0.7 g (medium)
- CGMacros-008-20230208T1812: true 117.0 g, estimated 37.0 g (medium)
