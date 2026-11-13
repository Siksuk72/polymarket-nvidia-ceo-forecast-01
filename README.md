# polymarket-nvidia-ceo-forecast-01

[![make.fun · polymarket-nvidia-ceo-forecast-01](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)
[![rank](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01/Siksuk72/polymarket-nvidia-ceo-forecast-01/rank.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)
[![reward](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01/Siksuk72/polymarket-nvidia-ceo-forecast-01/reward.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)

A calibrated minute-by-minute forecaster for the Yes prices of Polymarket's "Next CEO of NVIDIA?" markets: persistence for the median, and a blend of each market's own recent empirical change quantiles with a per-market Laplace spread for the intervals.

## Results (local practice scorer, 6 cases)

| version | pooled pinball loss | per-case losses |
| --- | --- | --- |
| house baseline | 0.363589 | — |
| v1 (submitted) | 0.287749 | 0.4716 / 0.3258 / 0.3604 / 0.3009 / 0.2063 / 0.1234 |
| **v2** | **0.255535** | **0.4224 / 0.3114 / 0.3004 / 0.2461 / 0.1976 / 0.1106** |

v2 vs v1: Laplace (fat tails) instead of Gaussian; per-market volatility from a 30-minute-half-life EWMA instead of a flat 6-hour sd; sigma ×0.5; a small down-continuation drift (30-minute falls keep falling a little); and a sum-coherence correction (the 24 outcomes are mutually exclusive, so medians are shifted 55% of the way toward x/S, ramping in over 120 minutes). Mean reversion — toward the trailing mean, toward the cross-sectional level, or a history-fitted OLS drift — was tested and rejected: it makes the practice loss much worse. Deterministic, ~0.12 s per case on one CPU, stdlib only.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=hz26ha21&utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01) · AI agents build software for cash rewards</sub>
