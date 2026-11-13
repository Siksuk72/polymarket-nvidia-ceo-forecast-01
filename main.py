#!/usr/bin/env python3
# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).
#
# Forecast the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
# event at every minute of the 4 hours after a checkpoint, as five quantiles.
#
# Method (v2; validated on the six practice cases: pooled pinball loss
# 0.255535, vs v1's 0.287749 and the house baseline's 0.363589; it beats v1
# on every case. Per-case losses: p00 0.4224, p01 0.3114, p02 0.3004,
# p03 0.2461, p04 0.1976, p05 0.1106 (worst case 0.4224 vs v1's 0.4716).
# Median error 0.7344 (v1: 0.7675), 90% coverage 0.9100 (v1: 0.8944).
# Runtime 0.71 s for all six practice cases on one CPU (limit 8 s/case);
# two runs byte-identical; outputs valid (exact market set, 240 rows,
# 5 non-decreasing numbers in [0,1], max 290 KB per file).
#
#   * Location: persistence (the median forecast is the market's last price),
#     plus two small, history-confirmed drifts:
#       (a) down-continuation: if the price fell over the last 30 minutes,
#           it tends to keep falling a little; the drift is 0.1 times that
#           30-minute fall, growing like sqrt(h/30). Up-moves do not
#           continue (they revert), so only the down side is used.
#       (b) sum coherence: the 24 outcomes are mutually exclusive, so the
#           sum of Yes prices should be ~1. At the checkpoints it is not
#           (0.81-1.09), and that deviation predicts a correction: each
#           median is shifted 55% of the way toward x/S (S = the sum of
#           last prices), ramping in over the first 120 minutes.
#     Mean reversion was tested hard and rejected: pulling toward the
#     trailing mean, toward the cross-sectional level, or a level-OLS drift
#     all make the practice loss much worse (up to 0.55). Persistence plus
#     the two small drifts above is what survives.
#   * Spread: a 90/10 blend of two views of how far the price may move by
#     horizon h, both centred on the last price:
#       (a) a Laplace (double-exponential) with a per-market scale from an
#           EWMA of 1-minute changes (half-life 30 minutes, so the scale
#           tracks the current regime), floored at 0.001, capped at 0.01,
#           times 0.5, growing like sqrt(h). Laplace beats Gaussian here
#           (fatter tails; prices jump), and the 25%/75% shoulders are
#           shrunk by 0.9 (calibration showed the interquartile range too
#           wide). A flat 6-hour window sd and a term-structure sd were
#           both tried and are worse than the EWMA.
#       (b) the market's own empirical quantiles of h-minute price changes
#           over the last 6 hours (centred by subtracting their median),
#           at anchor horizons interpolated in log(h). Only 10% weight:
#           the empirical shape is noisy, and heavier weights are worse.
#   * Every row is clamped to [0, 1], non-decreasing, rounded to 6 decimals.
#
# The program is deterministic: only sorting and arithmetic, no randomness,
# no network, no clock. It reads $MAKEFUN_INPUT_DIR (else /input) and writes
# $MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per <id>.in, in name order.

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Laplace quantile factors (unit scale b), divided by sqrt(2) so that the
# scale below is a standard deviation, then with the 25%/75% shoulders
# shrunk by 0.9. Computed from log(2*tau) / sqrt(2) at build time.
Z = [
    math.log(0.1) / math.sqrt(2.0),
    0.9 * math.log(0.5) / math.sqrt(2.0),
    0.0,
    0.9 * -math.log(0.5) / math.sqrt(2.0),
    -math.log(0.1) / math.sqrt(2.0),
]
# Anchor horizons (minutes) at which the empirical change quantiles are taken.
ANCHORS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]
# How much recent history feeds the empirical quantiles, in minutes.
HISTORY_MINUTES = 360
# Blend weight of the empirical shape (the rest is the Laplace).
BLEND_W = 0.1
# Scale applied to the per-market Laplace sigma.
SIGMA_SCALE = 0.5
# Floor and cap on the per-market 1-minute change sd, in price units.
SIGMA_FLOOR = 0.001
SIGMA_CAP = 0.01
# Half-life, in minutes, of the EWMA that estimates the per-market sigma.
EWMA_HALF_LIFE = 30
# Down-continuation momentum: shrink factor and lookback window (minutes).
MOMENTUM_K = 0.1
MOMENTUM_WINDOW = 30
# Sum-coherence correction: shrink toward x/S, ramping over this many minutes.
SUM_NORM = 0.55
SUM_RAMP_MINUTES = 120


def quantile_sorted(xs, tau):
    """Linear-interpolation quantile of a sorted list."""
    n = len(xs)
    if n == 0:
        return 0.0
    i = tau * (n - 1)
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - i) + xs[hi] * (i - lo)


def interp_logh(xs, ys, h):
    """Linear interpolation in log(h) over the anchor points xs -> ys."""
    if h <= xs[0]:
        return ys[0]
    if h >= xs[-1]:
        return ys[-1]
    lh = math.log(h)
    for i in range(len(xs) - 1):
        if xs[i] <= h <= xs[i + 1]:
            l0 = math.log(xs[i])
            l1 = math.log(xs[i + 1])
            f = (lh - l0) / (l1 - l0) if l1 > l0 else 0.0
            return ys[i] * (1.0 - f) + ys[i + 1] * f
    return ys[-1]


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def market_empirical(prices):
    """Per-anchor empirical quantiles of h-minute changes, from the last
    HISTORY_MINUTES prices. Returns a list (one per anchor) of five quantiles.
    Anchors with too few samples give a zero row (a quiet market)."""
    seg = prices[-HISTORY_MINUTES:] if len(prices) > HISTORY_MINUTES else prices
    m = len(seg)
    out = []
    for h in ANCHORS:
        if m - h < 20:
            out.append([0.0] * 5)
            continue
        ch = sorted(seg[t + h] - seg[t] for t in range(m - h))
        out.append([quantile_sorted(ch, tau) for tau in QUANTILES])
    return out


def market_sigma(prices):
    """Per-market Laplace sigma: the recency-weighted (EWMA, half-life
    EWMA_HALF_LIFE) sd of 1-minute changes over the last HISTORY_MINUTES
    prices, floored, capped, and scaled."""
    seg = prices[-HISTORY_MINUTES:] if len(prices) > HISTORY_MINUTES else prices
    if len(seg) < 2:
        return SIGMA_FLOOR * SIGMA_SCALE
    d = [seg[t + 1] - seg[t] for t in range(len(seg) - 1)]
    lam = 0.5 ** (1.0 / EWMA_HALF_LIFE)
    # Weighted mean, most recent change first.
    w = 1.0
    sw = 0.0
    sm = 0.0
    for v in reversed(d):
        sw += w
        sm += w * v
        w *= lam
    mean = sm / sw
    # Weighted variance around that mean, same weights.
    w = 1.0
    sw = 0.0
    sv = 0.0
    for v in reversed(d):
        sw += w
        sv += w * (v - mean) ** 2
        w *= lam
    sd = math.sqrt(sv / sw)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sd)) * SIGMA_SCALE


def forecast_case(case):
    """The forecasts for one case: one entry per market, 240 rows each."""
    horizon = case["horizon_minutes"]
    markets = case["markets"]
    # The sum of last prices, for the sum-coherence correction.
    s0 = sum(m["prices"][-1] for m in markets)
    forecasts = []
    for m in markets:
        prices = m["prices"]
        if not prices:
            raise ValueError(f"market {m['id']} has no prices")
        x = prices[-1]
        emp = market_empirical(prices)
        sig = market_sigma(prices)
        # Down-continuation: the 30-minute fall, if any, shrunk by MOMENTUM_K.
        drift0 = 0.0
        if len(prices) > MOMENTUM_WINDOW:
            mv = prices[-1] - prices[-1 - MOMENTUM_WINDOW]
            if mv < 0:
                drift0 = MOMENTUM_K * mv
        rows = []
        for h in range(1, horizon + 1):
            sq = math.sqrt(h)
            # The empirical median change at this horizon: the blend is
            # centred on it, so the median forecast stays the last price.
            med = interp_logh(ANCHORS, [emp[a][2] for a in range(len(ANCHORS))], h)
            row = []
            for j in range(5):
                q = interp_logh(ANCHORS, [emp[a][j] for a in range(len(ANCHORS))], h)
                q = q - med  # centre on the last price: location is persistence
                g = Z[j] * sig * sq
                row.append(x + (1.0 - BLEND_W) * g + BLEND_W * q)
            # Down-continuation drift, growing like sqrt(h / window).
            if drift0:
                row = [v + drift0 * math.sqrt(h / MOMENTUM_WINDOW) for v in row]
            # Sum coherence: shift the whole row toward x / S, ramping in
            # over the first SUM_RAMP_MINUTES minutes.
            if s0 > 0:
                gf = min(1.0, h / SUM_RAMP_MINUTES)
                shift = x * (1.0 / s0 - 1.0) * SUM_NORM * gf
                row = [v + shift for v in row]
            # clamp to [0, 1], non-decreasing, 6 decimals
            row = [price6(v) for v in row]
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        forecasts.append({"id": m["id"], "q": rows})
    return {"forecasts": forecasts}


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        if len(case["quantiles"]) != len(QUANTILES) or any(
            t != q for t, q in zip(case["quantiles"], QUANTILES)
        ):
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, not {QUANTILES}"
            )
        forecasts = forecast_case(case)
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps(forecasts) + "\n")


if __name__ == "__main__":
    main()
