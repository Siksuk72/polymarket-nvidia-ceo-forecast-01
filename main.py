#!/usr/bin/env python3
# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).
#
# Forecast the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
# event at every minute of the 4 hours after a checkpoint, as five quantiles.
#
# Method (validated on the six practice cases: pooled pinball loss 0.287749,
# vs the house baseline's 0.363589; it beats the baseline on every case):
#
#   * Location: persistence. The median forecast is the market's last price.
#     Trailing means are far worse as a location (a 24h mean's median error is
#     ~4x persistence's), and a mean-reversion pull toward the trailing mean
#     hurts too: these prices do not revert to their recent mean over 4 hours.
#   * Spread: a 50/50 blend of two views of how far the price may move by
#     horizon h, both centred on the last price:
#       (a) the market's own empirical quantiles of h-minute price changes,
#           taken over the last 6 hours of its history (the recent regime
#           predicts the next 4 hours far better than the full 72 hours,
#           whose big early moves would make the intervals far too wide),
#           centred by subtracting their median, at anchor horizons and
#           interpolated in log(h). This carries the real shape: long flats
#           (a big atom at zero change), then ticks, and fat, asymmetric
#           tails (down moves are larger than up moves at long horizons).
#       (b) a Gaussian with a per-market sigma: 0.7 times the market's
#           recent 1-minute change sd over the last 6 hours (floored at
#           0.0005, capped at 0.01), growing like sqrt(h). This keeps each
#           market's width tied to how jumpy it has been lately, and keeps
#           the blend honest where the 6-hour empirical sample is thin.
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
# The standard normal's quantile at each level, to 6 decimals.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# Anchor horizons (minutes) at which the empirical change quantiles are taken.
ANCHORS = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]
# How much recent history feeds the empirical quantiles and the per-market
# sigma, in minutes. Six hours: long enough to estimate the shape, short
# enough to track the current regime.
HISTORY_MINUTES = 360
# Blend weight of the empirical shape (the rest is the Gaussian).
BLEND_W = 0.5
# Scale applied to the per-market Gaussian sigma.
SIGMA_SCALE = 0.7
# Floor and cap on the per-market 1-minute change sd, in price units.
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.01


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
    """Per-market Gaussian sigma: the sd of 1-minute changes over the last
    HISTORY_MINUTES prices, floored, capped, and scaled."""
    seg = prices[-HISTORY_MINUTES:] if len(prices) > HISTORY_MINUTES else prices
    if len(seg) < 2:
        return SIGMA_FLOOR * SIGMA_SCALE
    d = [seg[t + 1] - seg[t] for t in range(len(seg) - 1)]
    mean = sum(d) / len(d)
    var = sum((v - mean) ** 2 for v in d) / len(d)
    sd = math.sqrt(var)
    return min(SIGMA_CAP, max(SIGMA_FLOOR, sd)) * SIGMA_SCALE


def forecast_market(prices, horizon):
    """The 240 rows of five quantiles for one market, around its last price."""
    x = prices[-1]
    emp = market_empirical(prices)
    sig = market_sigma(prices)
    rows = []
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        # The empirical median change at this horizon, computed once: the
        # blend is centred on it, so the median forecast stays the last price.
        med = interp_logh(ANCHORS, [emp[a][2] for a in range(len(ANCHORS))], h)
        row = []
        for j in range(5):
            q = interp_logh(ANCHORS, [emp[a][j] for a in range(len(ANCHORS))], h)
            q = q - med  # centre on the last price: location is persistence
            g = Z[j] * sig * sq
            row.append(x + (1.0 - BLEND_W) * g + BLEND_W * q)
        # clamp to [0, 1], non-decreasing, 6 decimals
        row = [price6(v) for v in row]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


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
        horizon = case["horizon_minutes"]
        forecasts = []
        for m in case["markets"]:
            prices = m["prices"]
            if not prices:
                raise ValueError(f"market {m['id']} has no prices")
            forecasts.append({"id": m["id"], "q": forecast_market(prices, horizon)})
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
